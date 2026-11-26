"""Streamlit UI: compare every reference signature with every test signature.

Run from this directory:

    streamlit run streamlit_app.py

The app loads ``best_model.pt`` from the project directory, preprocesses every
uploaded image with the same grayscale Stage 2.2 pipeline used in training, and
compares each reference with each test image (2 references x 4 tests = 8 pairs).

Two global scoring options are offered:

* ``deep``  - cosine similarity of the two 512-d L2-normalized embeddings.
* ``fused`` - score-level fusion from ``scoring/fusion.py``: a weighted blend of the
  deep cosine score and the nine handcrafted geometric/structural features.
  ``fusion.json`` stores the fitted weights and decision threshold.

Reported values:

* Deep cosine similarity - the raw model score in ``[-1, 1]``.
* Fused score - ``w_deep * deep + w_hand * mean(hand)`` using the weights in
  ``fusion.json`` (default ``0.7 / 0.3``), in ``[0, 1]`` for the weighted rule.
* Match percentage - the active score scaled to ``[0, 100]`` (score x 100). This is
  a convenient presentation of the score, not a calibrated probability.
* Decision - ``MATCH`` when the active score is at or above its threshold
  (deep default ``0.82``, fused default ``0.80``).

The thresholds are adjustable in the sidebar so a deployment can trade false
accepts against false rejects. They must be validated on representative data.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

from model import SiameseSignatureModel
from preprocessing.signature_preprocessing import preprocess_signature, to_model_input
from scoring.fusion import FusionConfig, FusionScorer
from scoring.handcrafted_features import (
    SignatureDescriptors,
    describe_image,
    pair_features,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "best_model.pt"
DEFAULT_FUSION_CONFIG = PROJECT_ROOT / "fusion.json"
# Cosine threshold: 0.82 sits between genuine pairs (~0.90) and skilled forgeries
# (~0.45) on the validation signers.
DEFAULT_THRESHOLD = 0.82
# Fused-score decision threshold. 0.80 trades some false accepts for fewer false
# rejects than the conservative value stored in fusion.json (0.8689).
DEFAULT_FUSED_THRESHOLD = 0.80
# A small band around the threshold is flagged for review instead of a hard call.
REVIEW_BAND = 0.03
MAX_REFERENCES = 2
MAX_TESTS = 4
IMAGE_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"]
_LFS_HEADER = b"version https://git-lfs"


def _read_image(uploaded_file: Any, label: str) -> np.ndarray:
    """Decode an uploaded file into a BGR image, or raise a readable error."""
    data = uploaded_file.getvalue()
    if not data:
        raise ValueError(f"{label} is empty.")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"{label} could not be decoded as an image.")
    return image


def _checkpoint_bytes() -> bytes:
    """Return the checkpoint bytes, guarding against a missing file or LFS pointer."""
    if not DEFAULT_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"No checkpoint found at {DEFAULT_CHECKPOINT}. "
            "Place best_model.pt in the project directory."
        )
    data = DEFAULT_CHECKPOINT.read_bytes()
    if data.startswith(_LFS_HEADER):
        raise RuntimeError(
            "best_model.pt is an un-downloaded Git LFS pointer, not the real weights. "
            "Run `git lfs install && git lfs pull` (or copy the checkpoint here) first."
        )
    return data


def _fusion_config_bytes() -> bytes | None:
    """Return the fusion config bytes, or ``None`` when the file is absent/empty."""
    if not DEFAULT_FUSION_CONFIG.is_file():
        return None
    data = DEFAULT_FUSION_CONFIG.read_bytes()
    return data or None


@st.cache_resource(show_spinner="Loading trained model...")
def _load_model(checkpoint_data: bytes) -> tuple[SiameseSignatureModel, dict[str, Any]]:
    """Rebuild the encoder from the checkpoint once and reuse it across reruns."""
    checkpoint = torch.load(
        io.BytesIO(checkpoint_data), map_location="cpu", weights_only=True
    )
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("The checkpoint must contain a model_state_dict.")
    embedding_dim = int(checkpoint.get("embedding_dim", 512))
    model = SiameseSignatureModel(pretrained=False, embedding_dim=embedding_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@st.cache_resource(show_spinner="Loading fusion configuration...")
def _load_fusion(config_data: bytes) -> tuple[FusionConfig, FusionScorer]:
    """Rebuild the fusion rule from ``fusion.json`` once and reuse it across reruns."""
    config = FusionConfig.from_dict(json.loads(config_data.decode("utf-8")))
    return config, FusionScorer(config)


def _preprocess(image: np.ndarray, *, binarize: bool) -> np.ndarray:
    """Run the grayscale Stage 2.2 pipeline the model was trained on."""
    return preprocess_signature(image, binarize=binarize)


def _embed(model: SiameseSignatureModel, processed: np.ndarray) -> torch.Tensor:
    """Return the unit-length embedding of an already-preprocessed image."""
    tensor = torch.from_numpy(to_model_input(processed)).unsqueeze(0)
    with torch.inference_mode():
        embedding = model(tensor).squeeze(0)
    return embedding


def _describe(processed: np.ndarray) -> SignatureDescriptors | None:
    """Summarise a preprocessed image for the handcrafted features, or ``None``.

    ``describe_image`` raises when an image has no ink. Returning ``None`` lets the
    caller report the fused score as unavailable for that pair instead of failing the
    whole run.
    """
    try:
        return describe_image(processed)
    except ValueError:
        return None


def _match_percentage(score: float) -> float:
    """Scale a score to a 0-100 match percentage (score x 100)."""
    return float(np.clip(score, 0.0, 1.0) * 100.0)


def _decision(score: float, threshold: float) -> str:
    """Return the MATCH / NO MATCH call for ``score`` at ``threshold``."""
    return "MATCH" if score >= threshold else "NO MATCH"


def _format_metadata(checkpoint: dict[str, Any]) -> dict[str, str]:
    """Collect the human-readable checkpoint fields shown in the details panel."""
    loss = checkpoint.get("best_validation_loss")
    return {
        "phase": str(checkpoint.get("phase", "unknown")),
        "epoch": str(checkpoint.get("epoch", "unknown")),
        "seed": str(checkpoint.get("seed", "unknown")),
        "embedding_dim": str(checkpoint.get("embedding_dim", 512)),
        "best_validation_loss": f"{float(loss):.6f}" if loss is not None else "unknown",
    }


st.set_page_config(
    page_title="Signature Verification",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Offline Signature Verification")
st.caption(
    f"Upload up to {MAX_REFERENCES} known genuine reference signatures and up to "
    f"{MAX_TESTS} test signatures. Every reference is compared with every test image "
    f"(up to {MAX_REFERENCES * MAX_TESTS} pairs) with the deep embedding, the fused "
    "score, or both."
)

# The fusion config is loaded before the sidebar so the fitted weights are available.
fusion_error: str | None = None
fusion_config: FusionConfig | None = None
scorer: FusionScorer | None = None
try:
    config_data = _fusion_config_bytes()
    if config_data is None:
        raise FileNotFoundError(f"No fusion config found at {DEFAULT_FUSION_CONFIG}.")
    fusion_config, scorer = _load_fusion(config_data)
except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
    fusion_error = str(error)

with st.sidebar:
    st.header("Verification settings")
    mode = st.radio(
        "Scoring mode",
        options=["fused", "deep"],
        index=0,
        horizontal=True,
        help=(
            "deep: cosine similarity of the learned embeddings. "
            "fused: weighted blend of the deep score and the nine handcrafted "
            "geometric/structural features."
        ),
    )
    if mode == "fused" and scorer is None:
        st.error(f"Fusion is unavailable: {fusion_error}")
        st.stop()

    deep_threshold = st.slider(
        "Cosine threshold (deep)",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_THRESHOLD,
        step=0.01,
        help=(
            "Similarity at or above this value is a MATCH. Increasing it reduces "
            "false accepts but rejects more genuine signatures. "
            f"Default: {DEFAULT_THRESHOLD}."
        ),
    )
    fused_threshold = st.slider(
        "Fused-score threshold",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_FUSED_THRESHOLD,
        step=0.01,
        help=(
            "Threshold for the fused score. Lowering it accepts more genuine pairs "
            "at the cost of more false accepts. "
            f"Default: {DEFAULT_FUSED_THRESHOLD}."
        ),
    )

    active_threshold = fused_threshold if mode == "fused" else deep_threshold
    st.caption(
        f"Active mode: **{mode}** at threshold **{active_threshold:.4f}**. "
        "Both scores are still reported for every pair."
    )

    binarize = st.checkbox(
        "Use binary black/white preprocessing",
        value=False,
        help=(
            "The model is trained on grayscale strokes, so keep this unchecked to "
            "match the training/inference preprocessing."
        ),
    )
    if binarize:
        st.warning(
            "Binary preprocessing differs from the grayscale training path and may "
            "degrade results."
        )

    with st.expander("Preprocessing details"):
        st.write(
            {
                "mode": "binary black/white" if binarize else "grayscale",
                "ink": "auto (dark ink on white)",
                "target_size": "384 x 192 (width x height)",
                "input_channels": "3 (grayscale repeated, ImageNet-normalized)",
            }
        )

    if fusion_config is not None:
        with st.expander("Fusion configuration"):
            st.write(
                {
                    "config": str(DEFAULT_FUSION_CONFIG.relative_to(PROJECT_ROOT)),
                    "type": fusion_config.type,
                    "deep_weight": round(fusion_config.deep_weight, 4),
                    "hand_weight": round(fusion_config.hand_weight, 4),
                    "features": list(fusion_config.features),
                    "stored_threshold": fusion_config.threshold,
                }
            )
    else:
        st.info(f"Fusion config not loaded: {fusion_error}")



st.subheader("Upload images")
upload_columns = st.columns(2)
with upload_columns[0]:
    reference_files = st.file_uploader(
        f"Reference images (1-{MAX_REFERENCES} known genuine signatures)",
        type=IMAGE_TYPES,
        accept_multiple_files=True,
        key="references",
        help="Each reference is compared with every test image.",
    )
with upload_columns[1]:
    test_files = st.file_uploader(
        f"Test images (1-{MAX_TESTS} signatures to verify)",
        type=IMAGE_TYPES,
        accept_multiple_files=True,
        key="tests",
        help="Each test image is compared with every reference.",
    )

reference_files = list(reference_files or [])
test_files = list(test_files or [])
if not reference_files or not test_files:
    st.info("Upload at least one reference image and one test image to run verification.")
    st.stop()
if len(reference_files) > MAX_REFERENCES:
    st.error(
        f"Upload at most {MAX_REFERENCES} reference images (got {len(reference_files)})."
    )
    st.stop()
if len(test_files) > MAX_TESTS:
    st.error(f"Upload at most {MAX_TESTS} test images (got {len(test_files)}).")
    st.stop()

try:
    checkpoint_data = _checkpoint_bytes()
    model, checkpoint = _load_model(checkpoint_data)
except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
    st.error(str(error))
    st.stop()

# Preprocess and encode every uploaded image once; the pair grid then reuses the
# embeddings and handcrafted descriptors instead of recomputing them.
reference_images: list[np.ndarray] = []
reference_processed: list[np.ndarray] = []
reference_embeddings: list[torch.Tensor] = []
reference_descriptors: list[SignatureDescriptors | None] = []
for index, uploaded in enumerate(reference_files, start=1):
    label = f"Reference {index} ({uploaded.name})"
    try:
        image = _read_image(uploaded, label)
        processed = _preprocess(image, binarize=binarize)
    except (TypeError, ValueError) as error:
        st.error(f"{label}: {error}")
        st.stop()
    reference_images.append(image)
    reference_processed.append(processed)
    reference_embeddings.append(_embed(model, processed))
    reference_descriptors.append(_describe(processed))

test_images: list[np.ndarray] = []
test_processed: list[np.ndarray] = []
test_embeddings: list[torch.Tensor] = []
test_descriptors: list[SignatureDescriptors | None] = []
for index, uploaded in enumerate(test_files, start=1):
    label = f"Test {index} ({uploaded.name})"
    try:
        image = _read_image(uploaded, label)
        processed = _preprocess(image, binarize=binarize)
    except (TypeError, ValueError) as error:
        st.error(f"{label}: {error}")
        st.stop()
    test_images.append(image)
    test_processed.append(processed)
    test_embeddings.append(_embed(model, processed))
    test_descriptors.append(_describe(processed))

reference_labels = [f"R{index + 1}" for index in range(len(reference_files))]
test_labels = [f"T{index + 1}" for index in range(len(test_files))]

# One row per (reference, test) pair, always carrying both the deep and the fused
# score so the table is informative regardless of the selected global mode.
pairs: list[dict[str, Any]] = []
for reference_index in range(len(reference_files)):
    for test_index in range(len(test_files)):
        deep = float(
            torch.dot(reference_embeddings[reference_index], test_embeddings[test_index]).item()
        )
        fused: float | None = None
        if (
            scorer is not None
            and reference_descriptors[reference_index] is not None
            and test_descriptors[test_index] is not None
        ):
            features = pair_features(
                reference_descriptors[reference_index], test_descriptors[test_index]
            )
            fused = float(scorer.score(deep, features))
        pairs.append(
            {
                "reference_index": reference_index,
                "test_index": test_index,
                "deep": deep,
                "deep_decision": _decision(deep, deep_threshold),
                "fused": fused,
                "fused_decision": (
                    "N/A" if fused is None else _decision(fused, fused_threshold)
                ),
            }
        )

active_scores: list[float | None] = [
    pair["fused"] if mode == "fused" else pair["deep"] for pair in pairs
]
active_decisions: list[str] = [
    "N/A" if score is None else _decision(float(score), active_threshold)
    for score in active_scores
]


st.subheader("Comparison results")
match_count = sum(1 for decision in active_decisions if decision == "MATCH")
borderline_count = sum(
    1
    for score in active_scores
    if score is not None and abs(float(score) - active_threshold) <= REVIEW_BAND
)
st.info(
    f"Scoring mode **{mode}** at threshold **{active_threshold:.4f}** - "
    f"**{match_count}/{len(pairs)}** pairs matched. "
    f"{len(reference_labels)} reference x {len(test_labels)} test images."
)
if borderline_count:
    st.warning(
        f"{borderline_count} pair(s) sit within {REVIEW_BAND:.2f} of the threshold; "
        "manual review is recommended."
    )

table_rows = []
for pair, active_decision in zip(pairs, active_decisions):
    reference_index = pair["reference_index"]
    test_index = pair["test_index"]
    table_rows.append(
        {
            "Pair": f"{reference_labels[reference_index]} x {test_labels[test_index]}",
            "Reference": f"{reference_labels[reference_index]}: "
            f"{reference_files[reference_index].name}",
            "Test": f"{test_labels[test_index]}: {test_files[test_index].name}",
            "Deep similarity": round(pair["deep"], 4),
            "Deep decision": pair["deep_decision"],
            "Fused score": "N/A" if pair["fused"] is None else round(pair["fused"], 4),
            "Fused decision": pair["fused_decision"],
            f"Active decision ({mode})": active_decision,
        }
    )
st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)
st.caption(
    "Deep similarity = cosine similarity x 100 when read as a percentage. "
    "Fused score blends the deep score with the handcrafted features using fusion.json. "
    "N/A marks a pair where an image had no measurable ink."
)

st.subheader("Score matrix")
st.caption(
    f"Rows are references, columns are tests, cells hold the active ({mode}) score."
)
score_matrix = []
for reference_index, reference_label in enumerate(reference_labels):
    record: dict[str, Any] = {"Reference": reference_label}
    for test_index, test_label in enumerate(test_labels):
        score = active_scores[reference_index * len(test_labels) + test_index]
        record[test_label] = "N/A" if score is None else round(float(score), 4)
    score_matrix.append(record)
st.dataframe(pd.DataFrame(score_matrix), width="stretch", hide_index=True)

decision_matrix = []
for reference_index, reference_label in enumerate(reference_labels):
    record = {"Reference": reference_label}
    for test_index, test_label in enumerate(test_labels):
        decision = active_decisions[reference_index * len(test_labels) + test_index]
        record[test_label] = {
            "MATCH": "MATCH",
            "NO MATCH": "NO MATCH",
        }.get(decision, "N/A")
    decision_matrix.append(record)
st.dataframe(pd.DataFrame(decision_matrix), width="stretch", hide_index=True)

st.subheader("Pair details")
for position, pair in enumerate(pairs):
    reference_index = pair["reference_index"]
    test_index = pair["test_index"]
    active_score = active_scores[position]
    active_decision = active_decisions[position]
    heading = (
        f"{reference_labels[reference_index]} x {test_labels[test_index]} - "
        f"{active_decision}"
    )
    with st.expander(heading):
        image_columns = st.columns(2)
        with image_columns[0]:
            st.image(
                reference_images[reference_index],
                caption=f"Reference {reference_labels[reference_index]}: "
                f"{reference_files[reference_index].name}",
                channels="BGR",
            )
            st.image(
                reference_processed[reference_index],
                caption="Reference after preprocessing",
                clamp=True,
            )
        with image_columns[1]:
            st.image(
                test_images[test_index],
                caption=f"Test {test_labels[test_index]}: {test_files[test_index].name}",
                channels="BGR",
            )
            st.image(
                test_processed[test_index],
                caption="Test after preprocessing",
                clamp=True,
            )

        deep = pair["deep"]
        fused = pair["fused"]
        st.write(
            {
                "Deep cosine similarity": round(deep, 4),
                "Deep match percentage": f"{_match_percentage(deep):.2f}%",
                "Deep decision": f"{pair['deep_decision']} "
                f"(threshold {deep_threshold:.4f})",
                "Fused score": "N/A" if fused is None else round(fused, 4),
                "Fused decision": pair["fused_decision"]
                + ("" if fused is None else f" (threshold {fused_threshold:.4f})"),
                "Active mode": mode,
                "Active score": "N/A" if active_score is None else round(float(active_score), 4),
                "Active match percentage": (
                    "N/A"
                    if active_score is None
                    else f"{_match_percentage(float(active_score)):.2f}%"
                ),
            }
        )

st.info(
    "The scores are model outputs, not calibrated probabilities or proof of "
    "authenticity. Validate the thresholds for your deployment data and keep manual "
    "review and institution-approved controls."
)

with st.expander("Model details"):
    details = _format_metadata(checkpoint)
    st.write(
        {
            "checkpoint": str(DEFAULT_CHECKPOINT.relative_to(PROJECT_ROOT)),
            "checkpoint_sha256": hashlib.sha256(checkpoint_data).hexdigest(),
            "reference_images": len(reference_files),
            "test_images": len(test_files),
            "pairs": len(pairs),
            **details,
            "checkpoint_preprocessing": checkpoint.get("preprocessing", {}),
        }
    )
    st.caption(
        "Match percentage = score x 100. Embeddings are L2-normalized, so the dot "
        "product is the cosine similarity."
    )

