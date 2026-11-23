"""Streamlit UI for verifying a signature against a genuine reference.

Run from this directory:

    streamlit run streamlit_app.py

The app loads ``best_model.pt`` from the project directory, preprocesses the two
uploaded images with the same grayscale Stage 2.2 pipeline used in training, and
compares their 512-dimensional L2-normalized embeddings with cosine similarity.

Reported values:

* Cosine similarity - the raw model score in ``[-1, 1]``.
* Match percentage - the cosine similarity scaled to ``[0, 100]`` (``similarity x 100``).
  This is a convenient presentation of the score, not a calibrated probability.
* Decision - ``MATCH`` when the similarity is at or above the threshold (default 0.82).

The threshold is adjustable in the sidebar so a deployment can trade false accepts
against false rejects. It must be validated on representative data before use.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
import torch

from model import SiameseSignatureModel
from preprocessing.signature_preprocessing import preprocess_signature, to_model_input

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "best_model.pt"
# Cosine threshold: 0.82 sits between genuine pairs (~0.90) and skilled forgeries
# (~0.45) on the validation signers.
DEFAULT_THRESHOLD = 0.82
# A small band around the threshold is flagged for review instead of a hard call.
REVIEW_BAND = 0.03
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


def _embed(
    model: SiameseSignatureModel,
    image: np.ndarray,
    *,
    binarize: bool,
) -> tuple[torch.Tensor, np.ndarray]:
    """Preprocess one image and return its unit-length embedding and clean image."""
    processed = preprocess_signature(image, binarize=binarize)
    tensor = torch.from_numpy(to_model_input(processed)).unsqueeze(0)
    with torch.inference_mode():
        embedding = model(tensor).squeeze(0)
    return embedding, processed


def _match_percentage(similarity: float) -> float:
    """Scale a cosine similarity to a 0-100 match percentage (similarity x 100)."""
    return float(np.clip(similarity, 0.0, 1.0) * 100.0)


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
    "Upload a known genuine reference signature and a test signature. The model "
    "compares their learned embeddings and applies a cosine-similarity threshold."
)

with st.sidebar:
    st.header("Verification settings")
    threshold = st.slider(
        "Cosine threshold",
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
    st.caption(
        f"Default threshold: **{DEFAULT_THRESHOLD}** - a value at or above "
        f"{threshold * 100:.0f}% match is a MATCH."
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

uploaded_reference = st.file_uploader(
    "Reference image (known genuine signature)",
    type=IMAGE_TYPES,
    key="reference",
)
uploaded_test = st.file_uploader(
    "Test image (signature to verify)",
    type=IMAGE_TYPES,
    key="test",
)

if uploaded_reference is None or uploaded_test is None:
    st.info("Upload both images to run verification.")
    st.stop()

try:
    checkpoint_data = _checkpoint_bytes()
    model, checkpoint = _load_model(checkpoint_data)
    reference_image = _read_image(uploaded_reference, "Reference image")
    test_image = _read_image(uploaded_test, "Test image")
    reference_embedding, reference_processed = _embed(
        model, reference_image, binarize=binarize
    )
    test_embedding, test_processed = _embed(model, test_image, binarize=binarize)
except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
    st.error(str(error))
    st.stop()


similarity = float(torch.dot(reference_embedding, test_embedding).item())
match_percentage = _match_percentage(similarity)
threshold = float(threshold)
margin = similarity - threshold
distance = float(np.sqrt(max(0.0, 2.0 - 2.0 * similarity)))
decision = "MATCH" if similarity >= threshold else "NO MATCH"

st.subheader("Verification result")
if decision == "MATCH":
    st.success(
        f"MATCH - {match_percentage:.2f}% match (cosine similarity {similarity:.4f})"
    )
elif abs(margin) <= REVIEW_BAND:
    st.warning(
        f"BORDERLINE - {match_percentage:.2f}% match is within {REVIEW_BAND:.2f} of the "
        f"threshold ({threshold:.4f}). Manual review is recommended."
    )
else:
    st.error(
        f"NO MATCH - {match_percentage:.2f}% match (cosine similarity {similarity:.4f})"
    )

st.progress(
    min(1.0, match_percentage / 100.0),
    text=f"Match percentage: {match_percentage:.2f}% (threshold = {threshold * 100:.2f}%)",
)

metrics = [
    ("Decision", decision),
    ("Match percentage", f"{match_percentage:.2f}%"),
    ("Cosine similarity", f"{similarity:.4f}"),
    ("Threshold", f"{threshold:.4f}"),
    ("Margin above threshold", f"{margin:+.4f}"),
    ("Embedding Euclidean distance", f"{distance:.4f}"),
]
metric_columns = st.columns(3)
for index, (name, value) in enumerate(metrics):
    with metric_columns[index % 3]:
        st.metric(name, value)

columns = st.columns(2)
with columns[0]:
    st.image(reference_image, caption="Reference image", channels="BGR")
    st.image(reference_processed, caption="Reference after preprocessing", clamp=True)
with columns[1]:
    st.image(test_image, caption="Test image", channels="BGR")
    st.image(test_processed, caption="Test after preprocessing", clamp=True)

st.info(
    "The cosine similarity and match percentage are model scores, not calibrated "
    "probabilities or proof of authenticity. Validate the threshold for your "
    "deployment data and keep manual review and institution-approved controls."
)

with st.expander("Model details"):
    details = _format_metadata(checkpoint)
    st.write(
        {
            "checkpoint": str(DEFAULT_CHECKPOINT.relative_to(PROJECT_ROOT)),
            "checkpoint_sha256": hashlib.sha256(checkpoint_data).hexdigest(),
            "input_image_shape": str(tuple(reference_image.shape)),
            **details,
            "checkpoint_preprocessing": checkpoint.get("preprocessing", {}),
        }
    )
    st.caption(
        "Match percentage = cosine similarity x 100. Embeddings are L2-normalized, "
        "so the dot product is the cosine similarity."
    )



