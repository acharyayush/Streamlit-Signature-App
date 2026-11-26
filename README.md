# Streamlit-Signature-App

Offline signature verification web app (Streamlit + PyTorch ConvNeXt-Tiny).

Self-contained Streamlit app for offline signature verification. It compares up to
**2 reference** signatures (known genuine) with up to **4 test** signatures, so up to
**8 pairs** are scored at once, and reports the result for each pair.

Two global scoring modes are available in the sidebar:

* **deep** - cosine similarity of the two 512-d L2-normalized embeddings
  (adjustable threshold, default **0.82**).
* **fused** - a weighted blend of the deep score and nine handcrafted
  geometric/structural features (`0.7 * deep + 0.3 * hand` by default), using the
  weights in `fusion.json` (adjustable threshold, default **0.80**).

Both scores are always shown for every pair; the selected mode drives the
per-pair decision and the score/decision matrices.

This folder can be pushed on its own; it has no imports from the wider project.

## Contents

| Path | Purpose |
|---|---|
| `streamlit_app.py` | The app (entry point). |
| `model.py` | Inference-only ConvNeXt-Tiny encoder (`SiameseSignatureModel`). |
| `preprocessing/signature_preprocessing.py` | Grayscale Stage 2.2 preprocessing. |
| `scoring/fusion.py` | Fusion config + `FusionScorer` (weighted / logistic rules). |
| `scoring/handcrafted_features.py` | The nine geometric/structural pair features. |
| `scoring/metrics.py` | FAR / FRR / EER helpers used by the fusion code. |
| `fusion.json` | Fitted fusion weights, features and decision threshold. |
| `best_model.pt` | Trained weights (~114 MB, tracked with Git LFS). |
| `requirements.txt` | Runtime dependencies. |
| `.streamlit/config.toml` | Theme, upload size, headless server settings. |
| `.gitattributes` | Keeps `best_model.pt` in Git LFS. |

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the printed URL (default http://localhost:8501) and upload a reference image
and a test image.

## Deploy (Streamlit Community Cloud)

1. Push this folder as its own repository (Git LFS must be enabled so
   `best_model.pt` is uploaded as real weights, not a pointer).
2. On https://share.streamlit.io create an app, select the repository, and set
   **Main file path** to `streamlit_app.py`.
3. Deploy. Streamlit installs `requirements.txt` and reads `.streamlit/config.toml`.

If the logs show an LFS/JSON error for `best_model.pt`, run
`git lfs install && git lfs pull` locally, commit the resolved file, and redeploy.
The app detects an un-downloaded LFS pointer and shows a clear error.

Note: the folder name contains a space. If a hosting tool rejects the path, rename
the folder (for example to `streamlit_app`) or set the main file path with the
space quoted.

## What the app reports

- **Results table** - one row per `(reference, test)` pair with the deep
  similarity, the fused score, and both decisions.
- **Score / decision matrices** - references as rows and tests as columns for the
  active mode, so the whole grid is visible at a glance.
- **Pair details** - an expandable per-pair view with the original and preprocessed
  images and every score.
- **Match percentage** - the active score x 100 (a presentation of the score, not
  a calibrated probability).
- **Decision** - `MATCH` when the active score is at or above its threshold, with a
  borderline warning within `0.03` of the threshold.
- Plus checkpoint metadata.

Both thresholds are adjustable in the sidebar. Validate them on your own data
before production use; the scores are not probabilities.

## Provenance

`model.py` and `preprocessing/signature_preprocessing.py` are copies of the
corresponding modules in the main project
(`training/stage3_metric_learning.py` and
`preprocessing/signature_preprocessing.py`). The `scoring/` package is a copy of
the main project's `scoring/` package, and `fusion.json` is the fitted fusion
config. Only the inference paths are needed, so the app does not pull in the
training code or its dependencies (no scikit-learn at runtime).

The fused threshold stored in `fusion.json` (`0.8689`) is deliberately
conservative - it was calibrated for a low skilled-forgery false-accept rate, so
at that value some genuine pairs that the deep score accepts are rejected by the
fused rule. The app therefore defaults the fused threshold to **0.80**, which
trades some false accepts for fewer false rejects. Validate the threshold on your
own data.
