# Streamlit-Signature-App

Offline signature verification web app (Streamlit + PyTorch ConvNeXt-Tiny).

Self-contained Streamlit app for offline signature verification. It compares a
known genuine reference signature with a test signature and applies an adjustable
cosine-similarity threshold (default **0.82**).

This folder can be pushed on its own; it has no imports from the wider project.

## Contents

| Path | Purpose |
|---|---|
| `streamlit_app.py` | The app (entry point). |
| `model.py` | Inference-only ConvNeXt-Tiny encoder (`SiameseSignatureModel`). |
| `preprocessing/signature_preprocessing.py` | Grayscale Stage 2.2 preprocessing. |
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

- **Match percentage** - cosine similarity x 100 (a presentation of the score,
  not a calibrated probability).
- **Cosine similarity** - the raw score in `[-1, 1]`.
- **Decision** - `MATCH` when the similarity is at or above the threshold, with a
  borderline warning near the threshold.
- Plus threshold margin, embedding distance, and checkpoint metadata.

The threshold is adjustable in the sidebar. Validate it on your own data before
production use; the scores are not probabilities.

## Provenance

`model.py` and `preprocessing/signature_preprocessing.py` are copies of the
corresponding modules in the main project
(`training/stage3_metric_learning.py` and
`preprocessing/signature_preprocessing.py`). Only the encoder class is kept, so
the app does not pull in the training code or its dependencies.
