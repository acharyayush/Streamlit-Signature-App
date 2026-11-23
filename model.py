"""Inference-only encoder used by the Streamlit app.

This is the ``SiameseSignatureModel`` class from
``training/stage3_metric_learning.py`` in the main project, trimmed to the parts
needed for inference so the app does not depend on the training package.

The architecture must stay in sync with the checkpoint: ``best_model.pt`` stores
``model_state_dict``, ``embedding_dim`` (512), ``phase``, ``epoch``, ``seed`` and
``preprocessing``.
"""

from __future__ import annotations

import timm
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SiameseSignatureModel(nn.Module):
    """Map each signature to a normalized embedding using a shared encoder.

    ConvNeXt extracts visual features. The projection head maps them to the
    embedding space, and L2 normalization makes cosine and Euclidean distances
    useful for comparing signatures.
    """

    def __init__(self, *, pretrained: bool = True, embedding_dim: int = 512) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.projection = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            # LayerNorm acts per sample, independent of PK batch composition.
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, embedding_dim),
        )
        self.embedding_dim = embedding_dim

    def forward(self, images: Tensor) -> Tensor:
        """Encode a batch and return one unit-length embedding per image."""
        return F.normalize(self.projection(self.backbone(images)), p=2, dim=1)

    def compare(self, reference: Tensor, query: Tensor) -> Tensor:
        """Return pairwise cosine similarities; these are not calibrated probabilities."""
        reference_embedding = self(reference)
        query_embedding = self(query)
        return F.cosine_similarity(reference_embedding, query_embedding, dim=1)
