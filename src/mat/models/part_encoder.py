from __future__ import annotations

from typing import Any
import hashlib
import numpy as np

from mat.core.errors import DependencyUnavailableError, ValidationError
from mat.core.types import DescriptorBatch, SpeciesSpec


try:  # optional until a verified local MegaDescriptor asset is imported
    import torch
    from torch import nn
except Exception:  # pragma: no cover - exercised on the lightweight audit host
    torch = None
    nn = object


class PartAwareIdentityEncoder(nn.Module if torch is not None else object):
    """Differentiable projection/gating head over a caller-supplied frozen encoder.

    The class does not download or construct a pretrained backbone. ``backbone`` must
    be supplied from a verified local checkpoint. Its ``forward`` remains differentiable;
    ``encode`` is the explicitly frozen convenience API.
    """

    def __init__(self, backbone: Any, global_dim: int, part_dim: int,
                 species: SpeciesSpec, alpha: float = 0.5):
        if torch is None:
            raise DependencyUnavailableError("PyTorch is required for PartAwareIdentityEncoder")
        super().__init__()
        if global_dim <= 0 or part_dim <= 0 or not species.part_groups:
            raise ValidationError("invalid encoder dimensions/species")
        self.backbone = backbone
        self.species = species
        self.alpha = float(alpha)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.global_projection = nn.Linear(global_dim, part_dim)
        self.part_gates = nn.Parameter(torch.ones(len(species.part_groups)))
        self.encoder_version = hashlib.sha256(
            f"{type(backbone).__module__}.{type(backbone).__qualname__}:{global_dim}:{part_dim}:{species.skeleton_version}".encode()
        ).hexdigest()[:16]

    def _base(self, images):
        base = self.backbone(images)
        if isinstance(base, (tuple, list)):
            base = base[0]
        if base.ndim == 4:
            base = base.mean(dim=(2, 3))
        elif base.ndim == 3:
            base = base.mean(dim=1)
        elif base.ndim != 2:
            base = base.flatten(1)
        return base

    def forward(self, images, keypoints, valid, masks=None):
        if torch is None:  # pragma: no cover
            raise DependencyUnavailableError("PyTorch unavailable")
        base = self._base(images)
        global_feature = torch.nn.functional.normalize(self.global_projection(base), dim=-1)
        # The first implementation uses the same frozen crop descriptor with explicit
        # validity gates. A dense crop implementation can replace this without changing
        # the output contract.
        parts = global_feature.unsqueeze(1).expand(-1, len(self.species.part_groups), -1)
        gates = torch.sigmoid(self.part_gates).view(1, -1, 1)
        parts = torch.nn.functional.normalize(parts * gates, dim=-1)
        part_valid = torch.stack([
            valid[:, list(indices)].all(dim=1) for indices in self.species.part_groups.values()
        ], dim=1)
        part_quality = part_valid.float()
        return global_feature, parts, part_valid, part_quality, self.encoder_version

    def encode(self, images, keypoints, valid, masks=None) -> DescriptorBatch:
        self.eval()
        with torch.no_grad():
            outputs = self.forward(images, keypoints, valid, masks)
        return DescriptorBatch(*(x.detach().cpu().numpy() if hasattr(x, "detach") else x for x in outputs[:-1]), outputs[-1])
