"""Pose-guided identity crops and the part-aware identity encoder.

The identity backbone is deliberately supplied by the caller.  MAT never
downloads a checkpoint from inside this module and never receives identity
labels in the forward path.  ``PosePartCropper`` is kept separate so its ROI
geometry can be audited independently from the trainable matching head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import hashlib

import numpy as np

from mat.core.errors import DependencyUnavailableError, ValidationError
from mat.core.types import DescriptorBatch, SpeciesSpec


try:  # Optional on the lightweight audit interpreter.
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - exercised when torch is unavailable
    torch = None
    nn = object
    F = None


@dataclass(frozen=True)
class CropBatch:
    """ROI tensors and auditable quality values produced by the cropper."""

    crops: Any
    part_valid: Any
    part_quality: Any
    rois_xyxy: Any


class PosePartCropper:
    """Construct one animal ROI and one ROI per pose part.

    Coordinates are absolute ``xyxy`` pixels in the input image.  Invalid
    parts still receive a deterministic fallback crop (the animal ROI), but
    are marked invalid and have zero quality; a zero-filled image is never
    presented as valid evidence.
    """

    def __init__(
        self,
        species: SpeciesSpec,
        *,
        output_size: int = 224,
        keypoint_threshold: float = 0.2,
        min_part_points: int = 1,
        part_extent_scale: float = 1.5,
        global_padding: float = 0.10,
    ) -> None:
        if torch is None:
            raise DependencyUnavailableError("PyTorch is required for PosePartCropper")
        if output_size <= 0 or min_part_points <= 0:
            raise ValidationError("output_size/min_part_points must be positive")
        if not 0.0 <= keypoint_threshold <= 1.0:
            raise ValidationError("keypoint_threshold must be in [0,1]")
        if part_extent_scale < 1.0 or global_padding < 0.0:
            raise ValidationError("invalid ROI padding")
        self.species = species
        self.output_size = int(output_size)
        self.keypoint_threshold = float(keypoint_threshold)
        self.min_part_points = int(min_part_points)
        self.part_extent_scale = float(part_extent_scale)
        self.global_padding = float(global_padding)
        self.part_names = tuple(species.part_groups)

    @staticmethod
    def _as_batched(value: Any, ndim: int, name: str):
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if tensor.ndim == ndim - 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != ndim:
            raise ValidationError(f"{name} must have {ndim - 1} or {ndim} dimensions")
        return tensor

    @staticmethod
    def _finite_box(box: Any, height: int, width: int):
        box = torch.as_tensor(box, dtype=torch.float32)
        if box.numel() != 4 or not torch.isfinite(box).all():
            return torch.tensor([0.0, 0.0, float(width), float(height)], dtype=torch.float32)
        x1, y1, x2, y2 = [float(x) for x in box]
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

    def _global_roi(self, box: Any, height: int, width: int) -> torch.Tensor:
        x1, y1, x2, y2 = self._finite_box(box, height, width)
        bw, bh = max(float(x2 - x1), 1.0), max(float(y2 - y1), 1.0)
        px, py = bw * self.global_padding, bh * self.global_padding
        return torch.tensor(
            [max(0.0, float(x1 - px)), max(0.0, float(y1 - py)),
             min(float(width), float(x2 + px)), min(float(height), float(y2 + py))],
            dtype=torch.float32,
        )

    def _part_roi(
        self,
        group_indices: tuple[int, ...],
        keypoints: torch.Tensor,
        scores: torch.Tensor,
        valid: torch.Tensor,
        global_roi: torch.Tensor,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, bool, float]:
        points = keypoints[list(group_indices)]
        point_scores = scores[list(group_indices)]
        point_valid = valid[list(group_indices)].bool()
        finite = torch.isfinite(points).all(dim=-1)
        candidate = finite & point_valid & torch.isfinite(point_scores) & (point_scores >= self.keypoint_threshold)
        count = int(candidate.sum().item())
        if count:
            selected = points[candidate]
            selected_scores = point_scores[candidate].clamp(0.0, 1.0)
            min_xy = selected.min(dim=0).values
            max_xy = selected.max(dim=0).values
            cx, cy = (min_xy + max_xy) / 2.0
            x_extent = max(float(max_xy[0] - min_xy[0]), 1.0)
            y_extent = max(float(max_xy[1] - min_xy[1]), 1.0)
            animal_w = max(float(global_roi[2] - global_roi[0]), 1.0)
            animal_h = max(float(global_roi[3] - global_roi[1]), 1.0)
            min_side = 0.25 * max(animal_w, animal_h)
            half_w = 0.5 * max(x_extent, min_side) * self.part_extent_scale
            half_h = 0.5 * max(y_extent, min_side) * self.part_extent_scale
            roi = torch.tensor([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dtype=torch.float32)
            inside = ((selected[:, 0] >= 0) & (selected[:, 0] <= width) &
                      (selected[:, 1] >= 0) & (selected[:, 1] <= height)).float().mean()
            visible_fraction = count / max(len(group_indices), 1)
            mean_score = float(selected_scores.mean().item())
            quality = float(torch.clamp(selected_scores.mean() * visible_fraction * inside, 0.0, 1.0).item())
            is_valid = count >= self.min_part_points
        else:
            # A deterministic, non-zero fallback crop is safe because validity
            # and quality remain false/zero and downstream masking excludes it.
            roi = global_roi.clone()
            quality = 0.0
            is_valid = False
        roi[0::2].clamp_(0.0, float(width))
        roi[1::2].clamp_(0.0, float(height))
        if float(roi[2] - roi[0]) < 1.0:
            roi[2] = min(float(width), roi[0] + 1.0)
        if float(roi[3] - roi[1]) < 1.0:
            roi[3] = min(float(height), roi[1] + 1.0)
        return roi, is_valid, quality

    def build_rois(
        self,
        boxes_xyxy: Any,
        keypoints: Any,
        keypoint_scores: Any,
        keypoint_valid: Any,
        *,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        boxes = self._as_batched(boxes_xyxy, 2, "boxes_xyxy").detach().float()
        points = self._as_batched(keypoints, 3, "keypoints").detach().float()
        scores = self._as_batched(keypoint_scores, 2, "keypoint_scores").detach().float()
        valid = self._as_batched(keypoint_valid, 2, "keypoint_valid").detach().bool()
        batch = boxes.shape[0]
        if points.shape[0] != batch or scores.shape != points.shape[:1] + points.shape[1:2] or valid.shape != scores.shape:
            raise ValidationError("boxes/keypoints/scores/valid batch shapes disagree")
        if points.shape[1] != len(self.species.keypoint_names):
            raise ValidationError("keypoints do not match species skeleton")
        rois: list[torch.Tensor] = []
        part_valid: list[list[bool]] = []
        part_quality: list[list[float]] = []
        for i in range(batch):
            global_roi = self._global_roi(boxes[i], height, width)
            rois.append(global_roi)
            row_valid: list[bool] = []
            row_quality: list[float] = []
            for indices in self.species.part_groups.values():
                roi, is_valid, quality = self._part_roi(indices, points[i], scores[i], valid[i], global_roi, height, width)
                rois.append(roi)
                row_valid.append(is_valid)
                row_quality.append(quality)
            part_valid.append(row_valid)
            part_quality.append(row_quality)
        return torch.stack(rois), torch.tensor(part_valid, dtype=torch.bool), torch.tensor(part_quality, dtype=torch.float32)

    def __call__(self, images: torch.Tensor, boxes_xyxy: Any, keypoints: Any,
                 keypoint_scores: Any, keypoint_valid: Any) -> CropBatch:
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] not in (1, 3):
            raise ValidationError("images must be [B,C,H,W]")
        _, _, height, width = images.shape
        rois, part_valid, part_quality = self.build_rois(
            boxes_xyxy, keypoints, keypoint_scores, keypoint_valid,
            height=height, width=width,
        )
        batch = images.shape[0]
        roi_indices = torch.arange(batch, device=images.device).repeat_interleave(1 + len(self.part_names))
        rois_with_indices = torch.cat([roi_indices[:, None].float(), rois.to(images.device)], dim=1)
        try:
            from torchvision.ops import roi_align
        except Exception as exc:  # pragma: no cover - optional dependency
            raise DependencyUnavailableError("torchvision.ops.roi_align is required for pose crops") from exc
        crops = roi_align(images, rois_with_indices, output_size=(self.output_size, self.output_size), spatial_scale=1.0, aligned=True)
        return CropBatch(crops, part_valid.to(images.device), part_quality.to(images.device), rois.to(images.device))


class PartAwareIdentityEncoder(nn.Module if torch is not None else object):
    """Shared frozen backbone over a global ROI and pose-derived part ROIs."""

    def __init__(self, backbone: Any, global_dim: int, part_dim: int,
                 species: SpeciesSpec, alpha: float = 0.5,
                 *, cropper: PosePartCropper | None = None):
        if torch is None:
            raise DependencyUnavailableError("PyTorch is required for PartAwareIdentityEncoder")
        super().__init__()
        if global_dim <= 0 or part_dim <= 0 or not species.part_groups:
            raise ValidationError("invalid encoder dimensions/species")
        self.backbone = backbone
        self.species = species
        self.alpha = float(alpha)
        if hasattr(self.backbone, "parameters"):
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
        self.global_projection = nn.Linear(global_dim, part_dim)
        self.part_gates = nn.Parameter(torch.ones(len(species.part_groups)))
        self.cropper = cropper or PosePartCropper(species)
        settings = f"{self.cropper.output_size}:{self.cropper.keypoint_threshold}:{self.cropper.min_part_points}"
        self.encoder_version = hashlib.sha256(
            f"{type(backbone).__module__}.{type(backbone).__qualname__}:{global_dim}:{part_dim}:{species.skeleton_version}:{settings}".encode()
        ).hexdigest()[:16]

    def _base(self, images):
        base = self.backbone(images)
        if isinstance(base, (tuple, list)):
            base = base[0]
        if not isinstance(base, torch.Tensor):
            base = torch.as_tensor(base, device=images.device)
        if base.ndim == 4:
            base = base.mean(dim=(2, 3))
        elif base.ndim == 3:
            base = base.mean(dim=1)
        elif base.ndim != 2:
            base = base.flatten(1)
        return base

    def forward(self, images, boxes_xyxy, keypoints, keypoint_scores, keypoint_valid, masks=None):
        if torch is None:  # pragma: no cover
            raise DependencyUnavailableError("PyTorch unavailable")
        # Identity training must not move pose coordinates or boxes.
        boxes_xyxy = boxes_xyxy.detach() if isinstance(boxes_xyxy, torch.Tensor) else torch.as_tensor(boxes_xyxy).detach()
        keypoints = keypoints.detach() if isinstance(keypoints, torch.Tensor) else torch.as_tensor(keypoints).detach()
        crops = self.cropper(images, boxes_xyxy, keypoints, keypoint_scores, keypoint_valid)
        batch = images.shape[0]
        regions = 1 + len(self.species.part_groups)
        features = self._base(crops.crops).reshape(batch, regions, -1)
        features = F.normalize(self.global_projection(features), dim=-1)
        global_feature = features[:, 0]
        gates = torch.sigmoid(self.part_gates).view(1, -1, 1)
        part_features = F.normalize(features[:, 1:] * gates, dim=-1)
        return global_feature, part_features, crops.part_valid, crops.part_quality, self.encoder_version

    def encode(self, images, boxes_xyxy, keypoints, keypoint_scores, keypoint_valid, masks=None) -> DescriptorBatch:
        self.eval()
        with torch.no_grad():
            outputs = self.forward(images, boxes_xyxy, keypoints, keypoint_scores, keypoint_valid, masks)
        return DescriptorBatch(*(x.detach().cpu().numpy() if hasattr(x, "detach") else x for x in outputs[:-1]), outputs[-1])
