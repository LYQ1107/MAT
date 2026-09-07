from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json
import numpy as np

from mat.core.errors import DependencyUnavailableError, MissingAssetError, ValidationError
from mat.core.types import SessionSpec, SpeciesSpec
from .base import PoseBackendBase, PoseCache


class DlcSuperAnimalBackend(PoseBackendBase):
    """Offline DLC adapter; no implicit model-zoo download is allowed."""

    def __init__(self, pose_checkpoint: Path | None, detector_checkpoint: Path | None,
                 model_config: Path | None, *, expected_hashes: dict[str, str] | None = None):
        super().__init__([p for p in (pose_checkpoint, detector_checkpoint, model_config) if p is not None])
        self.pose_checkpoint, self.detector_checkpoint, self.model_config = pose_checkpoint, detector_checkpoint, model_config
        self.expected_hashes = expected_hashes or {}

    def verify_assets(self) -> None:
        super().verify_assets()
        for label, path in (("pose", self.pose_checkpoint), ("detector", self.detector_checkpoint), ("config", self.model_config)):
            if path is None:
                raise MissingAssetError(f"{label} checkpoint/config is required")
            if label in self.expected_hashes:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != self.expected_hashes[label]:
                    raise MissingAssetError(f"{label} checkpoint hash mismatch")

    def predict_session(self, session: SessionSpec, species: SpeciesSpec, boxes=None) -> PoseCache:
        self.verify_assets()
        raise DependencyUnavailableError(
            "DLC inference integration is version-locked but not run: install the pinned offline DLC environment and supply verified checkpoints"
        )

    def predict_on_boxes(self, video, boxes, species: SpeciesSpec) -> PoseCache:
        self.verify_assets()
        raise DependencyUnavailableError("DLC predict_on_boxes requires the pinned offline DLC runtime")

