from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np

from mat.core.errors import MissingAssetError


@dataclass
class PoseCache:
    observation_uids: tuple[str, ...]
    keypoints_xy: np.ndarray
    keypoint_scores: np.ndarray
    coordinate_transform: dict[str, Any]
    backend_fingerprint: str


class PoseBackendBase:
    def __init__(self, assets: list[Path] | None = None):
        self.assets = assets or []

    def verify_assets(self) -> None:
        missing = [str(p) for p in self.assets if not p.is_file()]
        if missing:
            raise MissingAssetError("missing verified local pose assets: " + ", ".join(missing))

