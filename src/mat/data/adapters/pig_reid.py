from __future__ import annotations

from pathlib import Path
import re

from .common import inspect_files, build_common_manifests
from mat.data.base import DatasetInventory, ManifestBundle
from mat.core.types import SessionSpec


class PigReIDAdapter:
    dataset_name = "pig_reid"

    def inspect(self, raw_root: Path) -> DatasetInventory:
        return inspect_files(self.dataset_name, raw_root,
                             {"has_original_video": False, "has_local_track_ids": False,
                              "has_verified_persistent_ids": True, "has_pose_ground_truth": False,
                              "has_camera_calibration": False, "can_evaluate_A_end_to_end": False,
                              "can_evaluate_H_end_to_end": False, "can_evaluate_longitudinal_pose": False},
                             ["single-animal preprocessed crops; directory EID is private truth only",
                              "group/date/camera mapping must be confirmed from provider metadata"])

    @staticmethod
    def _truth(path: Path, ordinal: int):
        parent = path.parent.name
        return {"gt_id": parent, "provenance": "provider_path_private"}

    def build_manifests(self, raw_root: Path, output_root: Path) -> ManifestBundle:
        inv = self.inspect(raw_root)
        return build_common_manifests(self.dataset_name, raw_root, output_root, inv, self._truth)

    def iter_observations(self, session: SessionSpec):
        return iter(())

