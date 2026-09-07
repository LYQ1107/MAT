from __future__ import annotations

from pathlib import Path

from .common import inspect_files, build_common_manifests
from mat.data.base import DatasetInventory, ManifestBundle
from mat.core.types import SessionSpec


class MultiCamCowsAdapter:
    dataset_name = "multicamcows2024"

    def inspect(self, raw_root: Path) -> DatasetInventory:
        return inspect_files(self.dataset_name, raw_root,
                             {"has_original_video": None, "has_local_track_ids": None,
                              "has_verified_persistent_ids": None, "has_pose_ground_truth": False,
                              "has_camera_calibration": None, "can_evaluate_A_end_to_end": None,
                              "can_evaluate_H_end_to_end": None, "can_evaluate_longitudinal_pose": False},
                             ["crop→original CCTV frame mapping and identity visibility must be verified",
                              "same event across cameras must stay in one split"])

    @staticmethod
    def _truth(path: Path, ordinal: int):
        return {"gt_id": path.parent.name, "provenance": "provider_path_private"}

    def build_manifests(self, raw_root: Path, output_root: Path) -> ManifestBundle:
        inv = self.inspect(raw_root)
        return build_common_manifests(self.dataset_name, raw_root, output_root, inv, self._truth)

    def iter_observations(self, session: SessionSpec):
        return iter(())

