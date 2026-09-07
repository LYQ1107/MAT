from __future__ import annotations

from pathlib import Path
import re

from .common import inspect_files, build_common_manifests, records_to_frames
from mat.data.base import DatasetInventory, ManifestBundle
from mat.data.manifests import iter_manifest_records
from mat.core.types import SessionSpec


class RatIDAdapter:
    dataset_name = "rat_id"

    def inspect(self, raw_root: Path) -> DatasetInventory:
        return inspect_files(self.dataset_name, raw_root,
                             {"has_original_video": False, "has_local_track_ids": False,
                              "has_verified_persistent_ids": True, "has_pose_ground_truth": False,
                              "has_camera_calibration": False, "can_evaluate_A_end_to_end": False,
                              "can_evaluate_H_end_to_end": False, "can_evaluate_longitudinal_pose": False},
                             ["prelocalized crop identity diagnostic; archive must be indexed before extraction",
                              "week/date semantics are unresolved until metadata is parsed"])

    @staticmethod
    def _truth(path: Path, ordinal: int):
        # Private evaluator may use an EID-like directory label; never written to observations.
        parent = path.parent.name
        match = re.search(r"(?:rat|id|animal)[_-]?([A-Za-z0-9]+)", parent, re.I)
        return {"gt_id": match.group(1) if match else parent, "provenance": "provider_path_private"}

    def build_manifests(self, raw_root: Path, output_root: Path) -> ManifestBundle:
        inv = self.inspect(raw_root)
        return build_common_manifests(self.dataset_name, raw_root, output_root, inv, self._truth)

    def iter_observations(self, session: SessionSpec):
        return iter(())

