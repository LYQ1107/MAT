from __future__ import annotations

from pathlib import Path
import configparser
import re

from .common import inspect_files, build_common_manifests, iter_session_frames
from mat.data.base import DatasetInventory, ManifestBundle
from mat.core.types import SessionSpec


class PigTrackingAdapter:
    dataset_name = "pig_tracking"

    def inspect(self, raw_root: Path) -> DatasetInventory:
        inv = inspect_files(self.dataset_name, raw_root,
                            {"has_original_video": True, "has_local_track_ids": True,
                             "has_verified_persistent_ids": None, "has_pose_ground_truth": False,
                             "has_camera_calibration": False, "can_evaluate_A_end_to_end": None,
                             "can_evaluate_H_end_to_end": None, "can_evaluate_longitudinal_pose": False},
                            ["seqinfo/gt/mot_labels and localID→EID persistence require provider audit",
                             "1/2/5 FPS are separate sampling conditions; do not merge as frames"])
        for seqinfo in raw_root.rglob("seqinfo.ini"):
            parser = configparser.ConfigParser()
            try:
                parser.read(seqinfo)
                inv.sessions += 1
            except configparser.Error as exc:
                inv.parse_failures.append({"path": str(seqinfo), "reason": str(exc)})
        return inv

    def build_manifests(self, raw_root: Path, output_root: Path) -> ManifestBundle:
        inv = self.inspect(raw_root)
        bundle = build_common_manifests(self.dataset_name, raw_root, output_root, inv, None)
        inv.notes.append("No persistent ID truth is inferred from gt.txt; evaluator remains blocked until mapping evidence")
        inv.write(output_root / "inventory.json")
        return bundle

    def iter_observations(self, session: SessionSpec):
        return iter_session_frames(session)
