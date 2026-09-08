"""Adapter for the official SLEAP NYU gerbil package files.

The adapter deliberately keeps provider identity labels in a private truth
manifest.  ``observations.jsonl`` contains only neutral identifiers and image
references, so it is safe to pass to a model or an inference process.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np

from mat.core.errors import DependencyUnavailableError, ValidationError
from mat.data.base import DatasetInventory, ManifestBundle, dataset_uid
from mat.data.manifests import write_jsonl


SPLIT_FILES = {
    "train": "train.pkg.slp",
    "val": "val.pkg.slp",
    "test": "test.pkg.slp",
}
REQUIRED_FILES = {
    **SPLIT_FILES,
    "clip": "example_5min.mp4",
    "tracking": "example_tracking.slp",
}

CANONICAL_NODES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "spine1", "spine2", "spine3", "spine4", "spine5",
    "tail1", "tail2", "tail3", "tail4",
]
CANONICAL_EDGES = [
    ["spine1", "left_eye"], ["spine1", "left_ear"], ["spine1", "nose"],
    ["spine1", "right_eye"], ["spine1", "right_ear"], ["spine2", "spine1"],
    ["spine3", "spine2"], ["spine3", "spine4"], ["spine4", "spine5"],
    ["spine5", "tail1"], ["tail1", "tail2"], ["tail2", "tail3"],
    ["tail3", "tail4"],
]
PART_GROUPS = {
    "head": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "spine1"],
    "front_trunk": ["spine1", "spine2", "spine3"],
    "rear_trunk": ["spine3", "spine4", "spine5"],
    "tail_base": ["spine5", "tail1", "tail2"],
    "tail_distal": ["tail2", "tail3", "tail4"],
}


def _load_sleap_io():
    try:
        from sleap_io import load_slp
    except ImportError as exc:  # pragma: no cover - exercised on minimal installs
        # The normal MAT CLI runs on the small audit interpreter, while the
        # SLEAP wheels live under the work-root runtime.  Discover that local
        # path without downloading or inheriting any network configuration.
        import os
        import sys
        candidates = []
        work_root = os.environ.get("MAT_WORK_ROOT")
        if work_root:
            candidates.extend([
                Path(work_root) / "env" / "cli_runtime_overlay",
                Path(work_root) / "env" / "sleap_site",
                Path(work_root) / "env" / "sleap_py",
            ])
        for candidate in candidates:
            if candidate.is_dir() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
        try:
            from sleap_io import load_slp
        except ImportError as second_exc:
            raise DependencyUnavailableError(
                "sleap-io is required for the sleap_gerbils adapter; install it in the isolated runtime"
            ) from second_exc
    return load_slp


def _digest(value: str, length: int = 24) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _source_filename(video: Any) -> str:
    backend = getattr(video, "backend", None)
    source = getattr(backend, "source_filename", None) if backend is not None else None
    if source:
        return str(source)
    filename = getattr(video, "filename", None)
    return str(filename) if filename is not None else "unknown"


def _video_index(labels: Any, video: Any) -> int:
    """Resolve the public Video object to its stable index in ``labels.videos``."""
    for index, candidate in enumerate(labels.videos):
        if candidate is video or candidate == video:
            return index
    backend = getattr(video, "backend", None)
    dataset = str(getattr(backend, "dataset", ""))
    match = re.search(r"video(\d+)", dataset)
    if match:
        return int(match.group(1))
    raise ValidationError("SLEAP labeled frame references a video absent from labels.videos")


def _source_video_key(labels: Any, video: Any) -> tuple[str, int, str]:
    index = _video_index(labels, video)
    source_name = _source_filename(video)
    return f"{source_name}#video{index}", index, source_name


def _session_uid(source_video: str) -> str:
    return f"sleap_gerbils:session:{_digest(source_video, 20)}"


def _parse_datetime(source_name: str) -> str | None:
    match = re.search(r"(20\d{2})[-_](\d{1,2})[-_](\d{1,2})", source_name)
    if not match:
        return None
    try:
        return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3))).isoformat()
    except ValueError:
        return None


def _node_names(labels: Any) -> list[str]:
    names: list[str] = []
    for skeleton in getattr(labels, "skeletons", []):
        for node in getattr(skeleton, "nodes", []):
            name = str(getattr(node, "name", node))
            if name not in names:
                names.append(name)
    return names


def _track_name(instance: Any) -> str | None:
    track = getattr(instance, "track", None)
    name = getattr(track, "name", None) if track is not None else None
    return str(name) if name is not None else None


def _instance_truth(instance: Any) -> tuple[list[list[float | None]], list[bool]]:
    points = getattr(instance, "points", None)
    if points is None:
        array = np.asarray(instance.numpy(), dtype=float)
        visible = np.isfinite(array).all(axis=1)
        return (
            [[float(x), float(y)] if bool(ok) else [None, None]
             for (x, y), ok in zip(array, visible)],
            [bool(x) for x in visible],
        )
    xy = np.asarray(points["xy"], dtype=float)
    visible_field = np.asarray(points["visible"], dtype=bool)
    keypoints: list[list[float | None]] = []
    visibility: list[bool] = []
    for (x, y), visible in zip(xy, visible_field):
        valid = bool(visible) and np.isfinite(x) and np.isfinite(y)
        keypoints.append([float(x), float(y)] if valid else [None, None])
        visibility.append(bool(valid))
    return keypoints, visibility


class SleapGerbilsAdapter:
    dataset_name = "sleap_gerbils"

    def _paths(self, raw_root: Path) -> dict[str, Path]:
        root = raw_root.expanduser().resolve()
        return {key: root / filename for key, filename in REQUIRED_FILES.items()}

    def _validate_paths(self, raw_root: Path) -> dict[str, Path]:
        paths = self._paths(raw_root)
        missing = [f"{key}={path}" for key, path in paths.items() if not path.is_file()]
        if missing:
            raise ValidationError("missing SLEAP gerbil files: " + ", ".join(missing))
        return paths

    def _load(self, path: Path, *, open_videos: bool = True):
        return _load_sleap_io()(path, open_videos=open_videos, lazy=True)

    def inspect(self, raw_root: Path) -> DatasetInventory:
        paths = self._validate_paths(raw_root)
        split_details: dict[str, dict[str, Any]] = {}
        all_source_videos: set[str] = set()
        labeled_source_videos: set[str] = set()
        all_tracks: set[str] = set()
        all_nodes: list[str] = []
        all_frames: set[tuple[str, int]] = set()
        all_identities: set[str] = set()
        total_labeled_frames = 0
        total_instances = 0
        embedded = False
        source_filename_preserved = False
        try:
            for split, filename in SPLIT_FILES.items():
                labels = self._load(paths[split], open_videos=True)
                slot_videos: set[str] = set()
                for video in labels.videos:
                    source_video, _, source_name = _source_video_key(labels, video)
                    slot_videos.add(source_video)
                    all_source_videos.add(source_video)
                    source_filename_preserved |= bool(source_name and not source_name.startswith("/"))
                frame_indices: list[int] = []
                instance_count = 0
                split_videos: set[str] = set()
                split_tracks: set[str] = set()
                for labeled_frame in labels.labeled_frames:
                    source_video, _, source_name = _source_video_key(labels, labeled_frame.video)
                    frame_indices.append(int(labeled_frame.frame_idx))
                    split_videos.add(source_video)
                    labeled_source_videos.add(source_video)
                    all_source_videos.add(source_video)
                    all_frames.add((source_video, int(labeled_frame.frame_idx)))
                    source_filename_preserved |= bool(source_name and not source_name.startswith("/"))
                    for instance in labeled_frame.instances:
                        instance_count += 1
                        name = _track_name(instance)
                        if name is not None:
                            split_tracks.add(name)
                            all_tracks.add(name)
                            all_identities.add(name)
                    for node in _node_names(labels):
                        if node not in all_nodes:
                            all_nodes.append(node)
                embedded_for_split = False
                for video in labels.videos:
                    backend = getattr(video, "backend", None)
                    if backend is not None:
                        try:
                            embedded_for_split |= bool(backend.has_embedded_images())
                        except TypeError:
                            embedded_for_split |= bool(getattr(backend, "has_embedded_images", False))
                embedded |= embedded_for_split
                total_labeled_frames += len(labels.labeled_frames)
                total_instances += instance_count
                split_details[split] = {
                    "file": filename,
                    "bytes": paths[split].stat().st_size,
                    "labeled_frame_count": len(labels.labeled_frames),
                    "instance_count": instance_count,
                    "video_slot_count": len(slot_videos),
                    "unique_source_videos_with_labels": sorted(split_videos),
                    "track_names": sorted(split_tracks),
                    "frame_index_range": [min(frame_indices), max(frame_indices)] if frame_indices else None,
                    "embedded_images": embedded_for_split,
                    "skeleton_node_names": _node_names(labels),
                }
                labels.close()
        except Exception:
            # Ensure lazy HDF5 handles are not left open if a malformed split is
            # encountered; the original exception is intentionally propagated.
            raise

        tracking_labels = self._load(paths["tracking"], open_videos=True)
        tracking_details = {
            "file": REQUIRED_FILES["tracking"],
            "bytes": paths["tracking"].stat().st_size,
            "labeled_frame_count": len(tracking_labels.labeled_frames),
            "instance_count": sum(len(frame.instances) for frame in tracking_labels.labeled_frames),
            "source_videos": sorted({_source_video_key(tracking_labels, frame.video)[0]
                                      for frame in tracking_labels.labeled_frames}),
            "is_ground_truth": False,
            "use": "format_and_tracking_smoke_only",
        }
        tracking_labels.close()
        byte_count = sum(path.stat().st_size for path in paths.values())
        inv = DatasetInventory(
            dataset_uid(self.dataset_name, raw_root), self.dataset_name, str(raw_root.expanduser().resolve()),
            file_count=len(paths), byte_count=byte_count,
            video_count=len(all_source_videos), sessions=len(all_source_videos), groups=1,
            individuals=len(all_identities), image_count=len(all_frames),
            capabilities={
                "has_original_video": True,
                "has_embedded_labeled_images": embedded,
                "has_local_track_ids": True,
                "has_verified_persistent_ids": False,
                "has_identity_labels": True,
                "has_pose_ground_truth": True,
                "has_tracking_ground_truth": False,
                "has_camera_calibration": False,
                "can_evaluate_A_end_to_end": False,
                "can_evaluate_H_end_to_end": False,
                "can_evaluate_longitudinal_pose": False,
            },
            notes=[
                "train/val/test are random frame splits over the same source-video slots; session identity is source_video + video index",
                "provided example_tracking.slp is not treated as human ground truth",
                "SLEAP source track names are retained verbatim in private truth; observations contain no identity labels",
            ],
            details={
                "required_files": {key: {"path": str(path), "exists": path.is_file(), "bytes": path.stat().st_size}
                                   for key, path in paths.items()},
                "splits": split_details,
                "tracking": tracking_details,
                "total_labeled_frames": total_labeled_frames,
                "total_instances": total_instances,
                "unique_source_videos": sorted(all_source_videos),
                "unique_source_videos_with_labels": sorted(labeled_source_videos),
                "video_slot_count": len(all_source_videos),
                "track_names": sorted(all_tracks),
                "identity_names_present": sorted(all_identities),
                "skeleton_node_names": all_nodes,
                "frame_index_range": [min(f for _, f in all_frames), max(f for _, f in all_frames)] if all_frames else None,
                "source_filename_preserved": source_filename_preserved,
            },
        )
        return inv

    def _skeleton_manifest(self, source_nodes: list[str]) -> dict[str, Any]:
        source_to_canonical = {
            "lefteye": "left_eye", "righteye": "right_eye",
            "leftear": "left_ear", "rightear": "right_ear",
        }
        source_to_canonical.update({name: name for name in CANONICAL_NODES})
        return {
            "schema_version": "mat.sleap_gerbils.skeleton.v1",
            "canonical_node_names": CANONICAL_NODES,
            "source_node_names": source_nodes,
            "source_to_canonical": {name: source_to_canonical.get(name, name) for name in source_nodes},
            "edges": CANONICAL_EDGES,
            "part_groups": PART_GROUPS,
            "note": "Canonical names follow the SLEAP paper; source names are retained exactly as loaded.",
        }

    def build_manifests(self, raw_root: Path, output_root: Path) -> ManifestBundle:
        paths = self._validate_paths(raw_root)
        output_root = output_root.expanduser().resolve()
        frames_root = output_root / "frames"
        manifests_root = output_root / "manifests"
        frames_root.mkdir(parents=True, exist_ok=True)
        manifests_root.mkdir(parents=True, exist_ok=True)

        frame_rows: dict[str, dict[str, Any]] = {}
        observation_rows: dict[str, dict[str, Any]] = {}
        truth_rows: dict[str, dict[str, Any]] = {}
        session_rows: dict[str, dict[str, Any]] = {}
        identity_by_session: dict[str, set[str]] = defaultdict(set)
        source_nodes: list[str] = []
        split_for_observation: dict[str, str] = {}

        for split, _filename in SPLIT_FILES.items():
            labels = self._load(paths[split], open_videos=True)
            if not source_nodes:
                source_nodes = _node_names(labels)
            # Populate all source-video sessions, including slots with no
            # randomly selected labeled frame.  This preserves the provider's
            # 23-video cohort rather than silently reporting only 11 labeled
            # slots as independent sessions.
            for video in labels.videos:
                source_video, video_index, source_name = _source_video_key(labels, video)
                session_uid = _session_uid(source_video)
                session_rows.setdefault(session_uid, {
                    "schema_version": "mat.session.v1",
                    "session_uid": session_uid,
                    "source_video_name": source_video,
                    "source_filename": source_name,
                    "video_index": video_index,
                    "recording_datetime_if_parseable": _parse_datetime(source_name),
                    "dataset_uid": dataset_uid(self.dataset_name, raw_root),
                    "cohort_uid": "sleap_gerbils:cohort:unresolved",
                })
            for labeled_frame in labels.labeled_frames:
                source_video, video_index, source_name = _source_video_key(labels, labeled_frame.video)
                session_uid = _session_uid(source_video)
                frame_index = int(labeled_frame.frame_idx)
                frame_key = f"{source_video}:{frame_index}"
                frame_uid = f"sleap_gerbils:frame:{_digest(frame_key)}"
                session_rows.setdefault(session_uid, {
                    "schema_version": "mat.session.v1",
                    "session_uid": session_uid,
                    "source_video_name": source_video,
                    "source_filename": source_name,
                    "video_index": video_index,
                    "recording_datetime_if_parseable": _parse_datetime(source_name),
                    "dataset_uid": dataset_uid(self.dataset_name, raw_root),
                    "cohort_uid": "sleap_gerbils:cohort:unresolved",
                })
                if frame_uid not in frame_rows:
                    image = np.asarray(labeled_frame.video[frame_index])
                    if image.ndim == 2:
                        image = np.repeat(image[..., None], 3, axis=-1)
                    if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
                        raise ValidationError(f"unexpected SLEAP embedded image shape: {image.shape}")
                    if image.shape[-1] == 1:
                        image = np.repeat(image, 3, axis=-1)
                    if image.shape[-1] == 4:
                        image = image[..., :3]
                    image = np.asarray(np.clip(image, 0, 255), dtype=np.uint8)
                    image_name = f"{_digest(frame_key, 32)}.png"
                    image_path = frames_root / image_name
                    from PIL import Image
                    Image.fromarray(image).save(image_path, format="PNG")
                    frame_rows[frame_uid] = {
                        "schema_version": "mat.frame.v1",
                        "frame_uid": frame_uid,
                        "session_uid": session_uid,
                        "dataset_uid": dataset_uid(self.dataset_name, raw_root),
                        "cohort_uid": "sleap_gerbils:cohort:unresolved",
                        "frame_index": frame_index,
                        "timestamp_s": frame_index / 25.0,
                        "image_ref": f"frames/{image_name}",
                        "source_video_name": source_video,
                    }
                for instance_index, instance in enumerate(labeled_frame.instances):
                    track_name = _track_name(instance)
                    # Deduplicate the random provider splits by source video,
                    # frame index and track identity (not by split or instance
                    # list position).
                    instance_key = track_name or f"untracked-{instance_index}"
                    truth_uid = f"{frame_uid}:track:{instance_key}"
                    observation_uid = f"sleap_gerbils:observation:{_digest(truth_uid)}"
                    tracklet_uid = f"sleap_gerbils:tracklet:{_digest(session_uid + ':' + (track_name or str(instance_index)), 20)}"
                    # A repeated source frame/track can occur in a provider split;
                    # first occurrence wins and cannot create a duplicate truth row.
                    if observation_uid in observation_rows:
                        continue
                    observation_rows[observation_uid] = {
                        "schema_version": "mat.observation.v1",
                        "observation_uid": observation_uid,
                        "frame_uid": frame_uid,
                        "session_uid": session_uid,
                        "dataset_uid": dataset_uid(self.dataset_name, raw_root),
                        "cohort_uid": "sleap_gerbils:cohort:unresolved",
                        "frame_index": frame_index,
                        "timestamp_s": frame_index / 25.0,
                        "image_ref": f"frames/{Path(frame_rows[frame_uid]['image_ref']).name}",
                        "tracklet_uid": tracklet_uid,
                        "input_mode": "embedded_frame",
                    }
                    split_for_observation[observation_uid] = split
                    keypoints, visibility = _instance_truth(instance)
                    truth_rows[observation_uid] = {
                        "schema_version": "mat.private_pose_identity_truth.v1",
                        "observation_uid": observation_uid,
                        "source_video": source_video,
                        "frame_idx": frame_index,
                        "gt_identity": track_name,
                        "gt_keypoints": keypoints,
                        "gt_visibility": visibility,
                        "source_split": split,
                    }
                    if track_name is not None:
                        identity_by_session[session_uid].add(track_name)
            labels.close()

        for session_uid, identities in identity_by_session.items():
            session_rows[session_uid]["identity_names_present_private"] = sorted(identities)

        # Session inventory intentionally contains provider labels for auditing;
        # sessions.jsonl is neutral and strips that private field below.
        session_inventory = []
        for session_uid, row in sorted(session_rows.items()):
            session_inventory.append({
                **row,
                "labeled_frame_count": sum(1 for f in frame_rows.values() if f["session_uid"] == session_uid),
                "observation_count": sum(1 for o in observation_rows.values() if o["session_uid"] == session_uid),
                "identity_names_present": sorted(identity_by_session.get(session_uid, set())),
            })
        neutral_sessions = []
        for row in session_inventory:
            neutral_sessions.append({k: v for k, v in row.items() if k != "identity_names_present" and not k.endswith("_private")})

        frames_path = manifests_root / "frames.jsonl"
        sessions_path = manifests_root / "sessions.jsonl"
        observations_path = manifests_root / "observations.jsonl"
        truth_path = manifests_root / "private_pose_identity_truth.jsonl"
        session_inventory_path = manifests_root / "session_inventory.json"
        skeleton_path = manifests_root / "skeleton.json"
        write_jsonl(frames_path, (frame_rows[key] for key in sorted(frame_rows)))
        write_jsonl(sessions_path, neutral_sessions)
        write_jsonl(observations_path, (observation_rows[key] for key in sorted(observation_rows)))
        write_jsonl(truth_path, (truth_rows[key] for key in sorted(truth_rows)))
        session_inventory_path.write_text(json.dumps({
            "schema_version": "mat.sleap_gerbils.session_inventory.v1",
            "session_uid_rule": "source_video_name + video_index; never split filename",
            "sessions": session_inventory,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        skeleton = self._skeleton_manifest(source_nodes)
        skeleton_path.write_text(json.dumps(skeleton, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_jsonl(manifests_root / "source_train_labels.jsonl", [])
        write_jsonl(manifests_root / "reference_annotations.jsonl", [])

        inv = self.inspect(raw_root)
        inv.details["prepared"] = {
            "frames": len(frame_rows),
            "observations": len(observation_rows),
            "private_truth": len(truth_rows),
            "sessions": len(session_rows),
            "split_counts": {split: sum(1 for value in split_for_observation.values() if value == split)
                             for split in SPLIT_FILES},
        }
        inv.write(manifests_root / "inventory.json")
        return ManifestBundle(
            observations_path,
            manifests_root / "source_train_labels.jsonl",
            manifests_root / "reference_annotations.jsonl",
            truth_path,
            inv,
        )

    def iter_observations(self, session):
        raise ValidationError(
            "SLEAP gerbil observations are file-backed; use manifests/observations.jsonl and an explicit image loader"
        )
