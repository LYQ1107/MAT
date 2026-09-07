from __future__ import annotations

from pathlib import Path
from typing import Callable, Any
import re

from mat.core.types import FramePacket, SessionSpec
from mat.core.errors import ValidationError
from mat.data.base import DatasetInventory, ManifestBundle, bounded_files, dataset_uid, neutral_uid, IMAGE_SUFFIXES, VIDEO_SUFFIXES, ARCHIVE_SUFFIXES
from mat.data.manifests import write_jsonl, iter_manifest_records


def inspect_files(name: str, raw_root: Path, capabilities: dict[str, bool | None], notes: list[str]) -> DatasetInventory:
    files = bounded_files(raw_root)
    inv = DatasetInventory(dataset_uid(name, raw_root), name, str(raw_root.resolve()), capabilities=capabilities, notes=list(notes))
    for path in files:
        try:
            inv.byte_count += path.stat().st_size
        except OSError as exc:
            inv.parse_failures.append({"path": str(path), "reason": str(exc)})
            continue
        inv.file_count += 1
        suffix = path.suffix.lower()
        if suffix in IMAGE_SUFFIXES: inv.image_count += 1
        elif suffix in VIDEO_SUFFIXES: inv.video_count += 1
        elif suffix in ARCHIVE_SUFFIXES: inv.archive_count += 1
    return inv


def _session_from(path: Path, dataset_name: str, ordinal: int) -> tuple[str, str, str, float]:
    parts = path.parts
    # Session/camera are intentionally neutralized and never inferred as IDs.
    session = f"{dataset_name}:session:{parts[-2] if len(parts) > 1 else '0'}"
    camera = "camera:unknown"
    return session, camera, float(ordinal)


def build_common_manifests(dataset_name: str, raw_root: Path, output_root: Path,
                           inv: DatasetInventory,
                           label_parser: Callable[[Path, int], dict[str, Any] | None] | None = None) -> ManifestBundle:
    image_paths = sorted(p for p in bounded_files(raw_root) if p.suffix.lower() in IMAGE_SUFFIXES)
    observations: list[dict[str, Any]] = []
    truth: list[dict[str, Any]] = []
    index: list[dict[str, Any]] = []
    for ordinal, path in enumerate(image_paths):
        session_uid, camera_uid, timestamp = _session_from(path, dataset_name, ordinal)
        uid = neutral_uid(dataset_name, ordinal)
        object_ref = f"object://{dataset_name}/{uid}"
        row = {
            "schema_version": "mat.observation.v1", "observation_uid": uid,
            "frame_uid": f"{session_uid}:{ordinal}", "session_uid": session_uid,
            "cohort_uid": f"{dataset_name}:cohort:unresolved", "camera_uid": camera_uid,
            "frame_index": ordinal, "timestamp_s": timestamp, "image_ref": object_ref,
            "input_mode": "prelocalized_crops",
        }
        observations.append(row)
        index.append({"object_ref": object_ref, "relative_source": str(path.relative_to(raw_root))})
        if label_parser:
            parsed = label_parser(path, ordinal)
            if parsed:
                truth.append({"schema_version": "mat.truth.v1", "observation_uid": uid, **parsed})
    obs_path = output_root / "observations.jsonl"
    source_path = output_root / "source_train_labels.jsonl"
    ref_path = output_root / "reference_annotations.jsonl"
    truth_path = output_root / "private_eval_truth.jsonl"
    index_path = output_root / "private_object_index.jsonl"
    write_jsonl(obs_path, observations)
    write_jsonl(source_path, [])
    write_jsonl(ref_path, [])
    write_jsonl(truth_path, truth)
    write_jsonl(index_path, index)
    inv.notes.append(f"neutral object index written separately: {index_path.name}")
    inv.sessions = len({r["session_uid"] for r in observations})
    inv.write(output_root / "inventory.json")
    return ManifestBundle(obs_path, source_path, ref_path, truth_path, inv)


def records_to_frames(session: SessionSpec, records: list[dict[str, Any]], image_loader=None):
    for row in records:
        if row.get("session_uid") != session.session_uid:
            continue
        rgb = image_loader(row["image_ref"]) if image_loader else None
        if rgb is None:
            raise ValidationError("neutral manifest requires an explicit image_loader for pixel access")
        yield FramePacket(row.get("dataset_uid", "unknown"), session.cohort_uid, session.session_uid,
                          session.camera_uid, int(row["frame_index"]), float(row["timestamp_s"]), rgb)

