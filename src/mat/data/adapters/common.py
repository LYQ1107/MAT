from __future__ import annotations

from pathlib import Path
from typing import Callable, Any
import re
import hashlib
import numpy as np

from mat.core.types import FramePacket, SessionSpec
from mat.core.errors import ValidationError
from mat.data.base import DatasetInventory, ManifestBundle, bounded_files, dataset_uid, neutral_uid, neutral_tracklet_uid, IMAGE_SUFFIXES, VIDEO_SUFFIXES, ARCHIVE_SUFFIXES
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
    # Session/camera are opaque hashes: a provider folder can contain an EID and must
    # never be exposed verbatim to the model process.
    parent = parts[-2] if len(parts) > 1 else "0"
    token = hashlib.sha256(f"{dataset_name}:session:{parent}".encode()).hexdigest()[:16]
    session = f"{dataset_name}:session:{token}"
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
        tracklet_uid = neutral_tracklet_uid(dataset_name, str(path.parent.relative_to(raw_root)))
        row = {
            "schema_version": "mat.observation.v1", "observation_uid": uid,
            "frame_uid": f"{session_uid}:{ordinal}", "session_uid": session_uid,
            "dataset_uid": inv.dataset_uid, "cohort_uid": f"{dataset_name}:cohort:unresolved", "camera_uid": camera_uid,
            "frame_index": ordinal, "timestamp_s": timestamp, "image_ref": object_ref, "tracklet_uid": tracklet_uid,
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


def iter_session_frames(session: SessionSpec):
    """Resolve neutral object handles inside the explicitly supplied raw root.

    The caller can provide ``timebase.raw_root`` (a user-authorized path) and the
    manifest's sibling private object index. Neither path nor provider labels cross
    the FramePacket boundary.
    """
    raw_root = session.timebase.get("raw_root") if isinstance(session.timebase, dict) else None
    if not raw_root:
        raise ValidationError("SessionSpec.timebase.raw_root is required to resolve neutral objects")
    raw_root = Path(raw_root).resolve()
    index_path = session.observations_manifest.with_name("private_object_index.jsonl")
    object_index = {row["object_ref"]: row["relative_source"] for row in iter_manifest_records(index_path)} if index_path.exists() else {}

    def load(ref):
        source = object_index.get(ref)
        if source is None:
            raise ValidationError(f"object handle not found in private index: {ref}")
        path = (raw_root / source).resolve()
        try:
            path.relative_to(raw_root)
        except ValueError as exc:
            raise ValidationError("private object index escapes raw root") from exc
        try:
            from PIL import Image
            with Image.open(path) as image:
                return np.asarray(image.convert("RGB"), dtype=np.uint8)
        except ImportError as exc:  # pragma: no cover
            raise ValidationError("Pillow is required to decode prepared image observations") from exc

    yield from records_to_frames(session, list(iter_manifest_records(session.observations_manifest)), load)
