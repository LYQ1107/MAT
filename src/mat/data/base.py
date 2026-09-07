from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator, Protocol
import json
import hashlib
import os

from mat.core.errors import ValidationError
from mat.core.types import FramePacket, SessionSpec


@dataclass
class DatasetInventory:
    dataset_uid: str
    dataset_name: str
    raw_root: str
    file_count: int = 0
    image_count: int = 0
    video_count: int = 0
    archive_count: int = 0
    byte_count: int = 0
    sessions: int = 0
    groups: int = 0
    individuals: int = 0
    capabilities: dict[str, bool | None] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    parse_failures: list[dict[str, str]] = field(default_factory=list)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class ManifestBundle:
    observations: Path
    source_train_labels: Path
    reference_annotations: Path
    private_eval_truth: Path
    inventory: DatasetInventory


class DatasetAdapter(Protocol):
    dataset_name: str

    def inspect(self, raw_root: Path) -> DatasetInventory: ...
    def build_manifests(self, raw_root: Path, output_root: Path) -> ManifestBundle: ...
    def iter_observations(self, session: SessionSpec) -> Iterator[FramePacket]: ...


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ARCHIVE_SUFFIXES = {".zip", ".tar", ".gz", ".tgz", ".7z"}


def dataset_uid(name: str, raw_root: Path) -> str:
    # Root path is not placed in model manifests; this UID only namespaces a release.
    return hashlib.sha256(f"{name}:{raw_root.resolve()}".encode()).hexdigest()[:20]


def bounded_files(raw_root: Path) -> list[Path]:
    if not raw_root.exists() or not raw_root.is_dir():
        raise ValidationError(f"raw root is not a directory: {raw_root}")
    # No full-disk discovery: traversal is bounded to the user supplied root.
    return [p for p in raw_root.rglob("*") if p.is_file() and not p.is_symlink()]


def neutral_uid(dataset: str, ordinal: int, rel_suffix: str = "") -> str:
    return hashlib.sha256(f"{dataset}:observation:{ordinal}:{rel_suffix}".encode()).hexdigest()[:24]


def neutral_tracklet_uid(dataset: str, parent_token: str) -> str:
    return hashlib.sha256(f"{dataset}:tracklet:{parent_token}".encode()).hexdigest()[:24]
