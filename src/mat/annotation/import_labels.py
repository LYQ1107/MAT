from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import numpy as np

from mat.core.errors import ValidationError


@dataclass(frozen=True)
class VerifiedAnnotations:
    rows: tuple[dict[str, Any], ...]
    provenance: str
    skeleton_version: str


class AnnotationImporter:
    def load(self, annotation_path: Path, manifest: dict[str, Any] | Path,
             skeleton_spec: dict[str, Any]) -> VerifiedAnnotations:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        rows = payload.get("annotations", payload if isinstance(payload, list) else [])
        if not isinstance(rows, list):
            raise ValidationError("annotation file must contain a list or annotations field")
        expected = set(skeleton_spec.get("keypoint_names", []))
        for row in rows:
            if "observation_uid" not in row or "keypoints" not in row:
                raise ValidationError("annotation row lacks neutral UID/keypoints")
            if expected and isinstance(row["keypoints"], dict) and set(row["keypoints"]) - expected:
                raise ValidationError("annotation contains unknown keypoint name")
        version = str(skeleton_spec.get("skeleton_version", "unresolved"))
        return VerifiedAnnotations(tuple(rows), str(payload.get("provenance", "manual")), version)

