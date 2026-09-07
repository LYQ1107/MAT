from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable
import json
import uuid


@dataclass(frozen=True)
class AnnotationManifest:
    annotation_id: str
    status: str
    sessions: tuple[str, ...]
    sampling_spec: dict[str, Any]
    skeleton: dict[str, Any]
    items: tuple[dict[str, Any], ...]
    provenance: str = "MAT_annotation_export"

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class AnnotationExporter:
    def export(self, sessions: Iterable[Any], sampling_spec: dict[str, Any],
               skeleton_spec: dict[str, Any], output_dir: Path) -> AnnotationManifest:
        sessions = list(sessions)
        items = []
        for session in sessions:
            items.append({"item_uid": str(uuid.uuid4()), "session_uid": getattr(session, "session_uid", str(session)),
                          "frame_selection": sampling_spec, "status": "待人工核验"})
        manifest = AnnotationManifest(str(uuid.uuid4()), "PLANNED", tuple(getattr(s, "session_uid", str(s)) for s in sessions),
                                      dict(sampling_spec), dict(skeleton_spec), tuple(items))
        manifest.write(output_dir / "annotation_manifest.json")
        return manifest

