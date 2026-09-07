from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json

from mat.core.errors import ValidationError


@dataclass(frozen=True)
class AssetSpec:
    asset_id: str
    url: str
    role: str
    license: str | None = None
    expected_bytes: int | None = None
    provider_checksum: str | None = None
    checksum_algorithm: str | None = None
    source_revision: str | None = None
    allowed_hosts: tuple[str, ...] = ()
    max_bytes: int | None = None
    phase: str = "P0"
    metadata_url: str | None = None
    download_url_verified: bool = False

    def __post_init__(self) -> None:
        if not self.asset_id or not self.url:
            raise ValidationError("asset_id and url are required")
        if self.expected_bytes is not None and self.expected_bytes < 0:
            raise ValidationError("expected_bytes must be non-negative")
        if self.max_bytes is not None and self.max_bytes <= 0:
            raise ValidationError("max_bytes must be positive")


@dataclass(frozen=True)
class ProbeReceipt:
    asset_id: str
    status: str
    http_status: int | None
    content_length: int | None
    content_type: str | None
    downloaded_bytes: int
    redirect_chain: tuple[str, ...]
    url_origin: str
    route_status: str
    observed_at: str
    failure_reason: str | None = None


@dataclass(frozen=True)
class ArtifactReceipt:
    asset_id: str
    status: str
    path: str | None
    downloaded_bytes: int
    resumed_bytes: int
    provider_checksum: str | None
    sha256: str | None
    expected_bytes: int | None
    url_origin: str | None
    source_revision: str | None
    license: str | None
    route_status: str
    started_at: str
    finished_at: str
    redirect_chain: tuple[str, ...] = ()
    failure_reason: str | None = None

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class AssetCatalog:
    def __init__(self, assets: list[AssetSpec]):
        self.assets = {a.asset_id: a for a in assets}
        if len(self.assets) != len(assets):
            raise ValidationError("duplicate asset_id")

    def get(self, asset_id: str) -> AssetSpec:
        try:
            return self.assets[asset_id]
        except KeyError as exc:
            raise ValidationError(f"unknown asset: {asset_id}") from exc

    def for_phase(self, phase: str) -> list[AssetSpec]:
        order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        if phase not in order:
            raise ValidationError(f"unknown phase {phase}")
        return [a for a in self.assets.values() if order.get(a.phase, 99) <= order[phase]]

    @classmethod
    def from_yaml(cls, path: Path) -> "AssetCatalog":
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ValidationError("PyYAML is required to read an asset catalog") from exc
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        entries = raw.get("assets", []) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
        assets: list[AssetSpec] = []
        for item in entries:
            item = dict(item)
            hosts = item.get("allowed_hosts") or ()
            item["allowed_hosts"] = tuple(hosts)
            assets.append(AssetSpec(**item))
        return cls(assets)
