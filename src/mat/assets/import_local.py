from __future__ import annotations

from pathlib import Path
import shutil
from datetime import datetime, timezone

from .catalog import AssetSpec, ArtifactReceipt
from .verify import validate_archive, verify_file


def import_local(asset: AssetSpec, source: Path, destination_root: Path,
                 route_status: str = "LOCAL_AUTHORIZED") -> ArtifactReceipt:
    """Copy an authorized local original after verification; never mutate ``source``."""
    started = datetime.now(timezone.utc).isoformat()
    source = source.expanduser().resolve()
    if not source.is_file():
        return ArtifactReceipt(asset.asset_id, "FAILED", None, 0, 0, asset.provider_checksum,
                               None, asset.expected_bytes, None, asset.source_revision,
                               asset.license, route_status, started, datetime.now(timezone.utc).isoformat(),
                               failure_reason=f"source is not a regular file: {source}")
    try:
        digest = verify_file(source, asset.expected_bytes, asset.provider_checksum, asset.checksum_algorithm)
        if source.suffix.lower() in {".zip", ".tar", ".gz", ".tgz", ".tar.gz"}:
            validate_archive(source, destination_root / "_archive_check")
        destination_root.mkdir(parents=True, exist_ok=True)
        target = destination_root / source.name
        part = target.with_name(target.name + ".part")
        if target.exists():
            existing = verify_file(target, asset.expected_bytes, asset.provider_checksum, asset.checksum_algorithm)
            if existing != digest:
                raise ValueError("existing destination has a different hash")
        else:
            shutil.copyfile(source, part)
            copied = verify_file(part, asset.expected_bytes, asset.provider_checksum, asset.checksum_algorithm)
            if copied != digest:
                raise ValueError("copied file hash changed")
            part.replace(target)
        return ArtifactReceipt(asset.asset_id, "VERIFIED", str(target), target.stat().st_size, 0,
                               asset.provider_checksum, digest, asset.expected_bytes, None,
                               asset.source_revision, asset.license, route_status, started,
                               datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        return ArtifactReceipt(asset.asset_id, "FAILED", None, 0, 0, asset.provider_checksum,
                               None, asset.expected_bytes, None, asset.source_revision,
                               asset.license, route_status, started, datetime.now(timezone.utc).isoformat(),
                               failure_reason=str(exc))

