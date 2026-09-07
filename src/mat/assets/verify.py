from __future__ import annotations

from pathlib import Path
import hashlib
import os
import tarfile
import zipfile

from mat.core.errors import IntegrityError


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_file(path: Path, algorithm: str) -> str:
    try:
        digest = hashlib.new(algorithm.lower())
    except ValueError as exc:
        raise IntegrityError(f"unsupported checksum algorithm: {algorithm}") from exc
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_bytes: int | None = None,
                provider_checksum: str | None = None,
                algorithm: str | None = None) -> str:
    if not path.is_file():
        raise IntegrityError(f"not a regular file: {path}")
    size = path.stat().st_size
    if expected_bytes is not None and size != expected_bytes:
        raise IntegrityError(f"size mismatch: expected {expected_bytes}, got {size}")
    if provider_checksum:
        if not algorithm:
            raise IntegrityError("provider checksum supplied without algorithm")
        actual = digest_file(path, algorithm)
        if actual.lower() != provider_checksum.lower():
            raise IntegrityError(f"{algorithm} mismatch for {path}")
    return sha256_file(path)


def _safe_member(name: str, root: Path) -> bool:
    if not name or name.startswith("/"):
        return False
    candidate = (root / name).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return False
    return True


def validate_archive(path: Path, extraction_root: Path, max_members: int = 2_000_000,
                     max_uncompressed_bytes: int = 500 * 1024**3) -> dict[str, int]:
    """Validate paths, links, member count and expansion before extraction."""
    extraction_root = extraction_root.resolve()
    count = 0
    total = 0
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                count += 1
                total += max(0, info.file_size)
                if count > max_members or total > max_uncompressed_bytes:
                    raise IntegrityError("archive expansion exceeds safety budget")
                if not _safe_member(info.filename, extraction_root):
                    raise IntegrityError(f"zip-slip path: {info.filename!r}")
                # Unix mode symlink bit, when present.
                if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                    raise IntegrityError(f"symlink in zip is forbidden: {info.filename!r}")
        return {"members": count, "uncompressed_bytes": total}
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                count += 1
                total += max(0, member.size)
                if count > max_members or total > max_uncompressed_bytes:
                    raise IntegrityError("archive expansion exceeds safety budget")
                if not _safe_member(member.name, extraction_root):
                    raise IntegrityError(f"tar-slip path: {member.name!r}")
                if member.issym() or member.islnk():
                    raise IntegrityError(f"link in tar is forbidden: {member.name!r}")
        return {"members": count, "uncompressed_bytes": total}
    return {"members": 0, "uncompressed_bytes": 0}

