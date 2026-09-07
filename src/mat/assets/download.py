"""One guarded HTTP path for probes/fetches; no caller should bypass it."""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse, urlunparse
import hashlib
import json
import os
import time

import requests

from mat.core.errors import IntegrityError, RouteNotApprovedError, MATError
from .catalog import AssetSpec, ArtifactReceipt, ProbeReceipt
from .policy import DirectOnlyPolicy
from .verify import sha256_file, verify_file, validate_archive
from .import_local import import_local as import_local_artifact


def _origin(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


class AssetDownloader:
    def __init__(self, destination_root: Path, *, max_probe_bytes: int = 64 * 1024,
                 max_retries: int = 3, timeout: tuple[float, float] = (15.0, 30.0)):
        self.destination_root = destination_root
        self.max_probe_bytes = max_probe_bytes
        self.max_retries = max_retries
        self.timeout = timeout

    @staticmethod
    def _session() -> requests.Session:
        session = requests.Session()
        session.trust_env = False
        session.headers.update({"User-Agent": "MAT-AssetDownloader/0.1", "Accept-Encoding": "identity"})
        return session

    @staticmethod
    def _effective_policy(asset: AssetSpec, policy: DirectOnlyPolicy) -> DirectOnlyPolicy:
        """Merge per-asset provider allowlist without mutating the caller policy."""
        if asset.allowed_hosts:
            return replace(policy, allowed_hosts=frozenset(asset.allowed_hosts))
        return policy

    def _request(self, asset: AssetSpec, policy: DirectOnlyPolicy, method: str,
                 *, headers: dict[str, str] | None = None, stream: bool = True):
        policy = self._effective_policy(asset, policy)
        policy.validate_url(asset.url, asset.asset_id)
        current = asset.url
        chain: list[str] = [_origin(current)]
        session = self._session()
        for _ in range(8):
            response = session.request(method, current, headers=headers or {}, stream=stream,
                                       allow_redirects=False, timeout=self.timeout)
            if response.is_redirect or response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("Location")
                response.close()
                if not location:
                    raise MATError(f"{asset.asset_id}: redirect without Location")
                from urllib.parse import urljoin
                new_url = urljoin(current, location)
                policy.validate_redirect(current, new_url, asset.asset_id)
                current = new_url
                chain.append(_origin(current))
                continue
            return response, tuple(chain)
        raise MATError(f"{asset.asset_id}: redirect limit exceeded")

    def probe(self, asset: AssetSpec, policy: DirectOnlyPolicy) -> ProbeReceipt:
        started = datetime.now(timezone.utc).isoformat()
        try:
            policy = self._effective_policy(asset, policy)
            policy.validate_url(asset.url, asset.asset_id)
            # A probe is intentionally small and does not approve bulk transfer.
            response, chain = self._request(asset, policy, "HEAD", stream=True)
            if response.status_code in {405, 501}:
                response.close()
                response, chain = self._request(asset, policy, "GET",
                                                headers={"Range": f"bytes=0-{self.max_probe_bytes - 1}"},
                                                stream=True)
            content_length = response.headers.get("Content-Length")
            length = int(content_length) if content_length and content_length.isdigit() else None
            consumed = 0
            if response.request.method == "GET":
                for chunk in response.iter_content(8192):
                    consumed += len(chunk)
                    if consumed >= self.max_probe_bytes:
                        break
            response.close()
            return ProbeReceipt(asset.asset_id, "PROBED", response.status_code, length,
                                response.headers.get("Content-Type"), consumed, chain,
                                chain[-1], policy.route_status, started)
        except Exception as exc:
            return ProbeReceipt(asset.asset_id, "FAILED", None, None, None, 0, (), _origin(asset.url),
                                policy.route_status, started, str(exc))

    def fetch(self, asset: AssetSpec, policy: DirectOnlyPolicy) -> ArtifactReceipt:
        started = datetime.now(timezone.utc).isoformat()
        part_dir = self.destination_root / "downloads"
        part_dir.mkdir(parents=True, exist_ok=True)
        filename = Path(urlparse(asset.url).path).name or f"{asset.asset_id}.asset"
        target = part_dir / filename
        part = part_dir / (filename + ".part")
        receipt_path = part_dir / (filename + ".receipt.json")
        max_bytes = asset.max_bytes or (asset.expected_bytes or 0)
        if not max_bytes:
            max_bytes = 50 * 1024**3
        try:
            policy = self._effective_policy(asset, policy)
            if not asset.download_url_verified:
                raise MATError(f"{asset.asset_id}: download URL is not verified from provider metadata")
            policy.assert_route_approved((urlparse(asset.url).hostname or "").lower())
            policy.validate_url(asset.url, asset.asset_id)
            if target.exists():
                recorded = None
                if receipt_path.exists():
                    try:
                        previous = json.loads(receipt_path.read_text(encoding="utf-8"))
                        if previous.get("status") == "VERIFIED":
                            recorded = previous.get("sha256")
                    except (OSError, ValueError):
                        recorded = None
                digest = verify_file(target, asset.expected_bytes, asset.provider_checksum, asset.checksum_algorithm)
                if not asset.provider_checksum and not recorded:
                    raise IntegrityError("existing file lacks a prior verified receipt")
                if recorded and recorded != digest:
                    raise IntegrityError("existing file differs from its verified receipt")
                receipt = ArtifactReceipt(asset.asset_id, "VERIFIED", str(target), target.stat().st_size, 0,
                                          asset.provider_checksum, digest, asset.expected_bytes, _origin(asset.url),
                                          asset.source_revision, asset.license, policy.route_status, started,
                                          datetime.now(timezone.utc).isoformat())
                receipt.write(receipt_path)
                return receipt
            resume_from = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
            response, chain = self._request(asset, policy, "GET", headers=headers, stream=True)
            if response.status_code not in {200, 206}:
                raise MATError(f"HTTP {response.status_code}")
            mode = "ab" if resume_from and response.status_code == 206 else "wb"
            resumed = resume_from if mode == "ab" else 0
            if resume_from and response.status_code == 206:
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {resume_from}-"):
                    raise IntegrityError(f"invalid Content-Range for resume: {content_range}")
            # A 200 response after a Range request is a fresh download, never append.
            received = resume_from if mode == "ab" else 0
            with part.open(mode) as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if not chunk:
                        continue
                    received += len(chunk)
                    if received > max_bytes:
                        raise IntegrityError(f"asset exceeds byte budget {max_bytes}")
                    handle.write(chunk)
            response.close()
            if asset.expected_bytes is not None and received != asset.expected_bytes:
                raise IntegrityError(f"size mismatch: expected {asset.expected_bytes}, got {received}")
            digest = verify_file(part, asset.expected_bytes, asset.provider_checksum, asset.checksum_algorithm)
            if part.suffix.lower() in {".zip", ".tar", ".gz", ".tgz"}:
                validate_archive(part, self.destination_root / "archive_check")
            os.replace(part, target)
            receipt = ArtifactReceipt(asset.asset_id, "VERIFIED", str(target), received, resumed,
                                      asset.provider_checksum, digest, asset.expected_bytes, chain[-1],
                                      asset.source_revision, asset.license, policy.route_status, started,
                                      datetime.now(timezone.utc).isoformat(), chain)
            receipt.write(receipt_path)
            return receipt
        except Exception as exc:
            receipt = ArtifactReceipt(asset.asset_id, "FAILED", None, 0,
                                      part.stat().st_size if part.exists() else 0,
                                      asset.provider_checksum, None, asset.expected_bytes, _origin(asset.url),
                                      asset.source_revision, asset.license, policy.route_status, started,
                                      datetime.now(timezone.utc).isoformat(), failure_reason=str(exc))
            receipt.write(receipt_path)
            return receipt

    def import_local(self, asset: AssetSpec, source: Path) -> ArtifactReceipt:
        return import_local_artifact(asset, source, self.destination_root / "raw")
