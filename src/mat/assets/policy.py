"""Application-layer direct-only policy; does not change the parent process."""

from __future__ import annotations

from dataclasses import dataclass, field
from ipaddress import ip_address
from urllib.parse import urlparse
import os

from mat.core.errors import RouteNotApprovedError, ValidationError


PROXY_VARS = (
    "http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "ftp_proxy", "FTP_PROXY",
)


def _host(url: str) -> str:
    parsed = urlparse(url)
    return (parsed.hostname or "").lower().rstrip(".")


@dataclass(frozen=True)
class DirectOnlyPolicy:
    """Policy applied to a child download process.

    ``route_status`` is deliberately not inferred from removing environment variables:
    only an operator/network audit can set ``VERIFIED_DIRECT``. Tests may use a loopback
    server with that status because it has no external egress.
    """

    route_status: str = "DIRECT_ROUTE_UNVERIFIED"
    allowed_hosts: frozenset[str] = field(default_factory=frozenset)
    allow_insecure_localhost: bool = False
    max_metadata_bytes: int = 64 * 1024 * 1024

    def child_env(self, parent_env: dict[str, str] | None = None) -> dict[str, str]:
        env = dict(os.environ if parent_env is None else parent_env)
        for name in PROXY_VARS:
            env.pop(name, None)
        env["NO_PROXY"] = "*"
        env["no_proxy"] = "*"
        # Prevent common package clients from consulting ambient online caches.
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["PIP_NO_INDEX"] = env.get("PIP_NO_INDEX", "0")
        return env

    def validate_url(self, url: str, asset_id: str = "asset") -> None:
        parsed = urlparse(url)
        host = _host(url)
        if parsed.scheme != "https":
            if not (self.allow_insecure_localhost and host in {"127.0.0.1", "localhost", "::1"}):
                raise ValidationError(f"{asset_id}: only https URLs are allowed")
        if not host:
            raise ValidationError(f"{asset_id}: URL has no host")
        if self.allowed_hosts and host not in self.allowed_hosts:
            raise ValidationError(f"{asset_id}: host {host!r} is outside the allowlist")
        # Credentials in URLs are never accepted; query strings are retained only for
        # the caller's origin audit and are not forwarded as Authorization/Cookie.
        if parsed.username or parsed.password:
            raise ValidationError(f"{asset_id}: credentials in URL are forbidden")

    def validate_redirect(self, old_url: str, new_url: str, asset_id: str = "asset") -> None:
        self.validate_url(new_url, asset_id)
        if _host(old_url) != _host(new_url) and _host(new_url) not in self.allowed_hosts:
            raise ValidationError(f"{asset_id}: redirect changes host without explicit allowlist")

    def assert_route_approved(self, host: str) -> None:
        if self.route_status != "VERIFIED_DIRECT":
            raise RouteNotApprovedError(
                f"bulk asset blocked: route_status={self.route_status}; host={host}"
            )

    def is_loopback(self, host: str) -> bool:
        try:
            return ip_address(host).is_loopback
        except ValueError:
            return host == "localhost"

