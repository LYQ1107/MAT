from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from mat.assets.catalog import AssetSpec
from mat.assets.download import AssetDownloader
from mat.assets.policy import DirectOnlyPolicy
from mat.assets.verify import validate_archive
from mat.core.errors import RouteNotApprovedError, ValidationError


def test_child_env_removes_proxy_without_mutating_parent():
    parent = {"HTTP_PROXY": "secret-host", "PATH": "/bin"}
    child = DirectOnlyPolicy().child_env(parent)
    assert "HTTP_PROXY" not in child and child["NO_PROXY"] == "*"
    assert parent["HTTP_PROXY"] == "secret-host"


def test_redirect_and_route_are_explicit():
    policy = DirectOnlyPolicy(route_status="DIRECT_ROUTE_UNVERIFIED", allowed_hosts=frozenset({"example.org"}))
    with pytest.raises(RouteNotApprovedError): policy.assert_route_approved("example.org")
    with pytest.raises(ValidationError): policy.validate_redirect("https://example.org/a", "https://evil.org/b")


def test_zip_slip_is_rejected(tmp_path):
    import zipfile
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../../outside.txt", b"x")
    with pytest.raises(Exception): validate_archive(archive, tmp_path / "extract")


def test_fetch_respects_byte_budget_and_does_not_use_parent_proxy(tmp_path):
    payload = b"0123456789" * 20

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        def log_message(self, *_):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/file.bin"
        asset = AssetSpec("fixture", url, "TEST_FIXTURE", expected_bytes=len(payload), max_bytes=32,
                          allowed_hosts=("127.0.0.1",), download_url_verified=True)
        policy = DirectOnlyPolicy(route_status="VERIFIED_DIRECT", allowed_hosts=frozenset({"127.0.0.1"}), allow_insecure_localhost=True)
        receipt = AssetDownloader(tmp_path).fetch(asset, policy)
        assert receipt.status == "FAILED" and "budget" in (receipt.failure_reason or "")
        assert not (tmp_path / "downloads" / "file.bin").exists()
    finally:
        server.shutdown(); thread.join(timeout=2)
