import io
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading

from mat.assets import AssetDownloader, DirectOnlyPolicy
from mat.assets.catalog import AssetSpec


def test_archive_validation_uses_published_target_suffix(tmp_path, monkeypatch):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("inside.txt", "ok")
    data = payload.getvalue()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    calls = []
    import mat.assets.download as download
    real_validate = download.validate_archive

    def recording_validate(path, root):
        calls.append(path.name)
        return real_validate(path, root)

    monkeypatch.setattr(download, "validate_archive", recording_validate)
    try:
        asset = AssetSpec(
            "archive", f"http://127.0.0.1:{server.server_port}/labels.zip", "TEST",
            expected_bytes=len(data), allowed_hosts=("127.0.0.1",),
            download_url_verified=True,
        )
        policy = DirectOnlyPolicy(
            route_status="VERIFIED_DIRECT", allowed_hosts=frozenset({"127.0.0.1"}),
            allow_insecure_localhost=True,
        )
        receipt = AssetDownloader(tmp_path).fetch(asset, policy)
        assert receipt.status == "VERIFIED"
        assert calls == ["labels.zip.part"]
    finally:
        server.shutdown()
        thread.join(timeout=2)
