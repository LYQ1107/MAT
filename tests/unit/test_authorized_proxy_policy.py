import json

from mat.assets import AuthorizedProxyPolicy, AssetDownloader
from mat.assets.catalog import AssetSpec


def test_authorized_proxy_is_opt_in_and_preserves_child_proxy_without_receipt_secret(tmp_path):
    parent = {"HTTPS_PROXY": "http://user:secret@example.invalid:7890", "PATH": "/bin"}
    policy = AuthorizedProxyPolicy(allowed_hosts=frozenset({"storage.googleapis.com"}))
    child = policy.child_env(parent)
    assert policy.route_status == "AUTHORIZED_PROXY"
    assert policy.trust_env is True
    assert child["HTTPS_PROXY"] == parent["HTTPS_PROXY"]
    policy.assert_route_approved("storage.googleapis.com")
    policy.validate_url("https://storage.googleapis.com/sleap-data/file.slp", "sleap")

    # A route failure receipt contains only the URL origin and policy status,
    # never the ambient proxy value.
    asset = AssetSpec(
        "sleap", "https://storage.googleapis.com/sleap-data/file.slp", "TEST",
        allowed_hosts=("storage.googleapis.com",), download_url_verified=True,
    )
    downloader = AssetDownloader(tmp_path, timeout=(0.01, 0.01))
    downloader._request = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(parent["HTTPS_PROXY"]))
    receipt = downloader.fetch(asset, policy)
    raw = json.dumps(receipt.__dict__, sort_keys=True)
    assert "secret" not in raw and "example.invalid:7890" not in raw


def test_direct_only_still_blocks_unverified_route():
    policy = __import__("mat.assets", fromlist=["DirectOnlyPolicy"]).DirectOnlyPolicy()
    try:
        policy.assert_route_approved("storage.googleapis.com")
    except Exception as exc:
        assert type(exc).__name__ == "RouteNotApprovedError"
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("direct-only policy unexpectedly approved a route")
