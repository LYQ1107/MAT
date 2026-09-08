from pathlib import Path

from mat.cli import resolve_full_checkpoint, resolve_smoke_checkpoint


def test_full_resolution_never_returns_smoke_checkpoint(tmp_path):
    smoke = tmp_path / "runs" / "sleap_gerbils_pose_smoke" / "models" / "best.ckpt"
    smoke.parent.mkdir(parents=True)
    smoke.write_bytes(b"smoke")
    assert resolve_full_checkpoint(tmp_path) is None
    assert resolve_full_checkpoint(tmp_path, smoke) is None
    assert resolve_smoke_checkpoint(tmp_path) == smoke


def test_full_resolution_accepts_only_full_best_checkpoint(tmp_path):
    full = tmp_path / "runs" / "sleap_gerbils_pose_full" / "models" / "best.ckpt"
    full.parent.mkdir(parents=True)
    full.write_bytes(b"full")
    other = full.parent / "last.ckpt"
    other.write_bytes(b"last")
    assert resolve_full_checkpoint(tmp_path) == full
    assert resolve_full_checkpoint(tmp_path, full) == full
    assert resolve_full_checkpoint(tmp_path, other) is None
