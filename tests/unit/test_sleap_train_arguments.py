from pathlib import Path

from mat.backends.sleap_nn import SleapNNBackend


def test_train_steps_override_is_explicit_and_full_has_none(tmp_path):
    config = tmp_path / "config.yaml"; config.write_text("trainer_config: {}\n")
    train = tmp_path / "train.slp"; train.write_bytes(b"train")
    val = tmp_path / "val.slp"; val.write_bytes(b"val")
    captured = []

    backend = SleapNNBackend(executable="sleap-nn", env={})
    def fake_run(args, **kwargs):
        captured.append([str(x) for x in args])
        out = Path(kwargs["output_dir"]); (out / "models").mkdir(parents=True, exist_ok=True)
        (out / "models" / "best.ckpt").write_bytes(b"checkpoint")
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()
    backend._run = fake_run
    backend.train(config, train, val, tmp_path / "smoke", 2, train_steps_per_epoch=1)
    backend.train(config, train, val, tmp_path / "full", 50)
    assert any("train_steps_per_epoch=1" in x for x in captured[0])
    assert not any("train_steps_per_epoch" in x for x in captured[1])
    assert "trainer_config.max_epochs=50" in captured[1]
