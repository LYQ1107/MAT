import subprocess

from mat.backends.sleap_nn import SleapNNBackend


def test_run_uses_popen_pipes_for_captured_output(tmp_path, monkeypatch):
    seen = {}

    class FakeProcess:
        pid = 12345
        returncode = 0

        def communicate(self):
            return "version\n", ""

    def fake_popen(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    completed = SleapNNBackend(executable="sleap-nn", env={})._run(
        ["--version"], output_dir=tmp_path, operation="version"
    )

    assert completed.stdout == "version\n"
    assert seen["kwargs"]["stdout"] is subprocess.PIPE
    assert seen["kwargs"]["stderr"] is subprocess.PIPE
    assert "capture_output" not in seen["kwargs"]
