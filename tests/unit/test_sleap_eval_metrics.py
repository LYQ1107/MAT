import json

import numpy as np

from mat.backends.sleap_nn import SleapNNBackend


def test_evaluate_prefers_sleap_json_sibling_for_object_npz(tmp_path):
    gt = tmp_path / "gt.slp"
    pred = tmp_path / "pred.slp"
    gt.write_bytes(b"gt")
    pred.write_bytes(b"pred")
    backend = SleapNNBackend(executable="sleap-nn", env={})

    def fake_run(_args, **kwargs):
        output_dir = kwargs["output_dir"]
        npz = output_dir / "metrics_official.npz"
        np.savez(npz, metrics=np.array({"mOKS": np.float64(0.42)}, dtype=object))
        npz.with_suffix(".json").write_text(json.dumps({"mOKS": 0.42}), encoding="utf-8")
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    backend._run = fake_run
    result = backend.evaluate(gt, pred, tmp_path / "eval")

    assert result["status"] == "SUCCEEDED"
    assert result["metrics"] == {"mOKS": 0.42}
