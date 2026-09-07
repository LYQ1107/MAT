from mat.data.sanitize import validate_neutral_manifest


def test_neutral_manifest_rejects_ground_truth_field(tmp_path):
    path = tmp_path / "obs.jsonl"
    path.write_text('{"observation_uid":"o","session_uid":"s","gt_id":"secret"}\n')
    result = validate_neutral_manifest(path)
    assert not result["ok"] and "gt_id" in result["forbidden_fields"]

