import json
from pathlib import Path

from mat.data.gerbil_longitudinal_split import GerbilLongitudinalProtocol, build_gerbil_longitudinal_protocol


def _prepared(tmp_path: Path) -> Path:
    manifest = tmp_path / "prepared" / "sleap_gerbils" / "manifests"
    manifest.mkdir(parents=True)
    sessions = []
    for index in range(23):
        uid = f"session-{index:02d}"
        sessions.append({"session_uid": uid, "labeled_frame_count": 1 if index < 11 else 0,
                         "recording_datetime_if_parseable": None})
    (manifest / "session_inventory.json").write_text(json.dumps({"sessions": sessions}))
    (manifest / "observations.jsonl").write_text("")
    identities = ("female", "male", "pup shaved", "pup unshaved")
    truth = [{"observation_uid": f"o{index}", "session_uid": "session-00",
              "gt_identity": label} for index, label in enumerate(identities)]
    (manifest / "private_pose_identity_truth.jsonl").write_text(
        "\n".join(json.dumps(row) for row in truth) + "\n")
    return tmp_path


def test_prepared_gerbil_protocol_is_labeled_only_and_frozen(tmp_path):
    protocol = build_gerbil_longitudinal_protocol(
        _prepared(tmp_path), output=tmp_path / "protocol.json")
    assert protocol.status == "FROZEN"
    assert len(protocol.all_provider_sessions) == 23
    assert len(protocol.labeled_sessions) == 11
    assert len(protocol.reference_sessions) == 1
    assert protocol.development_sessions
    assert protocol.sealed_test_sessions
    assert not set(protocol.reference_sessions) & set(protocol.development_sessions)
    assert not set(protocol.reference_sessions) & set(protocol.sealed_test_sessions)
    assert protocol.ordering_basis == "deterministic_session_uid_fallback"
    assert "NOT_STRICT" in protocol.provider_identity_evidence
    loaded = GerbilLongitudinalProtocol.read(tmp_path / "protocol.json")
    assert loaded == protocol


def test_protocol_builder_does_not_use_unlabeled_sessions_in_roles(tmp_path):
    protocol = build_gerbil_longitudinal_protocol(
        _prepared(tmp_path), output=tmp_path / "protocol.json")
    roles = set(protocol.all_role_sessions)
    assert roles <= set(protocol.labeled_sessions)
    raw = json.loads((tmp_path / "protocol.json").read_text())
    assert raw["role_counts"]["all_provider_sessions"] == 23
    assert raw["role_counts"]["labeled_sessions"] == 11
