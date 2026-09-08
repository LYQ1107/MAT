import numpy as np
from pathlib import Path

from mat.core.types import IdentityDescriptor, PersistentIdentity
from mat.enrollment.base import EnrollmentResult
from mat.identity.longitudinal_gallery import (
    IdentityExemplar, LongitudinalGalleryStore, score_profile,
)
from mat.models.memory_commit_gate import MemoryCommitGate
from mat.core.types import LocalTracklet, SessionSpec
from mat.pipeline.cohort import LongitudinalCohortRunner


def _desc(value, fp="fixture"):
    value = np.asarray(value, dtype=np.float32)
    return IdentityDescriptor(value, np.zeros((0, value.size), np.float32), np.zeros(0, bool), np.zeros(0, np.float32), fp)


def test_anchor_survives_pending_and_pending_is_not_scored(tmp_path):
    d = _desc([1, 0])
    identity = PersistentIdentity("id:a", "cohort", ("s0:t0",), (), (), "s0", {})
    enrollment = EnrollmentResult("cohort", "s0", (identity,), {"id:a": d}, (), 1.0, "H_oracle_reference")
    store = LongitudinalGalleryStore()
    snap = store.create(enrollment)
    pending = IdentityExemplar("p", "id:a", "pending", d, "s1", "s1:t0", 0.8, "fixture", "now")
    proposal = store.propose("cohort", snap.version, pending)
    next_snap = store.commit([proposal], snap.version)
    profile = next_snap.profiles["id:a"]
    assert len(profile.anchors) == 1 and len(profile.pending) == 1 and not profile.confirmed
    assert score_profile(d, profile) == 1.0


def test_memory_gate_requires_two_time_windows_and_no_conflict():
    gate = MemoryCommitGate(high_threshold=0.7, margin_threshold=0.1, quality_threshold=0.6)
    waiting = gate.evaluate(top1_score=0.9, top2_score=0.5, tracklet_quality=0.9,
                            independent_window_count=1, has_conflict=False)
    assert waiting.action == "wait"
    promoted = gate.evaluate(top1_score=0.9, top2_score=0.5, tracklet_quality=0.9,
                             independent_window_count=2, has_conflict=False)
    assert promoted.action == "promote"
    rejected = gate.evaluate(top1_score=0.9, top2_score=0.5, tracklet_quality=0.9,
                             independent_window_count=2, has_conflict=True)
    assert rejected.action == "reject"


def test_runner_quarantines_first_session_then_promotes_pending():
    d = _desc([1, 0])
    identity = PersistentIdentity("id:a", "cohort", ("s0:t0",), (), (), "s0", {})
    enrollment = EnrollmentResult("cohort", "s0", (identity,), {"id:a": d}, (), 1.0, "H_oracle_reference")
    store = LongitudinalGalleryStore()
    start = store.create(enrollment)
    sessions = [
        SessionSpec("s1", "cohort", "2024-01-01T00:00:00", "cam", Path("s1.jsonl"), {}),
        SessionSpec("s2", "cohort", "2024-01-02T00:00:00", "cam", Path("s2.jsonl"), {}),
    ]

    def extract(session, _snapshot):
        tracklet = LocalTracklet(f"{session.session_uid}:t0", session.session_uid, "cam", ["o"], [(0.0, 1.0)])
        return [tracklet], {tracklet.tracklet_uid: d}, {
            tracklet.tracklet_uid: {"descriptor": d, "quality": 0.9, "independent_windows": 2}
        }

    result = LongitudinalCohortRunner(store).run(sessions, start, extract=extract)
    assert result.status == "SUCCEEDED"
    final = store.snapshot("cohort")
    profile = final.profiles["id:a"]
    assert len(profile.anchors) == 1
    assert len(profile.pending) == 1
    assert len(profile.confirmed) == 1
