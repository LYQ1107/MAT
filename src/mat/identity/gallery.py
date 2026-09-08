from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
import json
import os
import sqlite3
import uuid

import numpy as np

from mat.core.errors import IntegrityError, ProtocolError, ValidationError
from mat.core.types import IdentityDescriptor, PersistentIdentity, Assignment
from mat.enrollment.base import EnrollmentResult, EvidenceBundle


@dataclass(frozen=True)
class GallerySnapshot:
    cohort_uid: str
    version: str
    identities: dict[str, PersistentIdentity]
    descriptors: dict[str, IdentityDescriptor]
    fingerprint: str


def _pack_descriptor(desc: IdentityDescriptor) -> bytes:
    import io
    buf = io.BytesIO()
    np.savez(buf, global_feature=desc.global_feature, part_features=desc.part_features,
             part_valid=desc.part_valid, part_quality=desc.part_quality,
             encoder_fingerprint=np.asarray(desc.encoder_fingerprint))
    return buf.getvalue()


def _unpack_descriptor(blob: bytes) -> IdentityDescriptor:
    import io
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        fp = str(data["encoder_fingerprint"].item())
        return IdentityDescriptor(data["global_feature"], data["part_features"],
                                  data["part_valid"], data["part_quality"], fp)


class GalleryStore:
    """SQLite single-writer registry with immutable anchors and version snapshots."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), timeout=30, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS versions (
                cohort_uid TEXT NOT NULL, version TEXT NOT NULL, snapshot_json TEXT NOT NULL,
                created_at TEXT NOT NULL, PRIMARY KEY(cohort_uid, version));
            CREATE TABLE IF NOT EXISTS proposals (
                proposal_id TEXT PRIMARY KEY, cohort_uid TEXT NOT NULL, base_version TEXT NOT NULL,
                payload TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY, cohort_uid TEXT NOT NULL, from_version TEXT NOT NULL,
                to_version TEXT NOT NULL, event_type TEXT NOT NULL, payload TEXT NOT NULL,
                created_at TEXT NOT NULL);
        """)

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _snapshot_json(enrollment_or_snapshot, version: str) -> str:
        if isinstance(enrollment_or_snapshot, EnrollmentResult):
            identities = enrollment_or_snapshot.identities
            descriptors = enrollment_or_snapshot.descriptors
            cohort_uid = enrollment_or_snapshot.cohort_uid
            fingerprint = ";".join(sorted({d.encoder_fingerprint for d in descriptors.values()}))
        else:
            identities = tuple(enrollment_or_snapshot.identities.values())
            descriptors = enrollment_or_snapshot.descriptors
            cohort_uid = enrollment_or_snapshot.cohort_uid
            fingerprint = enrollment_or_snapshot.fingerprint
        return json.dumps({
            "cohort_uid": cohort_uid, "version": version, "fingerprint": fingerprint,
            "identities": [{"identity_uid": i.identity_uid, "cohort_uid": i.cohort_uid,
                             "anchor_refs": list(i.anchor_refs), "committed_refs": list(i.committed_refs),
                             "provisional_refs": list(i.provisional_refs),
                             "created_from_session": i.created_from_session, "provenance": i.provenance}
                            for i in identities],
            "descriptors": {uid: {"global_feature": d.global_feature.tolist(),
                                   "part_features": d.part_features.tolist(),
                                   "part_shape": list(d.part_features.shape),
                                   "part_valid": d.part_valid.tolist(), "part_quality": d.part_quality.tolist(),
                                   "encoder_fingerprint": d.encoder_fingerprint}
                            for uid, d in descriptors.items()},
        }, sort_keys=True)

    @staticmethod
    def _from_json(payload: str) -> GallerySnapshot:
        raw = json.loads(payload)
        identities = {}
        for item in raw["identities"]:
            identities[item["identity_uid"]] = PersistentIdentity(
                item["identity_uid"], item["cohort_uid"], tuple(item["anchor_refs"]),
                tuple(item["committed_refs"]), tuple(item["provisional_refs"]),
                item["created_from_session"], item["provenance"])
        descriptors = {}
        for uid, v in raw["descriptors"].items():
            parts = np.asarray(v["part_features"], dtype=np.float32)
            shape = tuple(v.get("part_shape", parts.shape))
            if parts.size == 0 and len(shape) == 2:
                parts = np.zeros(shape, dtype=np.float32)
            descriptors[uid] = IdentityDescriptor(v["global_feature"], parts,
                                                   v["part_valid"], v["part_quality"], v["encoder_fingerprint"])
        return GallerySnapshot(raw["cohort_uid"], raw["version"], identities, descriptors, raw["fingerprint"])

    def create(self, enrollment: EnrollmentResult) -> GallerySnapshot:
        if enrollment.status.startswith("BLOCKED"):
            raise ProtocolError(f"cannot create gallery from {enrollment.status} enrollment")
        fingerprints = {descriptor.encoder_fingerprint for descriptor in enrollment.descriptors.values()}
        if len(fingerprints) != 1:
            raise ProtocolError("all gallery anchors must share one encoder fingerprint")
        version = "v0"
        payload = self._snapshot_json(enrollment, version)
        with self._conn:
            exists = self._conn.execute("SELECT 1 FROM versions WHERE cohort_uid=?", (enrollment.cohort_uid,)).fetchone()
            if exists:
                raise ProtocolError(f"cohort already has a gallery: {enrollment.cohort_uid}")
            self._conn.execute("INSERT INTO versions VALUES (?,?,?,?)",
                               (enrollment.cohort_uid, version, payload, datetime.now(timezone.utc).isoformat()))
            self._conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (f"head:{enrollment.cohort_uid}", version))
            self._conn.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?)",
                               (str(uuid.uuid4()), enrollment.cohort_uid, "none", version, "create",
                                json.dumps({"provenance": enrollment.provenance}), datetime.now(timezone.utc).isoformat()))
        return self._from_json(payload)

    def _head(self, cohort_uid: str) -> str:
        row = self._conn.execute("SELECT value FROM meta WHERE key=?", (f"head:{cohort_uid}",)).fetchone()
        if not row:
            raise ProtocolError(f"no gallery for cohort {cohort_uid}")
        return str(row[0])

    def snapshot(self, cohort_uid: str, version: str | None = None) -> GallerySnapshot:
        version = version or self._head(cohort_uid)
        row = self._conn.execute("SELECT snapshot_json FROM versions WHERE cohort_uid=? AND version=?",
                                 (cohort_uid, version)).fetchone()
        if not row:
            raise ProtocolError(f"unknown gallery version {cohort_uid}/{version}")
        return self._from_json(row[0])

    def propose(self, assignment: Assignment, evidence: EvidenceBundle | dict[str, Any]) -> str:
        if assignment.persistent_uid is None:
            raise ProtocolError("cannot propose an unknown assignment")
        cohort = str((evidence.provenance if isinstance(evidence, EvidenceBundle) else evidence).get("cohort_uid", ""))
        if not cohort:
            cohort = str((evidence.provenance if isinstance(evidence, EvidenceBundle) else evidence).get("cohort", ""))
        if not cohort:
            raise ValidationError("evidence must identify cohort_uid")
        base = self._head(cohort)
        if isinstance(evidence, EvidenceBundle):
            data = {"cohort_uid": cohort, "assignment": assignment.__dict__,
                    "descriptor": {"global_feature": evidence.descriptor.global_feature.tolist(),
                                   "part_features": evidence.descriptor.part_features.tolist(),
                                   "part_shape": list(evidence.descriptor.part_features.shape),
                                   "part_valid": evidence.descriptor.part_valid.tolist(),
                                   "part_quality": evidence.descriptor.part_quality.tolist(),
                                   "encoder_fingerprint": evidence.descriptor.encoder_fingerprint},
                    "quality": evidence.quality, "independent_windows": evidence.independent_windows,
                    "source_observation_uids": evidence.source_observation_uids}
        else:
            data = dict(evidence)
            data["assignment"] = assignment.__dict__
        proposal_id = str(uuid.uuid4())
        self._conn.execute("INSERT INTO proposals VALUES (?,?,?,?,?,?)",
                           (proposal_id, cohort, base, json.dumps(data, default=str), "proposed",
                            datetime.now(timezone.utc).isoformat()))
        return proposal_id

    def commit(self, proposal_ids: list[str], expected_version: str) -> GallerySnapshot:
        if not proposal_ids:
            raise ValidationError("commit requires at least one proposal")
        rows = [self._conn.execute("SELECT cohort_uid,base_version,payload,status FROM proposals WHERE proposal_id=?", (pid,)).fetchone()
                for pid in proposal_ids]
        if any(row is None for row in rows):
            raise ProtocolError("unknown proposal")
        cohort = rows[0][0]
        if any(row[0] != cohort or row[1] != expected_version or row[3] != "proposed" for row in rows):
            raise ProtocolError("proposal cohort/version/status mismatch")
        current = self.snapshot(cohort)
        if current.version != expected_version:
            raise ProtocolError("expected_version is stale")
        identities = dict(current.identities)
        descriptors = dict(current.descriptors)
        for row in rows:
            data = json.loads(row[2])
            assignment = data["assignment"]
            uid = assignment["persistent_uid"]
            if uid not in identities:
                raise ProtocolError(f"proposal references unknown identity {uid}")
            descriptor = data.get("descriptor")
            if descriptor:
                part_values = np.asarray(descriptor["part_features"], dtype=np.float32)
                part_shape = tuple(descriptor.get("part_shape", part_values.shape))
                if len(part_shape) != 2:
                    part_shape = current.descriptors[uid].part_features.shape
                if part_values.size == 0 and len(part_shape) == 2:
                    part_values = np.zeros(part_shape, dtype=np.float32)
                desc = IdentityDescriptor(descriptor["global_feature"], part_values,
                                          descriptor["part_valid"], descriptor["part_quality"], descriptor["encoder_fingerprint"])
                if desc.encoder_fingerprint != current.descriptors[uid].encoder_fingerprint:
                    raise ProtocolError("feature-space fingerprint mismatch")
                # ``descriptors`` is the immutable S0 anchor index used by B0.
                # Candidate evidence is retained in the proposal/event audit
                # trail, while confirmed/quarantine exemplars live in the
                # longitudinal gallery.  Never replace an anchor descriptor
                # in this legacy static store.
            refs = list(identities[uid].committed_refs)
            refs.extend(data.get("source_observation_uids", []))
            identities[uid] = PersistentIdentity(identities[uid].identity_uid, identities[uid].cohort_uid,
                                                 identities[uid].anchor_refs, tuple(dict.fromkeys(refs)),
                                                 identities[uid].provisional_refs, identities[uid].created_from_session,
                                                 identities[uid].provenance)
        new_version = f"v{int(expected_version[1:]) + 1}" if expected_version.startswith("v") and expected_version[1:].isdigit() else str(uuid.uuid4())
        next_snapshot = GallerySnapshot(cohort, new_version, identities, descriptors, current.fingerprint)
        payload = self._snapshot_json(next_snapshot, new_version)
        now = datetime.now(timezone.utc).isoformat()
        with self._conn:
            self._conn.execute("INSERT INTO versions VALUES (?,?,?,?)", (cohort, new_version, payload, now))
            self._conn.execute("UPDATE meta SET value=? WHERE key=?", (new_version, f"head:{cohort}"))
            for pid in proposal_ids:
                self._conn.execute("UPDATE proposals SET status='committed' WHERE proposal_id=?", (pid,))
            self._conn.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?)",
                               (str(uuid.uuid4()), cohort, expected_version, new_version, "commit",
                                json.dumps({"proposal_ids": proposal_ids}), now))
        return next_snapshot

    def reject(self, proposal_id: str, reason: str) -> None:
        with self._conn:
            cur = self._conn.execute("UPDATE proposals SET status=? WHERE proposal_id=? AND status='proposed'",
                                     (f"rejected:{reason}", proposal_id))
            if cur.rowcount != 1:
                raise ProtocolError("proposal missing or already finalized")

    def rollback(self, target_version: str, cohort_uid: str | None = None) -> GallerySnapshot:
        if cohort_uid is None:
            row = self._conn.execute("SELECT cohort_uid FROM versions WHERE version=?", (target_version,)).fetchone()
            if not row:
                raise ProtocolError("target version not found")
            cohort_uid = row[0]
        target = self.snapshot(cohort_uid, target_version)
        current = self.snapshot(cohort_uid)
        new_version = f"v{int(current.version[1:]) + 1}" if current.version.startswith("v") else str(uuid.uuid4())
        rolled = GallerySnapshot(cohort_uid, new_version, target.identities, target.descriptors, target.fingerprint)
        payload = self._snapshot_json(rolled, new_version)
        now = datetime.now(timezone.utc).isoformat()
        with self._conn:
            self._conn.execute("INSERT INTO versions VALUES (?,?,?,?)", (cohort_uid, new_version, payload, now))
            self._conn.execute("UPDATE meta SET value=? WHERE key=?", (new_version, f"head:{cohort_uid}"))
            self._conn.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?)",
                               (str(uuid.uuid4()), cohort_uid, current.version, new_version, "rollback",
                                json.dumps({"target_version": target_version}), now))
        return rolled
