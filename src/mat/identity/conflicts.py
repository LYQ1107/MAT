from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from mat.core.types import LocalTracklet


@dataclass(frozen=True)
class ConflictGraph:
    pairs: frozenset[tuple[str, str]] = frozenset()

    def conflicts(self, left: str, right: str) -> bool:
        key = tuple(sorted((left, right)))
        return key in self.pairs


class ConflictGraphBuilder:
    """Build cannot-link pairs only for same-camera overlapping observations."""

    def build(self, tracklets: Iterable[LocalTracklet]) -> ConflictGraph:
        items = list(tracklets)
        pairs: set[tuple[str, str]] = set()
        for i, left in enumerate(items):
            for right in items[i + 1:]:
                if left.camera_uid != right.camera_uid:
                    continue
                if self._overlaps(left, right):
                    pairs.add(tuple(sorted((left.tracklet_uid, right.tracklet_uid))))
        return ConflictGraph(frozenset(pairs))

    @staticmethod
    def _overlaps(left: LocalTracklet, right: LocalTracklet) -> bool:
        for la, lb in left.time_support:
            for ra, rb in right.time_support:
                if max(la, ra) < min(lb, rb):
                    return True
        return False

