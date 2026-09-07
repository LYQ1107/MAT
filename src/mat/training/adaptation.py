from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptationDecision:
    status: str
    reason: str
    encoder_fingerprint_before: str | None = None
    encoder_fingerprint_after: str | None = None


def gate_unlabelled_adaptation(has_valid_signal: bool, fingerprint_before: str | None) -> AdaptationDecision:
    if not has_valid_signal:
        return AdaptationDecision("SKIPPED", "no reliable unlabeled self-supervision; target update is not fabricated", fingerprint_before, fingerprint_before)
    return AdaptationDecision("REQUIRES_EXPLICIT_EXPERIMENT", "re-encode all historical references before switching feature space", fingerprint_before, None)

