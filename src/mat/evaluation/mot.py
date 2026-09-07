from __future__ import annotations

from mat.core.errors import DependencyUnavailableError


def evaluate_trackeval(*args, **kwargs):
    try:
        import trackeval  # type: ignore
    except ImportError as exc:
        raise DependencyUnavailableError("official TrackEval is not installed") from exc
    raise NotImplementedError("TrackEval adapter requires a verified dataset converter and is not silently approximated")

