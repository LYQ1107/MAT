"""Adapters for detector/predictor outputs kept separate from evaluator truth."""

from .sleap_pose import SleapPosePredictionAdapter

__all__ = ["SleapPosePredictionAdapter"]
