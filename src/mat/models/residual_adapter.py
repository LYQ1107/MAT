from __future__ import annotations

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = object


class ResidualAdapter(nn.Module if torch is not None else object):
    """Optional small adapter; never silently enables target training."""
    def __init__(self, dimension: int):
        if torch is None:
            raise RuntimeError("PyTorch is required for ResidualAdapter")
        super().__init__()
        self.projection = nn.Linear(dimension, dimension)

    def forward(self, features):
        return features + self.projection(features)

