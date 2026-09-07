from __future__ import annotations

import numpy as np


def cosine_pair_loss(anchor, positive, negative, margin: float = 0.2):
    """Numpy diagnostic; training callers should use a differentiable torch loss."""
    a, p, n = np.asarray(anchor), np.asarray(positive), np.asarray(negative)
    sim = lambda x, y: np.sum(x * y, axis=-1) / np.maximum(np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1), 1e-9)
    return np.maximum(0.0, margin - sim(a, p) + sim(a, n)).mean()

