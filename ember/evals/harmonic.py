"""Harmonic mean used as the aggregate metric."""
from __future__ import annotations

from typing import Iterable

import numpy as np


def harmonic_mean(values: Iterable[float]) -> float:
    """Return ``len(vals) / sum(1/v)``, or 0.0 if any value is non-finite or <= 0."""
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    for v in vals:
        if not np.isfinite(v) or v <= 0.0:
            return 0.0
    return len(vals) / sum(1.0 / v for v in vals)


__all__ = ["harmonic_mean"]
