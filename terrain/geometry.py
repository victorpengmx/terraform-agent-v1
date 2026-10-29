"""Surface geometry shared by ops, materials, checks and observations."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def slope_deg(height: np.ndarray, smooth: float = 0.0, mode: str = "reflect") -> np.ndarray:
    """Surface slope in degrees per column (central differences, 1 block apart).

    `smooth`: Gaussian sigma in blocks applied first (`mode` is its edge handling). No dtype conversion:
    pass float64 where the caller needs it, so each call site keeps its exact numbers."""
    if smooth > 0:
        height = gaussian_filter(height, smooth, mode=mode)
    gz, gx = np.gradient(height)
    return np.degrees(np.arctan(np.hypot(gx, gz)))
