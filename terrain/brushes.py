"""Brush primitives (DESIGN §5.4): nearly every terrain op is mask × falloff × shape × blend.

All functions work on the `[z, x]` grid in block units. Points come in normalized (u, v) and are
converted here, so ops never do coordinate math themselves.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter

from terrain.state import WorldState

FalloffKind = Literal["linear", "smoothstep", "gaussian"]
BlendMode = Literal["add", "max", "min", "lerp_to_target", "smooth"]


# ---------------------------------------------------------------- coordinates


def to_blocks(state: WorldState, point: Sequence[float]) -> tuple[float, float]:
    """Normalized (u, v) -> block (x, z)."""
    return point[0] * (state.width - 1), point[1] * (state.depth - 1)


def length_to_blocks(state: WorldState, fraction: float) -> float:
    """Normalized length (fraction of the shorter side) -> blocks."""
    return fraction * (min(state.width, state.depth) - 1)


def block_grid(state: WorldState) -> tuple[np.ndarray, np.ndarray]:
    """Block-coordinate grids (x, z), each shaped (depth, width)."""
    return np.meshgrid(np.arange(state.width, dtype=np.float32), np.arange(state.depth, dtype=np.float32))


# ---------------------------------------------------------------- distances and masks


def circle_distance(state: WorldState, center: Sequence[float]) -> np.ndarray:
    """Distance in blocks from every column to `center`."""
    xx, zz = block_grid(state)
    cx, cz = to_blocks(state, center)
    return np.hypot(xx - cx, zz - cz)


def path_distance(state: WorldState, points: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Distance in blocks to a polyline, and the arc-length parameter t in [0, 1] of the nearest point."""
    if len(points) < 2:
        raise ValueError("a path needs at least 2 points")
    xx, zz = block_grid(state)
    pts = np.array([to_blocks(state, p) for p in points], dtype=np.float64)
    seg_len = np.hypot(*(pts[1:] - pts[:-1]).T)
    total = float(seg_len.sum()) or 1.0
    start_s = np.concatenate([[0.0], np.cumsum(seg_len)[:-1]])

    best_d = np.full(xx.shape, np.inf)
    best_t = np.zeros(xx.shape)
    for (ax, az), (bx, bz), s0, L in zip(pts[:-1], pts[1:], start_s, seg_len):
        if L == 0:
            continue
        f = np.clip(((xx - ax) * (bx - ax) + (zz - az) * (bz - az)) / (L * L), 0.0, 1.0)
        d = np.hypot(xx - (ax + f * (bx - ax)), zz - (az + f * (bz - az)))
        closer = d < best_d
        best_d = np.where(closer, d, best_d)
        best_t = np.where(closer, (s0 + f * L) / total, best_t)
    return best_d.astype(np.float32), best_t.astype(np.float32)


def polygon_inside(state: WorldState, points: Sequence[Sequence[float]]) -> np.ndarray:
    """Boolean mask of columns inside a polygon (even-odd rule)."""
    if len(points) < 3:
        raise ValueError("a polygon needs at least 3 points")
    xx, zz = block_grid(state)
    pts = [to_blocks(state, p) for p in points]
    inside = np.zeros(xx.shape, dtype=bool)
    for (x1, z1), (x2, z2) in zip(pts, pts[1:] + pts[:1]):
        if z1 == z2:
            continue
        crosses = (zz >= min(z1, z2)) & (zz < max(z1, z2))
        x_at = x1 + (zz - z1) * (x2 - x1) / (z2 - z1)
        inside ^= crosses & (xx < x_at)
    return inside


def polygon_distance(state: WorldState, points: Sequence[Sequence[float]]) -> np.ndarray:
    """Signed distance in blocks to a polygon's edge: negative inside, positive outside."""
    edge, _ = path_distance(state, list(points) + [points[0]])
    return np.where(polygon_inside(state, points), -edge, edge)


def feature_mask(state: WorldState, feature_id: str) -> np.ndarray:
    key = f"feature:{feature_id}"
    if key not in state.masks:
        known = sorted(k.removeprefix("feature:") for k in state.masks if k.startswith("feature:"))
        raise KeyError(f"unknown feature {feature_id!r}; known features: {', '.join(known) or 'none'}")
    return state.masks[key]


# ---------------------------------------------------------------- falloff and blending


def falloff(t: np.ndarray, kind: FalloffKind = "smoothstep") -> np.ndarray:
    """Map t in [0, 1] (0 = edge, 1 = core) to a weight in [0, 1]; monotonic non-decreasing in t."""
    t = np.clip(t, 0.0, 1.0)
    if kind == "linear":
        return t
    if kind == "smoothstep":
        return t * t * (3.0 - 2.0 * t)
    if kind == "gaussian":
        lo = np.exp(-4.5)  # bell value at the edge; rescaled so the edge is exactly 0
        return (np.exp(-4.5 * (1.0 - t) ** 2) - lo) / (1.0 - lo)
    raise ValueError(f"unknown falloff {kind!r}")


def radial_weight(dist: np.ndarray, radius: float, kind: FalloffKind = "smoothstep") -> np.ndarray:
    """1 at dist=0, fading to 0 at dist=radius."""
    return falloff(1.0 - dist / radius, kind)


def edge_weight(signed_dist: np.ndarray, edge: float, kind: FalloffKind = "smoothstep") -> np.ndarray:
    """1 inside a shape, fading to 0 over `edge` blocks outside it (signed distance: negative inside)."""
    if edge <= 0:
        return (signed_dist <= 0).astype(np.float32)
    return falloff(1.0 - np.clip(signed_dist, 0.0, None) / edge, kind)


def blend(height: np.ndarray, value: np.ndarray | float, weight: np.ndarray, mode: BlendMode, sigma: float = 2.0) -> np.ndarray:
    """Combine `height` with `value` under `weight` (0 = untouched, 1 = full effect). Returns a new array.

    add: height + value·w          max: raise toward value only     min: lower toward value only
    lerp_to_target: move toward value by w                           smooth: blur (value unused)
    """
    w = np.clip(weight, 0.0, 1.0)
    if mode == "add":
        out = height + value * w
    elif mode == "max":
        out = height + np.maximum(value - height, 0.0) * w
    elif mode == "min":
        out = height + np.minimum(value - height, 0.0) * w
    elif mode == "lerp_to_target":
        out = height + (value - height) * w
    elif mode == "smooth":
        out = height + (gaussian_filter(height, sigma=sigma, mode="nearest") - height) * w
    else:
        raise ValueError(f"unknown blend mode {mode!r}")
    return out.astype(np.float32)
