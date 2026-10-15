"""Peak- and ridge-shaped terrain ops."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import Field
from scipy.ndimage import map_coordinates

from terrain.brushes import (block_grid, blend, circle_distance, length_to_blocks, path_distance, path_wobble,
                             radial_weight, smooth_path, to_blocks)
from terrain.noise import fractal_noise
from terrain.ops.character import SCALES, CharacterParams
from terrain.ops.registry import Norm, Point, register_op
from terrain.state import WorldState

WARP = 0.5  # at irregularity 1, the outline's distance scale varies by ±50% (footprint up to 2× radius)
GULLY_DEPTH = 0.35  # at ridges 1, gullies remove up to 35% of the local height on mid-slopes
# steepness -> profile exponent ("sharpness"): 0 -> 0.5 (broad dome), 0.6 -> 1.5 (mountain), 1 -> 4 (spire)
SHARPNESS = (0.5, 4.0)
SHARPNESS_CURVE = 1.25
RIDGES_PER_ROUGHNESS = 1.5  # roughness -> ridges = 1.5 x roughness (capped at 1); irregularity = roughness


def sharpness(steepness: float) -> float:
    lo, hi = SHARPNESS
    return lo * (hi / lo) ** (steepness**SHARPNESS_CURVE)


class RaisePeakParams(CharacterParams):
    center: Point = Field(description="Peak centre as normalized (u, v); (0, 0) is the north-west corner.")
    radius: Norm = Field(gt=0.0, description="Footprint radius as a fraction of the area's shorter side.")
    height: float = Field(gt=0.0, le=384.0, description="Blocks added at the centre, above the current surface.")
    irregularity: Norm | None = Field(
        default=None,
        description="0 = perfectly round outline; 0.5 = natural lobes and spurs; 1 = strongly lobed. "
        "The footprint can reach up to radius × (1 + irregularity). Default: from roughness.",
    )
    ridges: Norm | None = Field(
        default=None,
        description="0 = smooth flanks; 1 = pronounced ridges and gullies running down the slopes. Default: from roughness.",
    )

    DERIVED: ClassVar[dict[str, tuple[str, ...]]] = {"character": SCALES, "roughness": ("irregularity", "ridges")}

    @classmethod
    def derive(cls, data: dict[str, Any]) -> dict[str, Any]:
        r = data.get("roughness")
        if not isinstance(r, (int, float)):
            return data  # field validation reports it
        derived = {"irregularity": float(r), "ridges": min(1.0, RIDGES_PER_ROUGHNESS * float(r))}
        return {**data, **{k: v for k, v in derived.items() if data.get(k) is None}}


@register_op("raise_peak", params=RaisePeakParams, stage="terrain")
def raise_peak(state: WorldState, p: RaisePeakParams) -> WorldState:
    """Raise a single peak (mountain, hill) on top of the existing terrain.

    Adds height that is largest at the centre and fades smoothly to zero at the radius, so the peak
    blends into the surrounding surface. steepness shapes the profile (low = broad dome, high = narrow
    spire); roughness gives a natural lobed outline and ridges/gullies (round cone at 0). Calling it on an
    existing peak makes that peak taller.
    """
    # Distances in blocks, so the footprint stays circular on non-square areas.
    dist = circle_distance(state, p.center)
    r = length_to_blocks(state, p.radius)

    if p.irregularity > 0:
        # Domain warp: scale the distance field by smooth noise, so contours become lobed. The centre
        # (dist 0) is unchanged, so the summit keeps its height.
        warp = fractal_noise(state, feature_size=p.radius * 0.8, octaves=3, roughness=0.5)
        dist = dist * (1.0 + WARP * p.irregularity * warp)

    profile = radial_weight(dist, r) ** sharpness(p.steepness)

    if p.ridges > 0:
        profile = profile * (1.0 - GULLY_DEPTH * p.ridges * _gullies(state, p, dist / r) * 4.0 * profile * (1.0 - profile))

    state.height = blend(state.height, p.height, profile, "add")
    return state


def _gullies(state: WorldState, p: RaisePeakParams, rel_dist: np.ndarray) -> np.ndarray:
    """Gully strength in [0, 1] (0 on ridges): irregular periodic noise around the peak, so gullies have
    uneven spacing, width and depth, and curve as they run downhill. Multiplied by 4·p·(1−p), it is
    zero at the summit and the foot."""
    xx, zz = block_grid(state)
    cx, cz = to_blocks(state, p.center)
    rng = np.random.default_rng(state.seed + 1)
    wiggle = fractal_noise(state, feature_size=p.radius * 0.6, octaves=2, roughness=0.5, seed=state.seed + 2)
    angle = np.arctan2(zz - cz, xx - cx) + 0.6 * wiggle + rng.uniform(-0.8, 0.8) * rel_dist  # curve and twist
    n = int(rng.integers(7, 14))  # control points around the circumference
    ring = rng.uniform(-1.0, 1.0, n)
    pos = (angle / (2 * np.pi) % 1.0) * n + n  # index into the ring tiled 3× (periodic, cubic-smooth)
    noise = map_coordinates(np.tile(ring, 3), [pos], order=3, mode="wrap")
    t = np.clip(noise * 1.5, 0.0, 1.0)  # only the positive part cuts gullies; the rest stay ridges
    return t * t * (3.0 - 2.0 * t)


# raise_ridge: an elongated landform along a path (DESIGN §5.5). Same cross-section as raise_peak along the
# path's normal; the crest varies into summits and saddles, the width swells and narrows, the ends taper.
SADDLE_DEPTH = 0.4  # roughness 1 -> saddles dip to 60% of the crest height
SUMMIT_SPACING = 3.0  # summits/saddles repeat about every 3 base widths along the crest (min 24 blocks)
RIDGE_WOBBLE = 0.3  # roughness 1 -> the base width varies by up to ±30% along the path


class RaiseRidgeParams(CharacterParams):
    path: list[Point] = Field(min_length=2, max_length=32, description="Crest line as normalized (u, v) points.")
    height: float = Field(gt=0.0, le=384.0, description="Blocks added along the crest at its highest points.")
    width: Norm = Field(gt=0.0, description="Base width across both flanks, as a fraction of the area's shorter side.")


@register_op("raise_ridge", params=RaiseRidgeParams, stage="terrain")
def raise_ridge(state: WorldState, p: RaiseRidgeParams) -> WorldState:
    """Raise a long ridge along a path: mountain ranges, ridgelines, spurs and long valley flanks.

    Use one raise_ridge per range or flank, then raise_peak for individual summits on it. steepness shapes
    the flanks like raise_peak's (low = broad, high = sharp crest); roughness varies the crest into summits
    and saddles and lets the width swell and narrow. The ridge tapers at both ends instead of stopping in a
    wall. Adds height on top of the existing terrain.
    """
    points = smooth_path(state, p.path)
    dist, t = path_distance(state, points)
    half = length_to_blocks(state, p.width) / 2
    half_local = half * (1.0 + RIDGE_WOBBLE * p.roughness * path_wobble(state, points, t, half))
    curve = np.array([to_blocks(state, q) for q in points])
    length = max(float(np.hypot(*np.diff(curve, axis=0).T).sum()), 1.0)
    s = t.astype(np.float64) * length  # blocks along the crest

    spacing = max(SUMMIT_SPACING * 2 * half, 24.0)
    phase = np.random.default_rng(state.seed + 3).uniform(0.0, 2 * np.pi, 2)
    wave = (np.sin(2 * np.pi * s / spacing + phase[0]) + 0.5 * np.sin(2 * np.pi * s / (0.43 * spacing) + phase[1])) / 1.5
    crest = 1.0 - SADDLE_DEPTH * p.roughness * (1.0 - wave) / 2.0  # 1 on summits, lower in saddles
    end = np.clip(np.minimum(s, length - s) / max(half, 1.0), 0.0, 1.0)
    taper = end * end * (3.0 - 2.0 * end)

    profile = radial_weight(dist, half_local) ** sharpness(p.steepness)
    state.height = blend(state.height, p.height, crest * taper * profile, "add")
    return state
