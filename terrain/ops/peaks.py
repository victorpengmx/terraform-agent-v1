"""Peak-shaped terrain ops."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import Field
from scipy.ndimage import map_coordinates

from terrain.brushes import block_grid, blend, circle_distance, length_to_blocks, radial_weight, to_blocks
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
