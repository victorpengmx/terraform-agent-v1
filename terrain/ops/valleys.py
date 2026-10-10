"""Valley op: carve a channel along a path.

Cross-section (DESIGN §5.5): a floor (none for V, flat for U), then walls shaped like an S: a concave talus
apron at the toe, the steep face a little below mid-wall, and a broader convex shoulder that eases into the
land at the rim, so the walls don't run at full steepness into the plain (the old U profile, 1 - s^4, was
steepest exactly at the rim: a ship-hull section). `steepness` concentrates the drop into a narrower,
steeper face; `roughness` lets the width swell and narrow along the path.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field
from scipy.special import betainc

from terrain.brushes import blend, length_to_blocks, path_distance, path_wobble, smooth_path
from terrain.ops.character import CharacterParams
from terrain.ops.registry import Norm, Point, register_op
from terrain.state import WorldState

FLOOR = {"v": 0.0, "u": 0.3}  # share of the half width that is flat valley floor
# Walls follow a beta CDF from toe (0) to rim (1). FACE puts the steepest point ~0.42 of the way up, so the
# shoulder is broader than the talus apron; steepness sets the concentration: 0 -> a soft S (max slope 1.6x
# the wall's mean, close to the old V smoothstep), 1 -> a distinct face, 2.5x. Both exponents stay >= 2, so
# toe and shoulder are rounded. Calibration: EVIDENCE.md 2026-10-22 (a first try with the face at 0.6 kept
# the steep top and flattened V bottoms; reverted).
FACE = 0.45
CONCENTRATION = (4.5, 10.0)
WOBBLE = 0.3  # roughness 1 -> the width varies by up to ±30% along the path


def valley_profile(s: np.ndarray, profile: str, steepness: float) -> np.ndarray:
    """Share of the depth removed at relative distance s from the floor line (0) to the rim (1)."""
    f = FLOOR[profile]
    t = np.clip((np.asarray(s, dtype=np.float64) - f) / (1.0 - f), 0.0, 1.0)
    k = CONCENTRATION[0] + (CONCENTRATION[1] - CONCENTRATION[0]) * steepness
    return 1.0 - betainc(k * FACE, k * (1.0 - FACE), t)


class CarveValleyParams(CharacterParams):
    path: list[Point] = Field(min_length=2, max_length=32, description="Valley floor line as normalized (u, v) points.")
    width: Norm = Field(gt=0.0, description="Full width at the rim, as a fraction of the shorter side.")
    depth: float = Field(gt=0.0, le=256.0, description="Blocks removed along the valley floor.")
    profile: Literal["v", "u"] = Field(default="v", description="v: river-cut V shape. u: glacial, flat floor and steep walls.")


@register_op("carve_valley", params=CarveValleyParams, stage="terrain")
def carve_valley(state: WorldState, p: CarveValleyParams) -> WorldState:
    """Carve a valley along a path, lowering the terrain (never raising it).

    Walls rise from a gentle talus apron at the foot, through a steep face, to a rounded shoulder that
    blends into the surrounding land. Depth relative to width sets how steep the walls are overall (depth
    above ~1/3 of the width in blocks gives cliff-like walls); steepness (or character) shapes them, from
    broad slopes to a sheer face. roughness makes the valley widen and narrow along its length.
    Depth is relative to the existing terrain, so the floor follows the land. For flowing water use
    carve_river afterwards in the valley; it enforces a downhill bed.
    """
    points = smooth_path(state, p.path)
    dist, t = path_distance(state, points)
    half = length_to_blocks(state, p.width) / 2
    half = half * (1.0 + WOBBLE * p.roughness * path_wobble(state, points, t, half))
    shape = valley_profile(dist / half, p.profile, p.steepness)
    state.height = blend(state.height, state.height - p.depth * shape, np.ones_like(shape), "min")
    return state
