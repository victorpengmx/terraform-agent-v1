"""Valley op: carve a channel along a path."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field

from terrain.brushes import blend, length_to_blocks, path_distance, smooth_path
from terrain.ops.registry import Norm, OpParams, Point, register_op
from terrain.state import WorldState


class CarveValleyParams(OpParams):
    path: list[Point] = Field(min_length=2, max_length=32, description="Valley floor line as normalized (u, v) points.")
    width: Norm = Field(gt=0.0, description="Full width at the rim, as a fraction of the shorter side.")
    depth: float = Field(gt=0.0, le=256.0, description="Blocks removed along the valley floor.")
    profile: Literal["v", "u"] = Field(default="v", description="v: river-cut V shape. u: glacial, flat floor and steep walls.")


@register_op("carve_valley", params=CarveValleyParams, stage="terrain")
def carve_valley(state: WorldState, p: CarveValleyParams) -> WorldState:
    """Carve a valley along a path, lowering the terrain (never raising it).

    Depth is relative to the existing terrain, so the floor follows the land. For flowing water use
    carve_river afterwards in the valley; it enforces a downhill bed.
    """
    dist, _ = path_distance(state, smooth_path(state, p.path))
    s = np.clip(dist / (length_to_blocks(state, p.width) / 2), 0.0, 1.0)  # 0 on the floor line, 1 at the rim
    t = 1.0 - s
    shape = t * t * (3.0 - 2.0 * t) if p.profile == "v" else 1.0 - s**4
    state.height = blend(state.height, state.height - p.depth * shape, np.ones_like(shape), "min")
    return state
