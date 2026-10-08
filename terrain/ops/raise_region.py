"""Region ops: raise or lower an area given as a shared `region` (terrain/ops/region.py)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from terrain.brushes import blend, edge_weight, length_to_blocks
from terrain.ops.region import EDGE_HELP, REGION_HELP, Region, region_shape
from terrain.ops.registry import OpParams, register_op
from terrain.state import WorldState


class RaiseRegionParams(OpParams):
    region: Region = Field(description=f"Area to raise or lower: {REGION_HELP}.")
    height: float = Field(ge=-256.0, le=256.0, description="Blocks to raise (negative lowers, e.g. to dig a lake basin).")
    edge: float = Field(default=0.05, ge=0.0, le=0.5, description=EDGE_HELP)
    mode: Literal["add", "plateau", "basin"] = Field(
        default="add",
        description="add: shift the terrain by height, keeping its shape. plateau: raise to a flat top at "
        "(mean height inside) + height, never lowering anything. basin: dig a flat floor at (lowest point of "
        "the outline) + height (height < 0), never raising anything; always holds water (craters, lake beds).",
    )

    @model_validator(mode="after")
    def check_height(self) -> RaiseRegionParams:
        if self.height == 0:
            raise ValueError("height must not be 0")
        if self.mode == "plateau" and self.height < 0:
            raise ValueError("plateau mode needs a positive height; use mode 'add' with a negative height to lower")
        if self.mode == "basin" and self.height > 0:
            raise ValueError("basin mode needs a negative height (depth below the rim)")
        return self


@register_op("raise_region", params=RaiseRegionParams, stage="terrain")
def raise_region(state: WorldState, p: RaiseRegionParams) -> WorldState:
    """Raise (or lower) an area: plateaus, mesas, ridges, terraces, craters, basins.

    The edge fades smoothly into the surrounding terrain over `edge`.
    """
    inside, signed = region_shape(state, p.region)
    weight = edge_weight(signed, length_to_blocks(state, p.edge))
    if p.mode == "add":
        state.height = blend(state.height, p.height, weight, "add")
    elif p.mode == "plateau":
        level = float(state.height[inside].mean()) + p.height
        state.height = blend(state.height, np.float32(level), weight, "max")
    else:
        # The rim is the region's outline: columns within a block of its edge, inside or out.
        rim = np.abs(signed) <= 1.0
        level = float(state.height[rim].min()) + p.height
        state.height = blend(state.height, np.float32(level), weight, "min")
    return state
