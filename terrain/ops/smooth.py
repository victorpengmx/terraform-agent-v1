"""Smoothing op."""

from __future__ import annotations

from pydantic import Field

from terrain.brushes import blend, length_to_blocks
from terrain.ops.region import AreaParams, region_weight
from terrain.ops.registry import Norm, register_op
from terrain.state import WorldState


class SmoothParams(AreaParams):
    strength: Norm = Field(gt=0.0, description="0..1: how far to move toward the blurred terrain.")
    radius: float = Field(default=3.0, ge=0.5, le=32.0, description="Blur radius in blocks; larger removes bigger bumps.")


@register_op("smooth", params=SmoothParams, stage="terrain")
def smooth(state: WorldState, p: SmoothParams) -> WorldState:
    """Soften the terrain: removes spikes, sharp ridges and harsh slopes, everywhere or in a region.

    Use it to make cones and steps read as natural landforms.
    """
    weight = p.strength * region_weight(state, p.region, length_to_blocks(state, p.edge))
    state.height = blend(state.height, 0.0, weight, "smooth", sigma=p.radius)
    return state
