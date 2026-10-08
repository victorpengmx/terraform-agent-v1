"""Terrace op: cut slopes into steps (flat ledges, steep risers).

Hand-script evidence (2026-10-20): stacked `raise_region` plateaus gave a regular
"wedding cake" and needed many precisely placed outlines; one pass over a steep peak gave the layered
cliffs of a horn. Steps follow the existing contours, so the model never places them by hand.
"""

from __future__ import annotations

import numpy as np
from pydantic import Field
from scipy.ndimage import gaussian_filter

from terrain.brushes import length_to_blocks
from terrain.noise import fractal_noise
from terrain.ops.character import CharacterParams
from terrain.ops.region import AreaParams, region_weight
from terrain.ops.registry import register_op
from terrain.state import WorldState

FLAT_DEG = (2.0, 8.0)  # slopes below 2° stay untouched, full effect from 8°: plains keep their level, meadows step
LEDGE_MIN, LEDGE_SPAN = 0.35, 0.55  # steepness 0 -> 35% of each step is ledge (soft); 1 -> 90% (sheer risers)
LEDGE_TILT = 0.6  # steepness 0 -> ledges keep 60% of the step's rise (rounded steps); 1 -> flat ledges
JITTER = 0.6  # roughness 1 -> step heights wander by up to 0.6 x step across the slope


class TerraceParams(AreaParams, CharacterParams):
    step: float = Field(ge=2.0, le=32.0, description="Height of each step in blocks.")


@register_op("terrace", params=TerraceParams, stage="terrain")
def terrace(state: WorldState, p: TerraceParams) -> WorldState:
    """Cut slopes into steps: flat ledges between steep risers, following the contours. Layered cliffs and
    horns (steep peak + terrace + banded palette), stepped mesas, paddy hillsides, and cascades (terrace a
    stream's slope before carve_river). Flat ground is left alone; heights move by at most one step.
    """
    h = state.height.astype(np.float64)
    gz, gx = np.gradient(gaussian_filter(h, 1.0, mode="nearest"))
    slope = np.degrees(np.arctan(np.hypot(gx, gz)))
    g = np.clip((slope - FLAT_DEG[0]) / (FLAT_DEG[1] - FLAT_DEG[0]), 0.0, 1.0)
    weight = region_weight(state, p.region, length_to_blocks(state, p.edge)) * g * g * (3.0 - 2.0 * g)

    jitter = JITTER * p.roughness * p.step * fractal_noise(state, 0.12, octaves=3, roughness=0.5, seed=state.seed)
    x = (h + jitter) / p.step
    k = np.floor(x)
    f = x - k  # position within the step, 0..1
    ledge = LEDGE_MIN + LEDGE_SPAN * p.steepness
    tilt = LEDGE_TILT * (1.0 - p.steepness)  # share of the step's height climbed along the ledge
    rise = np.clip((f - ledge) / (1.0 - ledge), 0.0, 1.0)
    profile = np.where(f < ledge, tilt * f / ledge, tilt + (1.0 - tilt) * rise * rise * (3.0 - 2.0 * rise))
    stepped = (k + profile) * p.step - jitter
    state.height = (h + (stepped - h) * weight).astype(np.float32)
    return state
