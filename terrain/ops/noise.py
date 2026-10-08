"""Base noise op."""

from __future__ import annotations

import numpy as np
from pydantic import Field

from terrain.noise import detail_octaves, fractal_noise
from terrain.ops.character import RoughnessParams
from terrain.ops.registry import register_op
from terrain.state import WorldState

PERSISTENCE = (0.1, 0.9)  # roughness 0 -> each finer octave is 10% as strong as the last (smooth); 1 -> 90% (rugged)


class NoiseBaseParams(RoughnessParams):
    amplitude: float = Field(gt=0.0, le=128.0, description="Max height change in blocks, up or down.")
    feature_size: float = Field(
        default=0.25, gt=0.0, le=1.0, description="Size of the largest bumps as a fraction of the area's shorter side."
    )


def persistence(roughness: float) -> float:
    lo, hi = PERSISTENCE
    return lo + (hi - lo) * roughness


@register_op("noise_base", params=NoiseBaseParams, stage="terrain")
def noise_base(state: WorldState, p: NoiseBaseParams) -> WorldState:
    """Add natural-looking rolling noise (bumps, hollows) over the whole area.

    Use it first, at low amplitude, to break up a flat base; larger amplitudes give hilly country.
    roughness (or character) sets how much finer detail rides on the largest bumps.
    Deterministic: the same op id always gives the same noise.
    """
    n = fractal_noise(state, p.feature_size, detail_octaves(state, p.feature_size), persistence(p.roughness))
    state.height = (state.height + p.amplitude * n).astype(np.float32)
    return state
