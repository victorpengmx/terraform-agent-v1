"""Seeded noise fields shared by ops and the material engine (kept outside terrain.ops to avoid cycles)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from terrain.brushes import length_to_blocks
from terrain.state import WorldState

MIN_DETAIL_BLOCKS = 4.0  # finest noise feature a detail octave may add: smaller ones only make 1-column pillars
MAX_OCTAVES = 8


def detail_octaves(state: WorldState, feature_size: float) -> int:
    """Octave count for noise starting at `feature_size` (fraction of the shorter side), halving each
    octave down to the finest that is still >= MIN_DETAIL_BLOCKS. Always at least 1. Code-side, not a
    schema knob (DESIGN §5.1 principle 3)."""
    size, n = length_to_blocks(state, feature_size), 1
    while n < MAX_OCTAVES and size / 2**n >= MIN_DETAIL_BLOCKS:
        n += 1
    return n


def fractal_noise(state: WorldState, feature_size: float, octaves: int, roughness: float, seed: int | None = None) -> np.ndarray:
    """Seeded fractal value noise in [-1, 1], shaped like the layers. `seed` defaults to the op's seed;
    pass derived seeds (state.seed + k) when one op needs several independent fields."""
    rng = np.random.default_rng(state.seed if seed is None else seed)
    z, x = np.mgrid[0 : state.depth, 0 : state.width].astype(np.float64)
    total = np.zeros(state.height.shape)
    weight_sum = 0.0
    cell, weight = max(length_to_blocks(state, feature_size), 2.0), 1.0
    for _ in range(octaves):
        grid = rng.uniform(-1.0, 1.0, (int(state.depth / cell) + 4, int(state.width / cell) + 4))
        total += weight * map_coordinates(grid, [z / cell + 1, x / cell + 1], order=3, mode="reflect")
        weight_sum += weight
        cell, weight = cell / 2, weight * roughness
        if cell < 2.0:  # finer than 2 blocks is just per-block jitter
            break
    return np.clip(total / weight_sum, -1.0, 1.0)
