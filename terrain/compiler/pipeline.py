"""State + backend -> BlockPatch: compile over a y range that covers both the new terrain and the
world's current surface, then diff against what is there."""

from __future__ import annotations

import numpy as np

from terrain.backends.base import WorldBackend
from terrain.compiler.compile import compile_state, default_y_range
from terrain.compiler.volume import BlockPatch, Box, Volume, diff
from terrain.config import WORLD_MAX_Y, WORLD_MIN_Y
from terrain.state import WorldState


def build_patch(state: WorldState, backend: WorldBackend, origin: tuple[int, int]) -> tuple[BlockPatch, Volume]:
    """Returns the patch (changed blocks only) and the compiled volume (for L2 rendering)."""
    x0, z0 = origin
    x1, z1 = x0 + state.width, z0 + state.depth
    ground, water = backend.read_surface(x0, z0, x1, z1)
    y0, y1 = default_y_range(state)
    world_top = float(np.nanmax(np.fmax(ground, water)))
    y0 = max(min(y0, int(ground.min()) - 4), WORLD_MIN_Y + 1)
    y1 = min(max(y1, int(world_top) + 2), WORLD_MAX_Y)
    new = compile_state(state, origin, (y0, y1))
    old = backend.read_region(Box(x0, z0, x1, z1, y0, y1))
    return diff(new, old), new
