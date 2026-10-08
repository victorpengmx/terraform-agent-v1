"""Layers → block columns → BlockPatch (DESIGN §7)."""

from terrain.compiler.compile import compile_state, default_y_range, surface_y, water_top
from terrain.compiler.volume import AIR, WATER, BlockPatch, Box, Volume, diff

__all__ = ["AIR", "WATER", "BlockPatch", "Box", "Volume", "compile_state", "default_y_range", "diff", "surface_y", "water_top"]
