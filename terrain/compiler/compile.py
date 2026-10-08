"""Compiler (DESIGN §7): canvas layers -> block columns. Deterministic, no LLM.

Per column with surface y = round(height):
  - air above the surface (and above water)
  - water from surface+1 up to round(water level)
  - the surface block, extended down the exposed face (how far the column stands above its lowest
    4-neighbour), so cliffs show the palette's (banded) rock rather than soil
  - subsurface layers (e.g. dirt under grass), then stone
Heights are rounded only here (CLAUDE.md conventions).
"""

from __future__ import annotations

import numpy as np

from terrain.compiler.volume import AIR, WATER, Box, Volume
from terrain.config import WORLD_MAX_Y, WORLD_MIN_Y
from terrain.materials import SurfacePlan, resolve_surface
from terrain.state import WorldState

STONE = "stone"


def surface_y(state: WorldState) -> np.ndarray:
    return np.clip(np.round(state.height), WORLD_MIN_Y + 1, WORLD_MAX_Y - 1).astype(np.int32)


def water_top(state: WorldState) -> np.ndarray:
    """Top water block y per column (WORLD_MIN_Y - 1 where dry), contained in block space: rounding can
    turn a barely-wet column into dry ground one block lower, so water is re-checked against rounded
    banks and lowered (never raised) until no water block stands beside lower dry ground."""
    top = np.where(np.isnan(state.water), WORLD_MIN_Y - 1, np.round(np.nan_to_num(state.water, nan=0.0)))
    top = np.minimum(top, WORLD_MAX_Y - 1).astype(np.int32)
    h = surface_y(state)
    dry_level = WORLD_MIN_Y - 1
    top[top <= h] = dry_level
    big = np.iinfo(np.int32).max
    while True:
        wet = top > h
        hp = np.pad(np.where(wet, big, h), 1, constant_values=big)  # area border acts as a wall
        bank = np.minimum.reduce([hp[:-2, 1:-1], hp[2:, 1:-1], hp[1:-1, :-2], hp[1:-1, 2:]])
        lowered = wet & (bank < top)
        if not lowered.any():
            return top
        top = np.where(lowered, bank, top)
        top[top <= h] = dry_level


def exposed_depth(h: np.ndarray) -> np.ndarray:
    """Blocks of the column visible from the side: height above the lowest 4-neighbour (at least 1)."""
    p = np.pad(h, 1, mode="edge")
    lowest = np.minimum.reduce([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
    return np.maximum(h - lowest, 1)


def default_y_range(state: WorldState, plan: SurfacePlan | None = None) -> tuple[int, int]:
    h = surface_y(state)
    depth = exposed_depth(h) + (plan.subsurface_depth if plan is not None else 3)
    y0 = max(int((h - depth).min()) - 1, WORLD_MIN_Y + 1)  # never touch the bedrock floor
    y1 = min(int(max(h.max(), water_top(state).max())) + 2, WORLD_MAX_Y)
    return y0, y1


def compile_state(state: WorldState, origin: tuple[int, int] = (0, 0), y_range: tuple[int, int] | None = None) -> Volume:
    """Blocks for the whole area. `origin` is the world (x, z) of the area's north-west column;
    `y_range` (min inclusive, max exclusive) must cover the existing world surface when diffing against it."""
    plan = resolve_surface(state)
    y0, y1 = y_range or default_y_range(state, plan)
    y0 = max(y0, WORLD_MIN_Y + 1)
    y1 = min(y1, WORLD_MAX_Y)
    if y1 <= y0:
        raise ValueError(f"empty y range {y0}..{y1}")

    palette = [AIR, WATER, STONE]
    lut: dict[str, int] = {n: i for i, n in enumerate(palette)}

    def ids(names: np.ndarray) -> np.ndarray:
        flat = names.ravel()
        uniq, inverse = np.unique(flat.astype(str), return_inverse=True)
        for n in uniq:
            if n not in lut:
                lut[n] = len(palette)
                palette.append(n)
        return np.array([lut[n] for n in uniq], dtype=np.uint16)[inverse].reshape(names.shape)

    col_ids = ids(np.where(plan.column_block == "", STONE, plan.column_block))
    sub_ids = ids(plan.subsurface)
    band_ids = {r: (ids(table), y_start) for r, (table, y_start) in plan.band_tables.items()}

    h = surface_y(state)
    wt = water_top(state)
    face_bottom = h - exposed_depth(h) + 1  # lowest y of the exposed surface material
    sub_bottom = face_bottom - plan.subsurface_depth

    depth, width = h.shape
    data = np.empty((depth, y1 - y0, width), dtype=np.uint16)
    for j, y in enumerate(range(y0, y1)):
        layer = np.full(h.shape, lut[AIR], dtype=np.uint16)
        layer[(y > h) & (y <= wt)] = lut[WATER]
        solid = y <= h
        layer[solid] = lut[STONE]
        sub = solid & (y >= sub_bottom) & (y < face_bottom)
        layer[sub] = sub_ids[sub]
        face = solid & (y >= face_bottom)
        if face.any():
            surf = col_ids.copy()
            for r, (table, y_start) in band_ids.items():
                cols = face & (plan.rule_index == r)
                if cols.any():
                    k = np.clip(np.round(y + plan.band_offset[cols]).astype(int) - y_start, 0, len(table) - 1)
                    surf[cols] = table[k]
            layer[face] = surf[face]
        data[:, j, :] = layer

    box = Box(origin[0], origin[1], origin[0] + width, origin[1] + depth, y0, y1)
    return Volume(box, palette, data)
