"""General detail passes (DESIGN §5.5): sculpt the big shape with any op, then add natural detail.

- warp: move terrain sideways along a smooth noise field -> irregular outlines (lobed peaks, scalloped
  mesa walls, ragged coasts). Height noise cannot do this: it doesn't move a wall.
- roughen: add vertical noise (smooth, ridged or billowy), weighted by relief or steepness -> crags,
  rough ridgelines, broken cliffs.
Both take an optional shared `region` (terrain/ops/region.py) to limit where they apply.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field
from scipy.ndimage import map_coordinates

from terrain.brushes import length_to_blocks
from terrain.noise import detail_octaves, fractal_noise
from terrain.ops.region import AreaParams, region_weight
from terrain.ops.registry import register_op
from terrain.state import WorldState


def _full_range(n: np.ndarray) -> np.ndarray:
    """Rescale noise so its largest magnitude is exactly 1: `amount` then means what it says."""
    peak = float(np.abs(n).max())
    return n / peak if peak > 0 else n


class WarpParams(AreaParams):
    amount: float = Field(gt=0.0, le=0.2, description="Largest sideways shift, as a fraction of the shorter side.")
    scale: float = Field(default=0.15, gt=0.0, le=1.0, description="Size of the bends, fraction of the shorter side.")


@register_op("warp", params=WarpParams, stage="terrain")
def warp(state: WorldState, p: WarpParams) -> WorldState:
    """Distort terrain sideways along smooth random bends: makes straight edges, round outlines and
    regular shapes irregular and natural (coastlines, mesa walls, mountain outlines). Heights are only
    moved around, never created: the highest and lowest points stay the same."""
    # The selection must fade out over at least the shift, or outlines could only move inward.
    w = region_weight(state, p.region, length_to_blocks(state, max(p.edge, 1.5 * p.amount)))
    shift = length_to_blocks(state, p.amount)
    dx = shift * w * _full_range(fractal_noise(state, p.scale, octaves=3, roughness=0.5, seed=state.seed))
    dz = shift * w * _full_range(fractal_noise(state, p.scale, octaves=3, roughness=0.5, seed=state.seed + 1))
    z, x = np.mgrid[0 : state.depth, 0 : state.width].astype(np.float64)
    warped = map_coordinates(state.height.astype(np.float64), [z + dz, x + dx], order=1, mode="nearest")
    state.height = warped.astype(np.float32)
    return state


class RoughenParams(AreaParams):
    amount: float = Field(gt=0.0, le=64.0, description="Largest height change in blocks, up or down.")
    scale: float = Field(default=0.06, gt=0.0, le=1.0, description="Size of the largest bumps, fraction of the shorter side.")
    style: Literal["fbm", "ridged", "billow"] = Field(
        default="ridged", description="fbm: rolling bumps. ridged: sharp crests and crags. billow: rounded lumps.")
    weight: Literal["uniform", "relief", "slope"] = Field(
        default="relief", description="uniform: same everywhere. relief: stronger higher up (peaks, ridgelines). "
        "slope: stronger on steep ground (cliffs, flanks).")


@register_op("roughen", params=RoughenParams, stage="terrain")
def roughen(state: WorldState, p: RoughenParams) -> WorldState:
    """Add natural surface detail: crags and jagged ridgelines (ridged), rolling bumps (fbm) or lumps
    (billow). Use after sculpting a smooth shape; weight by relief to roughen peaks more than valleys."""
    noise = detail_noise(state, p.scale, p.style)
    mask = region_weight(state, p.region, length_to_blocks(state, p.edge))
    h = state.height
    state.height = (h + p.amount * noise * detail_strength(h, mask, p.weight) * mask).astype(np.float32)
    return state


def detail_noise(state: WorldState, scale: float, style: str) -> np.ndarray:
    """Multi-octave noise in [-1, 1] (largest magnitude exactly 1), from `scale` down to 4-block features
    (finer ones only make pillars). ridged: sharp crests, round troughs; billow: round lumps; fbm: bumps.
    Shared by roughen and raise_ridge (whose summits and cols are this noise, ridged)."""
    total = np.zeros(state.height.shape)
    norm, w_k, size = 0.0, 1.0, scale
    for k in range(detail_octaves(state, scale)):
        n = fractal_noise(state, size, octaves=1, roughness=0.5, seed=state.seed + k)
        if style == "ridged":
            layer = 2.0 * (1.0 - np.abs(n)) ** 2 - 1.0
        elif style == "billow":
            layer = 2.0 * np.abs(n) - 1.0
        else:
            layer = n
        total += w_k * layer
        norm += w_k
        w_k, size = w_k * 0.5, size / 2
    return _full_range(total / norm)


def detail_strength(h: np.ndarray, mask: np.ndarray, weight: str) -> np.ndarray | float:
    """Where detail applies fully: relief = higher ground (within the masked area), slope = steeper ground."""
    if weight == "relief":
        sel = mask > 0.5
        lo = float(np.percentile(h[sel], 5)) if sel.any() else float(h.min())
        hi = float(h[sel].max()) if sel.any() else float(h.max())
        return np.clip((h - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    if weight == "slope":
        gz, gx = np.gradient(h.astype(np.float64))
        return np.clip(np.degrees(np.arctan(np.hypot(gx, gz))) / 60.0, 0.0, 1.0)
    return 1.0
