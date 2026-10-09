"""Water ops (stage "water"): rivers and lakes. Physical rules are enforced here, not left to the agent:
river beds never run uphill, water never sits above its banks, lakes fill only to their spill height.
"""

from __future__ import annotations

import heapq

import numpy as np
from pydantic import Field
from scipy.ndimage import label, map_coordinates
from scipy.spatial import cKDTree

from terrain.brushes import length_to_blocks, smooth_path, to_blocks
from terrain.ops.registry import Norm, OpError, OpParams, Point, register_op
from terrain.state import WorldState

EDGE_TOLERANCE = 0.02  # a river end this close (normalized) to the border counts as reaching the edge
SAMPLE_STEP = 0.5  # blocks between samples along a river path
FOUR = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])


# ---------------------------------------------------------------- helpers


def densify(points: np.ndarray, step: float = SAMPLE_STEP) -> np.ndarray:
    """Resample a polyline (block coords, shape (n, 2) as x, z) at roughly uniform spacing."""
    out = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        n = max(int(np.ceil(np.hypot(*(b - a)) / step)), 1)
        out.extend(a + (b - a) * (i / n) for i in range(1, n + 1))
    return np.array(out)


def meander(points: np.ndarray, amplitude: float, seed: int) -> np.ndarray:
    """Displace a dense polyline sideways with smooth seeded waves; the endpoints stay fixed."""
    seg = np.hypot(*np.diff(points, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)]) / max(seg.sum(), 1e-9)
    tangent = np.gradient(points, axis=0)
    tangent /= np.maximum(np.hypot(*tangent.T), 1e-9)[:, None]
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    rng = np.random.default_rng(seed)
    waves = sum(np.sin(2 * np.pi * f * s + rng.uniform(0, 2 * np.pi)) / k for k, f in enumerate((1.3, 2.9, 5.1), start=1))
    offset = amplitude * np.sin(np.pi * s) * waves / np.abs(waves).max(initial=1e-9)
    return points + normal * offset[:, None]


def snap_to_edge(state: WorldState, u: float, v: float) -> tuple[float, float]:
    """Move a point onto the nearest area border."""
    d = {"w": u, "e": 1 - u, "n": v, "s": 1 - v}
    side = min(d, key=d.get)
    return {"w": (0.0, v), "e": (1.0, v), "n": (u, 0.0), "s": (u, 1.0)}[side]


def spill_levels(height: np.ndarray) -> np.ndarray:
    """Priority-flood from the area border: for each column, the lowest water level that would drain out.

    A column whose spill level is above its ground lies in a closed basin.
    """
    rows, cols = height.shape
    level = np.full(height.shape, np.inf)
    heap: list[tuple[float, int, int]] = []
    for z in range(rows):
        for x in (0, cols - 1):
            heap.append((float(height[z, x]), z, x))
    for x in range(1, cols - 1):
        for z in (0, rows - 1):
            heap.append((float(height[z, x]), z, x))
    for h, z, x in heap:
        level[z, x] = h
    heapq.heapify(heap)
    while heap:
        h, z, x = heapq.heappop(heap)
        if h > level[z, x]:
            continue
        for dz, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nz, nx = z + dz, x + dx
            if 0 <= nz < rows and 0 <= nx < cols:
                nh = max(h, float(height[nz, nx]))
                if nh < level[nz, nx]:
                    level[nz, nx] = nh
                    heapq.heappush(heap, (nh, nz, nx))
    return level


MIN_WATER = 0.05  # blocks; shallower than this counts as dry


# ---------------------------------------------------------------- carve_river


class CarveRiverParams(OpParams):
    path: list[Point] = Field(
        min_length=2, max_length=32,
        description="From source to mouth as normalized (u, v) points. Must end at the area edge or in existing water.",
    )
    width: float = Field(gt=0.0, le=0.25, description="Water width as a fraction of the shorter side.")
    depth: float = Field(default=4.0, ge=2.0, le=24.0, description="Channel depth in blocks below the land it cuts through.")
    meander: Norm = Field(default=0.0, description="0 = follows the path exactly, 1 = strongly winding.")


@register_op("carve_river", params=CarveRiverParams, stage="water")
def carve_river(state: WorldState, p: CarveRiverParams) -> WorldState:
    """Carve a river channel along a path and fill it with water, flowing from the first point to the last.

    The bed always runs downhill (it cuts through rises instead of climbing them), and water never stands
    above its banks. The river must end at the area edge or in existing water (e.g. a lake from fill_lake).
    """
    end_u, end_v = p.path[-1]
    ex, ez = (int(round(c)) for c in to_blocks(state, (end_u, end_v)))
    if not _ends_on_edge(end_u, end_v) and np.isnan(state.water[ez, ex]):
        raise OpError(
            f"the river's last point (u={end_u:.2f}, v={end_v:.2f}) is neither at the area edge nor in water. "
            "Extend the path to an edge (u or v = 0 or 1) or create a lake there first with fill_lake."
        )
    pts = river_centerline(state, p)
    half = max(length_to_blocks(state, p.width) / 2, 1.0)

    # Bed profile along the path: never higher than any earlier point.
    ground = map_coordinates(state.height, [pts[:, 1], pts[:, 0]], order=1, mode="nearest")
    bed = np.minimum.accumulate(ground - p.depth)

    # Per column: distance to the path and the bed height of the nearest path sample.
    zz, xx = np.mgrid[0 : state.depth, 0 : state.width]
    dist, idx = cKDTree(pts).query(np.stack([xx.ravel(), zz.ravel()], axis=1))
    dist = dist.reshape(state.height.shape).astype(np.float32)
    bed_col = bed[idx].reshape(state.height.shape).astype(np.float32)

    # Whole blocks, so the water survives the compiler's rounding: the bed floored to a block, the surface
    # max(1, depth - 1) blocks above it, the bank rim one block above the surface (~depth above the bed).
    # Water less than a block deep would round away (and drain its neighbours at compile time), so every
    # column under the surface is cut a full block below it: a flat floor, banks never below the water.
    bed_block = np.floor(bed_col)
    surface = bed_block + max(1.0, round(p.depth) - 1.0)
    # Parabolic cross-section with a floor half a sample step wide: columns between samples would otherwise
    # sit up to ~0.01 above the bed, and the bed must never rise along the river.
    floor_dist = np.maximum(dist - SAMPLE_STEP / 2, 0.0)
    channel = bed_block + (surface + 1.0 - bed_block) * (floor_dist / half) ** 2
    channel = np.where(channel < surface, np.minimum(channel, surface - 1.0), channel)
    state.height = np.minimum(state.height, channel).astype(np.float32)

    existing = ~np.isnan(state.water)
    candidates = (dist <= half) & (state.height < surface - MIN_WATER) & ~existing
    wet, level, state.height = contain_by_digging(state.height, candidates, surface, outlets=existing)
    state.water = np.where(wet, level, state.water).astype(np.float32)
    return state


def _ends_on_edge(u: float, v: float) -> bool:
    return min(u, 1 - u, v, 1 - v) <= EDGE_TOLERANCE


def river_centerline(state: WorldState, p: CarveRiverParams) -> np.ndarray:
    """The river's channel line in block coordinates (n, 2) as (x, z), source to mouth: end snapped to the
    edge, spline-smoothed, densified, meandered. Depends only on the area size, params and seed, so feature
    measurement can recompute it for an op in the stack."""
    path = [tuple(pt) for pt in p.path]
    if _ends_on_edge(*path[-1]):
        path[-1] = snap_to_edge(state, *path[-1])
    pts = densify(np.array([to_blocks(state, pt) for pt in smooth_path(state, path)]))
    if p.meander > 0:
        half = max(length_to_blocks(state, p.width) / 2, 1.0)
        path_len = len(pts) * SAMPLE_STEP
        pts = meander(pts, amplitude=p.meander * (3 * half + 0.05 * path_len), seed=state.seed)
        pts[:, 0] = np.clip(pts[:, 0], 0, state.width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, state.depth - 1)
    return pts


def contain_by_digging(height: np.ndarray, wet: np.ndarray, level: np.ndarray, outlets: np.ndarray):
    """River containment in block space: (wet, whole-block levels, height).

    Levels drop to the lowest dry 4-neighbour as the compiler sees it (rounded; area border and water are
    not banks);
    but a column left with under one block of water is dug one block below its level instead of drained.
    A drained column would become a low bank and drain its neighbours in turn, all along a river that
    crosses a slope. Only wet columns change, so the dry banks are fixed and one pass is enough.
    """
    ground = np.round(height)
    big = np.inf
    gp = np.pad(np.where(wet | outlets, big, ground), 1, constant_values=big)  # border and water are not banks
    bank = np.minimum.reduce([gp[:-2, 1:-1], gp[2:, 1:-1], gp[1:-1, :-2], gp[1:-1, 2:]])
    level = np.where(wet, np.minimum(level, bank), np.nan)
    shallow = wet & (level < ground + 1.0)
    height = np.where(shallow, np.minimum(height, level - 1.0), height).astype(np.float32)
    return wet, level, height


# ---------------------------------------------------------------- fill_lake


class FillLakeParams(OpParams):
    at: Point = Field(description="A point inside the basin to fill, normalized (u, v).")
    max_depth: float = Field(
        default=64.0, gt=0.0, le=128.0,
        description="Cap on water depth at `at`. The lake fills to the basin's spill height or this depth, whichever is lower.",
    )


@register_op("fill_lake", params=FillLakeParams, stage="water")
def fill_lake(state: WorldState, p: FillLakeParams) -> WorldState:
    """Fill the basin around a point with still water, up to the level where it would spill over its rim.

    The point must be in a closed depression (dig one with raise_region using a negative height, or
    carve_valley). The lake covers every connected column below the water level.
    """
    x, z = (int(round(c)) for c in to_blocks(state, p.at))
    ground = float(state.height[z, x])
    spill = float(spill_levels(state.height)[z, x])
    level = min(spill, ground + p.max_depth)
    if level - ground < 0.5:
        raise OpError(
            f"no basin at (u={p.at[0]:.2f}, v={p.at[1]:.2f}): water there drains away (ground y={ground:.0f}, "
            f"spill level y={spill:.0f}). Dig a basin first (raise_region with negative height, or carve_valley) "
            "or pick a lower point."
        )
    labels, _ = label(state.height < level, structure=FOUR)
    lake = labels == labels[z, x]
    state.water = np.where(lake, np.fmax(state.water, level), state.water).astype(np.float32)
    return state


# ---------------------------------------------------------------- set_sea_level


class SetSeaLevelParams(OpParams):
    level: float = Field(ge=-60.0, le=300.0, description="Sea surface y. Keep it below the ground of the world around the area.")


@register_op("set_sea_level", params=SetSeaLevelParams, stage="water")
def set_sea_level(state: WorldState, p: SetSeaLevelParams) -> WorldState:
    """Flood everything below the given level that connects to the area edge: oceans, bays, coasts.

    Enclosed basins below sea level stay dry (fill them with fill_lake). Lower the terrain near the edge
    first (e.g. raise_region with negative height over the whole area) to make an island.
    """
    below = state.height < p.level
    labels, _ = label(below, structure=FOUR)
    border = np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
    sea = np.isin(labels, border[border > 0])
    if not sea.any():
        raise OpError(
            f"no terrain below y={p.level:.0f} touches the area edge, so there is nothing for the sea to flood "
            f"(edge ground is y={min(state.height[0].min(), state.height[-1].min(), state.height[:, 0].min(), state.height[:, -1].min()):.0f}). "
            "Lower the outer terrain first or raise the level."
        )
    state.water = np.where(sea, np.fmax(state.water, p.level), state.water).astype(np.float32)
    return state
