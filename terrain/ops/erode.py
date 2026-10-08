"""Erosion op: thermal (slopes crumble to a talus angle) or hydraulic (runoff carves gullies and
deposits sediment aprons at the foot of slopes). Hydraulic is grid-based stream-power erosion with
transport capacity, not particle droplets: ~0.1 s per iteration at 256².

The schema exposes one calibrated `strength` (0-1) per kind; iteration counts and talus angles are
derived here (DESIGN §5.1 principle 3). strength 0.5 reproduces the old defaults (30 iterations,
talus 40° / hydraulic rate 0.5)."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import Field
from scipy.ndimage import gaussian_filter

from terrain.brushes import length_to_blocks
from terrain.noise import fractal_noise
from terrain.ops.region import AreaParams, region_weight
from terrain.ops.registry import Norm, register_op
from terrain.state import WorldState

# 4-neighbour offsets (dz, dx)
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
# 8-neighbour offsets and their distances, for flow routing
_D8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
_D8_DIST = tuple(math.hypot(dz, dx) for dz, dx in _D8)

ERODE_RATE = 0.4  # blocks removed per iteration per unit stream power (sqrt(area) x slope), at rate 1
CAPACITY = 0.8  # sediment a cell's flow can carry, same units; above this it is deposited
ROUTE_NOISE = 4.0  # blocks of fixed small-scale noise added for routing only, so runoff converges into channels
CUT_WIDTH = 1.5  # gaussian sigma (blocks) spreading each cut, so channels are wider than one column
DEPOSIT_WIDTH = 2.0  # gaussian sigma (blocks) spreading deposits into fans instead of single-column piles
SETTLE_DEG = 55.0  # thermal settle inside hydraulic runs: keeps channels, removes spikes
HYDRAULIC_RATE = 0.5  # how fast water cuts (fraction of ERODE_RATE); strength scales the number of rain events
DEPOSITION = 0.5  # share of excess sediment that settles per cell where the flow slows

# strength -> (iterations, talus angle). thermal: 0 -> 5 passes at 65° (only spikes and sheer walls crumble),
# 1 -> 55 passes at 15° (heavy slumping into scree). hydraulic: 0 -> 10 rain events, 1 -> 50.
THERMAL_ITERATIONS = (5, 55)
THERMAL_TALUS_DEG = (65.0, 15.0)
HYDRAULIC_ITERATIONS = (10, 50)


def _lerp(ends: tuple[float, float], t: float) -> float:
    return ends[0] + (ends[1] - ends[0]) * t


def thermal_settings(strength: float) -> tuple[int, float]:
    """(iterations, talus angle in degrees) for thermal erosion of the given strength."""
    return round(_lerp(THERMAL_ITERATIONS, strength)), _lerp(THERMAL_TALUS_DEG, strength)


def hydraulic_iterations(strength: float) -> int:
    return round(_lerp(HYDRAULIC_ITERATIONS, strength))


class ErodeParams(AreaParams):
    kind: Literal["thermal", "hydraulic"] = Field(
        default="thermal",
        description="thermal: steep slopes crumble into scree (ages any shape). hydraulic: rain runoff carves "
        "branching gullies down slopes and deposits sediment at their feet, so mountains merge into the land "
        "around them.",
    )
    strength: Norm = Field(
        default=0.5,
        description="0 = light weathering, 0.5 = clearly aged, 1 = heavily worn (thermal: slopes slump into wide "
        "scree; hydraulic: deep gullies, broad sediment aprons).",
    )


def thermal_step(h: np.ndarray, talus: float) -> np.ndarray:
    """Move material from each column to lower neighbours where the drop exceeds `talus` blocks.

    Mass-conserving: the area edge is closed (nothing flows out).
    """
    pad = np.pad(h, 1, mode="edge")
    rows, cols = h.shape
    excess = []
    for dz, dx in _DIRS:
        neighbour = pad[1 + dz : 1 + dz + rows, 1 + dx : 1 + dx + cols]
        excess.append(np.maximum(h - neighbour - talus, 0.0))
    total = sum(excess)
    largest = np.maximum.reduce(excess)
    # Move half the largest excess (enough to level that pair), split across downhill neighbours.
    frac = np.divide(0.5 * largest, total, out=np.zeros_like(total), where=total > 0)

    acc = np.zeros_like(pad)
    out = np.zeros_like(h)
    for (dz, dx), e in zip(_DIRS, excess):
        moved = frac * e
        out += moved
        acc[1 + dz : 1 + dz + rows, 1 + dx : 1 + dx + cols] += moved
    # Flow into the padding is impossible: padding equals the edge cell, so the edge drop is 0.
    return h - out + acc[1:-1, 1:-1]


def flow_routing(h: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """D8 steepest-descent routing. Returns (receiver flat index, slope to receiver, cells high->low).
    Cells with no lower neighbour (pits, and the closed area edge) are their own receiver."""
    rows, cols = h.shape
    pad = np.pad(h, 1, constant_values=np.inf)  # closed edge: nothing flows out of the area
    best_slope = np.zeros(h.shape)
    best_dir = np.full(h.shape, -1)
    for k, ((dz, dx), dist) in enumerate(zip(_D8, _D8_DIST)):
        slope = (h - pad[1 + dz : 1 + dz + rows, 1 + dx : 1 + dx + cols]) / dist
        better = slope > best_slope
        best_slope = np.where(better, slope, best_slope)
        best_dir = np.where(better, k, best_dir)
    idx = np.arange(rows * cols).reshape(rows, cols)
    offsets = np.array([dz * cols + dx for dz, dx in _D8] + [0])
    receiver = idx + offsets[best_dir]  # best_dir -1 -> offset 0 (self)
    order = np.argsort(-h, axis=None, kind="stable")
    return receiver.ravel(), best_slope.ravel(), order


def hydraulic_step(h: np.ndarray, rate: float, deposition: float, talus: float, route_noise: np.ndarray) -> np.ndarray:
    """One rain event: route runoff downhill (high to low), each cell erodes in proportion to
    sqrt(upstream area) x slope, and deposits sediment beyond what its flow can carry.

    Routing runs on terrain + `route_noise`: on smooth shapes (cones) steepest-descent paths run in
    parallel and never merge, so no channels form; small fixed irregularities make runoff converge and
    the cuts then reinforce themselves. Cuts are spread over a few blocks (CUT_WIDTH) so the settle pass
    doesn't refill them. Material is conserved: pits and the closed area edge keep what reaches them."""
    rows, cols = h.shape
    receiver, _, order = flow_routing(h + route_noise)  # noise only steers where water goes
    flat = h.ravel().copy()

    # Upstream area: each cell receives the runoff of everything that drains through it.
    area = np.ones(flat.size)
    for i in order:
        r = receiver[i]
        if r != i:
            area[r] += area[i]
    # Cutting power uses the real terrain's slope, so flat ground doesn't erode into speckle.
    drop = flat - flat[receiver]
    zi, xi = np.divmod(np.arange(flat.size), cols)
    rz, rx = np.divmod(receiver, cols)
    dist = np.maximum(np.hypot(zi - rz, xi - rx), 1.0)
    slope = np.maximum(drop, 0.0) / dist
    power = np.sqrt(area) * slope
    erode_cap = rate * ERODE_RATE * power
    carry_cap = CAPACITY * power
    # Never cut below the receiver: that would dig a pit and stall the flow.
    erode_cap = np.minimum(erode_cap, 0.5 * np.maximum(drop, 0.0))

    sediment = np.zeros(flat.size)
    order_list, recv_list = order.tolist(), receiver.tolist()
    ecap, ccap = erode_cap.tolist(), carry_cap.tolist()
    cut = [0.0] * flat.size
    dep = [0.0] * flat.size
    sed = sediment.tolist()
    for i in order_list:
        s = sed[i]
        r = recv_list[i]
        if r == i:  # pit or closed edge: everything settles here
            dep[i] += s
            continue
        if s > ccap[i]:
            settle = (s - ccap[i]) * deposition
            dep[i] += settle
            s -= settle
        else:
            c = min(ecap[i], ccap[i] - s)
            cut[i] += c
            s += c
        sed[r] += s
    # Spread both cuts and deposits over a few blocks. Unspread deposits pile a pit's whole sediment load
    # onto one column, which shows up in game as isolated pillars (the first in-game review found ~300).
    out = h - _spread(np.array(cut).reshape(rows, cols), CUT_WIDTH) + _spread(np.array(dep).reshape(rows, cols), DEPOSIT_WIDTH)
    return thermal_step(out, talus)


def _spread(field: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-spread a non-negative field without changing its total."""
    total = field.sum()
    if total <= 0:
        return field
    spread = gaussian_filter(field, sigma, mode="nearest")
    return spread * (total / spread.sum())


@register_op("erode", params=ErodeParams, stage="terrain")
def erode(state: WorldState, p: ErodeParams) -> WorldState:
    """Weather the terrain (or a region). thermal: steep slopes crumble, building scree at their feet;
    removes 1-block spikes. hydraulic: runoff carves branching gullies and spurs into slopes and builds
    sediment aprons at their base (natural foothills, valley floors).

    Use it after placing big forms to make them look aged and natural.
    """
    h0 = state.height.astype(np.float64)
    h = h0
    if p.kind == "thermal":
        iterations, talus_deg = thermal_settings(p.strength)
        talus = math.tan(math.radians(talus_deg))  # blocks of drop per block of distance
        for _ in range(iterations):
            h = thermal_step(h, talus)
    else:
        settle = math.tan(math.radians(SETTLE_DEG))
        route_noise = ROUTE_NOISE * fractal_noise(state, feature_size=0.04, octaves=3, roughness=0.5)
        for _ in range(hydraulic_iterations(p.strength)):
            h = hydraulic_step(h, HYDRAULIC_RATE, DEPOSITION, settle, route_noise)
    if p.region is not None:  # erosion runs on the whole area (runoff comes from outside too), kept inside the region
        h = h0 + region_weight(state, p.region, length_to_blocks(state, p.edge)) * (h - h0)
    state.height = h.astype(np.float32)
    return state
