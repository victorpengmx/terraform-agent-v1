"""Feature attributes v1 (DESIGN §4.2): what code measures about each landform an op created.

The LLM talks in categories ("steep", "winding"); code owns the numeric ranges, measures the canvas and
compares. Categories and `category_distance` are shared by the critic, the spec checklist (M5) and the
eval (M6).

Features come from the op stack: every form op (peak, valley, river, lake, ...) is a feature measured over
its footprint (`masks["feature:<id>"]`). Modifier ops (smooth, erode, warp, roughen, terrace, noise_base)
and materials ops reshape features but are not features themselves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from terrain.materials import surface_metrics
from terrain.opstack import OpEntry, OpStack, op_seed
from terrain.state import WorldState

# (name, upper bound): the first class whose bound the value is below.
SLOPE_CLASSES: tuple[tuple[str, float], ...] = (("gentle", 15.0), ("moderate", 35.0), ("steep", 60.0), ("cliff", math.inf))
# Sinuosity = channel length / straight distance (standard geomorphology classes, after Mueller 1968).
WINDING_CLASSES: tuple[tuple[str, float], ...] = (
    ("almost straight", 1.05), ("winding", 1.25), ("twisty", 1.5), ("meandering", math.inf))
CORE = 0.5  # footprint weight above which a column is the feature's body (the soft edge is excluded)
FOOTPRINT = 0.02  # weight above which a column belongs to the feature at all
RAISED = ("peak", "plateau", "raised_area")


def classify(value: float, classes: tuple[tuple[str, float], ...]) -> str:
    for name, bound in classes:
        if value < bound:
            return name
    return classes[-1][0]


def category_distance(expected: str, actual: str, classes: tuple[tuple[str, float], ...]) -> int:
    """How many categories apart two class names are (0 = same)."""
    names = [n for n, _ in classes]
    for n in (expected, actual):
        if n not in names:
            raise ValueError(f"unknown category {n!r}; expected one of {names}")
    return abs(names.index(expected) - names.index(actual))


def slack(distance: int) -> Literal["ok", "minor", "major"]:
    """One category of slack (DESIGN §4.2): one off is minor, two or more is major."""
    return "ok" if distance == 0 else "minor" if distance == 1 else "major"


@dataclass(frozen=True)
class FeatureAttributes:
    id: str
    op: str
    type: str
    bounds: tuple[float, float, float, float]  # normalized (u0, v0, u1, v1)
    top_y: float  # highest surface in the body (summit for raised features)
    top_uv: tuple[float, float]
    bottom_y: float  # lowest surface in the body (floor for valleys and basins)
    slope_mean: float  # degrees, over the body
    slope_class: str
    edges: tuple[str, ...]  # area edges the footprint touches: n, s, w, e
    prominence: float | None = None  # raised features only
    sinuosity: float | None = None  # rivers only
    winding: str | None = None


def feature_type(entry: OpEntry) -> str | None:
    """The landform an op creates, or None for modifier and materials ops."""
    p = entry.params
    if entry.op == "raise_region":
        mode = p.get("mode", "add")
        if mode == "add":
            return "raised_area" if p["height"] > 0 else "depression"
        return mode  # plateau | basin
    return {"raise_peak": "peak", "carve_valley": "valley", "carve_river": "river", "fill_lake": "lake",
            "set_sea_level": "sea"}.get(entry.op)


def measure_features(stack: OpStack) -> dict[str, FeatureAttributes]:
    """Attributes of every form-op feature in the stack, in replay order."""
    state = stack.state
    slope, _ = surface_metrics(state)
    peaks: dict[int, float] | None = None  # computed once, only if a raised feature needs it
    out: dict[str, FeatureAttributes] = {}
    for entry in stack.entries:
        kind = feature_type(entry)
        feature = state.features.get(entry.id)
        if kind is None or feature is None:
            continue
        mask = state.masks[f"feature:{entry.id}"]
        body = mask > CORE
        if not body.any():
            body = mask > FOOTPRINT
        h = np.where(body, state.height, -np.inf)
        z, x = np.unravel_index(int(h.argmax()), h.shape)
        prominence = sinuosity = None
        if kind in RAISED:
            if peaks is None:
                peaks = peak_prominences(state.height)
            prominence = max((p for cell, p in peaks.items() if body.flat[cell]), default=0.0)
        if kind == "river":
            sinuosity = river_sinuosity(stack, entry)
        mean_slope = float(slope[body].mean())
        out[entry.id] = FeatureAttributes(
            id=entry.id, op=entry.op, type=kind, bounds=feature.bounds,
            top_y=float(state.height[z, x]), top_uv=(x / (state.width - 1), z / (state.depth - 1)),
            bottom_y=float(state.height[body].min()),
            slope_mean=mean_slope, slope_class=classify(mean_slope, SLOPE_CLASSES),
            edges=touched_edges(mask > FOOTPRINT),
            prominence=None if prominence is None else float(prominence),
            sinuosity=sinuosity, winding=None if sinuosity is None else classify(sinuosity, WINDING_CLASSES),
        )
    return out


def touched_edges(footprint: np.ndarray) -> tuple[str, ...]:
    sides = (("n", footprint[0, :]), ("s", footprint[-1, :]), ("w", footprint[:, 0]), ("e", footprint[:, -1]))
    return tuple(name for name, line in sides if line.any())


def river_sinuosity(stack: OpStack, entry: OpEntry) -> float | None:
    """Channel length / straight distance, from the same centre line carve_river cut (same params, seed)."""
    from terrain.ops.water import CarveRiverParams, river_centerline

    base = stack.base
    seeded = WorldState.flat(base.width, base.depth, base_y=base.base_y, seed=op_seed(base.seed, entry.id))
    pts = river_centerline(seeded, CarveRiverParams.model_validate(entry.params))
    straight = float(np.hypot(*(pts[-1] - pts[0])))
    if straight < 1.0:
        return None
    return float(np.hypot(*np.diff(pts, axis=0).T).sum()) / straight


def peak_prominences(height: np.ndarray) -> dict[int, float]:
    """Topographic prominence of every local peak (flat index -> blocks).

    Cells are added from high to low with a union-find over 4-neighbours; when two regions meet at a col,
    the region with the lower peak ends there, and that peak's prominence is its height minus the col's.
    The highest peak's prominence is its height above the lowest point of the area.
    """
    rows, cols = height.shape
    flat = height.ravel().astype(np.float64)
    parent = np.full(flat.size, -1, dtype=np.int64).tolist()
    peak = list(range(flat.size))
    prom: dict[int, float] = {}
    h = flat.tolist()

    def find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    for i in np.argsort(-flat, kind="stable").tolist():
        parent[i] = i
        z, x = divmod(i, cols)
        for nz, nx in ((z - 1, x), (z + 1, x), (z, x - 1), (z, x + 1)):
            if not (0 <= nz < rows and 0 <= nx < cols):
                continue
            n = nz * cols + nx
            if parent[n] == -1:
                continue
            a, b = find(i), find(n)
            if a == b:
                continue
            pa, pb = peak[a], peak[b]
            # Ties keep the region already there (b), so a flat top stays one peak.
            (keep, keep_peak), (lose, lose_peak) = ((a, pa), (b, pb)) if h[pa] > h[pb] else ((b, pb), (a, pa))
            if h[lose_peak] > h[i]:  # not a peak: a cell that only joins a region, or flat low ground
                prom[lose_peak] = h[lose_peak] - h[i]
            parent[lose] = keep
            peak[keep] = keep_peak
    top = int(flat.argmax())
    prom[top] = h[top] - float(flat.min())
    return prom


def feature_report(attrs: dict[str, FeatureAttributes]) -> str:
    """One line per feature, for the critic (not the blind comparison: op ids reveal recency)."""
    if not attrs:
        return "(no landform features yet)"
    lines = []
    for a in attrs.values():
        where = f"u={a.bounds[0]:.2f}..{a.bounds[2]:.2f}, v={a.bounds[1]:.2f}..{a.bounds[3]:.2f}"
        parts = [f"{a.id} ({a.type}, {where})"]
        if a.type in RAISED:
            parts.append(f"summit y={a.top_y:.0f} at (u={a.top_uv[0]:.2f}, v={a.top_uv[1]:.2f}), prominence {a.prominence:.0f}")
        elif a.type in ("valley", "basin", "depression"):
            parts.append(f"floor y={a.bottom_y:.0f}, rim up to y={a.top_y:.0f}")
        if a.type == "river" and a.sinuosity is not None:
            parts.append(f"sinuosity {a.sinuosity:.2f} ({a.winding})")
        if a.type not in ("lake", "sea"):
            parts.append(f"slopes {a.slope_class} (mean {a.slope_mean:.0f}°)")
        parts.append("reaches " + (", ".join({"n": "north", "s": "south", "w": "west", "e": "east"}[e] for e in a.edges) + " edge"
                                   if a.edges else "no edge"))
        lines.append(f"{parts[0]}: {'; '.join(parts[1:])}")
    return "\n".join(lines)
