"""Shared `region` type for every area-scoped op (DESIGN §5.2).

    {"circle":  {"center": [u, v], "radius": r}}
    {"polygon": [[u, v], ...]}
    {"path":    {"points": [[u, v], ...], "width": w}}
    {"feature": "mountain_1"}                          # an op's footprint

One schema, one implementation, one set of error messages. Lengths are fractions of the shorter side.
Ops call `region_shape` (hard selection + signed distance) or `region_weight` (soft selection).
"""

from __future__ import annotations

from typing import Annotated, Any

import numpy as np
from pydantic import Discriminator, Field, Tag
from scipy.ndimage import distance_transform_edt

from terrain.brushes import circle_distance, edge_weight, feature_mask, length_to_blocks, path_distance, polygon_distance, polygon_inside, to_blocks
from terrain.ops.registry import Norm, OpError, OpParams, Point
from terrain.state import WorldState

KINDS = ("circle", "polygon", "path", "feature")
PATH_WOBBLE = 0.2  # path width varies by up to ±20% along its length, so painted trails have natural edges


class Circle(OpParams):
    center: Point
    radius: Norm = Field(gt=0.0, description="Fraction of the shorter side.")


class CircleRegion(OpParams):
    circle: Circle


class PolygonRegion(OpParams):
    polygon: list[Point] = Field(min_length=3, max_length=32, description="Outline as (u, v) points, in order.")


class PathSpec(OpParams):
    points: list[Point] = Field(min_length=2, max_length=32)
    width: Norm = Field(gt=0.0, description="Full width, fraction of the shorter side.")


class PathRegion(OpParams):
    path: PathSpec


class FeatureRegion(OpParams):
    feature: str = Field(min_length=1, description="An op id; selects that op's footprint.")


def _kind(v: Any) -> str | None:
    """Union tag = the region's single key; None (-> one clear error) for zero or several keys."""
    if isinstance(v, OpParams):
        return next(iter(type(v).model_fields))
    if isinstance(v, dict) and len(v) == 1 and next(iter(v)) in KINDS:
        return next(iter(v))
    return None


Region = Annotated[
    Annotated[CircleRegion, Tag("circle")] | Annotated[PolygonRegion, Tag("polygon")]
    | Annotated[PathRegion, Tag("path")] | Annotated[FeatureRegion, Tag("feature")],
    Discriminator(_kind, custom_error_type="invalid_region",
                  custom_error_message="region needs exactly one of the keys circle, polygon, path, feature"),
]

REGION_HELP = ('{"circle": {"center": [u, v], "radius": r}}, {"polygon": [[u, v], ...]}, '
               '{"path": {"points": [[u, v], ...], "width": w}} or {"feature": "<op id>"} (that op\'s footprint)')
EDGE_HELP = "Fade-out width beyond the region, fraction of the shorter side."


class AreaParams(OpParams):
    """Base for modifiers that apply everywhere unless limited to a region."""

    region: Region | None = Field(default=None, description=f"Where to apply: {REGION_HELP}. Omit for the whole area.")
    edge: Norm = Field(default=0.05, description=EDGE_HELP)


def region_shape(state: WorldState, region: Region | None) -> tuple[np.ndarray, np.ndarray]:
    """(inside, signed distance in blocks: negative inside, positive outside). None = the whole area.

    Raises OpError for an unknown feature or a region that covers no column of the area.
    """
    if region is None:
        z, x = np.mgrid[0 : state.depth, 0 : state.width]
        border = np.minimum.reduce([x, z, state.width - 1 - x, state.depth - 1 - z]).astype(np.float32)
        return np.ones(state.height.shape, dtype=bool), -border
    if isinstance(region, PolygonRegion):
        inside, signed = polygon_inside(state, region.polygon), polygon_distance(state, region.polygon)
    elif isinstance(region, CircleRegion):
        signed = circle_distance(state, region.circle.center) - length_to_blocks(state, region.circle.radius)
        inside = signed <= 0
    elif isinstance(region, PathRegion):
        signed = _path_signed_distance(state, region.path)
        inside = signed <= 0
    else:
        try:
            inside = feature_mask(state, region.feature) > 0.02
        except KeyError as e:
            raise OpError(str(e).strip("'\"")) from None
        if not inside.any():
            raise OpError(f"feature {region.feature!r} has no footprint (it changed no height or water); pick another region")
        signed = distance_transform_edt(~inside) - (distance_transform_edt(inside) if not inside.all() else 0.0)
    if not inside.any():
        raise OpError("region covers no column of the area: keep coordinates within 0..1 and make it at least a block wide")
    return inside, np.asarray(signed)  # keep its dtype: feature fades were float64 before the migration


def _smooth_points(state: WorldState, points: list) -> list[tuple[float, float]]:
    """Catmull-Rom curve through the points (they stay on the path), sampled about every 2 blocks."""
    pts = np.array([to_blocks(state, p) for p in points], dtype=np.float64)
    ext = np.vstack([pts[0], pts, pts[-1]])  # repeat the ends so the curve starts and ends on them
    out = []
    for i in range(len(pts) - 1):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        n = max(2, int(np.ceil(np.hypot(*(p2 - p1)) / 2.0)))
        t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
        out.append(0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2 + (3 * p1 - p0 - 3 * p2 + p3) * t**3))
    curve = np.vstack([*out, pts[-1:]])
    scale = np.array([max(state.width - 1, 1), max(state.depth - 1, 1)], dtype=np.float64)
    return [tuple(q) for q in np.clip(curve / scale, 0.0, 1.0)]


def _path_signed_distance(state: WorldState, path: PathSpec) -> np.ndarray:
    """Signed distance to a smooth path whose half-width wobbles gently along its length."""
    points = _smooth_points(state, path.points)
    dist, t = path_distance(state, points)
    curve = np.array([to_blocks(state, p) for p in points])
    length = float(np.hypot(*np.diff(curve, axis=0).T).sum())
    half = length_to_blocks(state, path.width) / 2
    wavelength = max(6.0 * half, 12.0)  # blocks; long, gentle swells rather than jitter
    phase = np.random.default_rng(state.seed).uniform(0.0, 2 * np.pi, 2)
    s = t.astype(np.float64) * length
    wobble = (np.sin(2 * np.pi * s / wavelength + phase[0]) + 0.5 * np.sin(2 * np.pi * s / (0.37 * wavelength) + phase[1])) / 1.5
    return (dist - half * (1.0 + PATH_WOBBLE * wobble)).astype(np.float32)


def region_weight(state: WorldState, region: Region | None, edge_blocks: float) -> np.ndarray:
    """1 inside the region, fading to 0 over `edge_blocks` outside it."""
    _, signed = region_shape(state, region)
    return edge_weight(signed, edge_blocks).astype(np.float32)
