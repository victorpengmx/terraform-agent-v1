"""L2 voxel preview (DESIGN §10.2): the compiled block grid rendered as coloured cubes, without Minecraft.

1. Visible-face extraction: a face only where a block touches air (or, for solids, water).
2. Colour by block from block_colors.json (top vs side texture average).
3. Baked shading for a crisp Minecraft-like look: fixed per-direction brightness (sun from the
   north-west) plus a cheap ambient-occlusion term on top faces. Water is a translucent second mesh.
4. Offscreen PyVista render at fixed camera presets, 1024 px, with a north arrow and a scale bar.

World axes map to VTK as X = x (east), Y = -z (north), Z = y (up).
Needs the `render` extra (pyvista).
"""

from __future__ import annotations

import io
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from terrain.compiler.volume import AIR, WATER, Volume
from terrain.state import Feature

COLORS_FILE = Path(__file__).with_name("block_colors.json")
FALLBACK_RGB = (200, 0, 200)  # unmistakable: a block missing from the colour table
VIEWS = ("overview_ne", "overview_sw", "overview_se", "overview_nw", "overview_s")
# Horizontal camera direction from the area's centre, in (east, north) units of 0.85 × the area's span.
# overview_s stands due south at the corner views' distance: the viewpoint the spec assumes for photos.
_VIEW_DIRS = {"ne": (1.0, 1.0), "se": (1.0, -1.0), "sw": (-1.0, -1.0), "nw": (-1.0, 1.0), "s": (0.0, -math.sqrt(2.0))}
# Seen square-on, the whole width faces the camera, so overview_s backs off (same elevation angle) to keep the
# near corners, compass and scale bar in frame.
_VIEW_PULL = {"s": 1.3}

# (axis offset dz, dy, dx), face name, brightness. Bottom faces are never visible from our cameras.
_DIRS = (
    ((0, 1, 0), "top", 1.00),
    ((-1, 0, 0), "side", 0.86),  # north (-z)
    ((0, 0, -1), "side", 0.80),  # west
    ((1, 0, 0), "side", 0.62),  # south
    ((0, 0, 1), "side", 0.70),  # east
)
# Unit-square corners for each face, in (x, y, z) block-local coordinates.
_CORNERS = {
    (0, 1, 0): [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)],
    (-1, 0, 0): [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
    (1, 0, 0): [(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)],
    (0, 0, -1): [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)],
    (0, 0, 1): [(1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)],
}


@lru_cache(maxsize=1)
def block_colors() -> dict[str, dict[str, tuple[int, int, int]]]:
    data = json.loads(COLORS_FILE.read_text(encoding="utf-8"))["colors"]
    return {name: {face: tuple(rgb) for face, rgb in faces.items()} for name, faces in data.items()}


def _shifted(a: np.ndarray, d: tuple[int, int, int], fill) -> np.ndarray:
    """Neighbour values: out[i] = a[i + d], `fill` outside the array."""
    out = np.full_like(a, fill)
    src = tuple(slice(max(k, 0), a.shape[i] + min(k, 0)) for i, k in enumerate(d))
    dst = tuple(slice(max(-k, 0), a.shape[i] + min(-k, 0)) for i, k in enumerate(d))
    out[dst] = a[src]
    return out


def _faces(volume: Volume, kind: np.ndarray, which: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quads for blocks of category `which` (1 solid, 2 water). kind: 0 air, 1 solid, 2 water.
    Returns (points (n*4, 3), faces (n*5,), colours (n, 3) uint8)."""
    colors = block_colors()
    pal_rgb = {
        face: np.array([colors.get(n, {}).get(face, FALLBACK_RGB) for n in volume.palette], dtype=np.float64)
        for face in ("top", "side")
    }
    solid_above = _shifted(kind == 1, (0, 1, 0), False)
    pts, quads, rgbs = [], [], []
    b = volume.box
    for d, face, light in _DIRS:
        nbr = _shifted(kind, d, 0)  # outside the box counts as air: the area edge shows as a wall
        visible = (kind == which) & ((nbr == 0) if which == 2 else (nbr != 1))
        if which == 2 and face != "top":
            continue  # water: surface only
        z, y, x = np.nonzero(visible)
        if len(z) == 0:
            continue
        rgb = pal_rgb[face][volume.data[z, y, x]] * light
        if face == "top" and which == 1:
            # Ambient occlusion: darken tops beside taller neighbours.
            occ = sum(_shifted(solid_above, (dz, 0, dx), False)[z, y, x] for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            rgb *= 1.0 - 0.07 * occ[:, None]
        corners = np.array(_CORNERS[d], dtype=np.float64)  # (4, 3) as x, y, z
        base = np.stack([x + b.x0, y + b.y0, z + b.z0], axis=1).astype(np.float64)
        world = base[:, None, :] + corners[None, :, :]  # (n, 4, 3)
        vtk_pts = np.stack([world[..., 0], -world[..., 2], world[..., 1]], axis=-1).reshape(-1, 3)
        start = sum(len(p) for p in pts)
        n = len(z)
        idx = start + np.arange(n * 4).reshape(n, 4)
        quads.append(np.hstack([np.full((n, 1), 4), idx]).ravel())
        pts.append(vtk_pts)
        rgbs.append(np.clip(rgb, 0, 255).astype(np.uint8))
    if not pts:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64), np.zeros((0, 3), dtype=np.uint8)
    return np.vstack(pts), np.concatenate(quads).astype(np.int64), np.vstack(rgbs)


def _camera(volume: Volume, view: str, feature: Feature | None) -> tuple[tuple, tuple]:
    b = volume.box
    if feature is not None:
        u0, v0, u1, v1 = feature.bounds
        w, d = b.x1 - b.x0 - 1, b.z1 - b.z0 - 1
        x0, x1 = b.x0 + u0 * w, b.x0 + u1 * w
        z0, z1 = b.z0 + v0 * d, b.z0 + v1 * d
        top = float(feature.meta.get("top_y", b.y1))
        span = max(x1 - x0, z1 - z0, 24.0)
        focus = ((x0 + x1) / 2, -(z0 + z1) / 2, top - span * 0.3)
        dist = span * 1.6
        return (focus[0] + dist * 0.6, focus[1] + dist * 0.6, top + dist * 0.55), focus
    side = view.removeprefix("overview_")
    sx, sy = _VIEW_DIRS[side]
    pull = _VIEW_PULL.get(side, 1.0)
    cx, cz = (b.x0 + b.x1) / 2, (b.z0 + b.z1) / 2
    span = max(b.x1 - b.x0, b.z1 - b.z0)
    focus = (cx, -cz, b.y0 + (b.y1 - b.y0) * 0.25)
    return (cx + sx * span * 0.85 * pull, -cz + sy * span * 0.85 * pull, b.y1 + span * 0.75 * pull), focus


def render_volume(volume: Volume, view: str = "overview_ne", feature: Feature | None = None, size_px: int = 1024,
                  title: str = "") -> bytes:
    """PNG bytes. `view` is one of VIEWS, or pass `feature` for a close-up of that feature."""
    import pyvista as pv
    from PIL import Image

    if feature is None and view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; use one of {VIEWS} or pass a feature for a close-up")
    names = np.array(volume.palette, dtype=object)
    category = np.where(names == AIR, 0, np.where(names == WATER, 2, 1)).astype(np.int8)
    kind = category[volume.data]

    plotter = pv.Plotter(off_screen=True, window_size=(size_px, size_px))
    plotter.set_background((0.72, 0.82, 0.95), top=(0.42, 0.6, 0.88))
    for which, opacity in ((1, 1.0), (2, 0.72)):
        p, f, rgb = _faces(volume, kind, which)
        if len(p):
            mesh = pv.PolyData(p, f)
            mesh.cell_data["rgb"] = rgb
            plotter.add_mesh(mesh, scalars="rgb", rgb=True, lighting=False, opacity=opacity, show_scalar_bar=False)

    b = volume.box
    position, focus = _camera(volume, view, feature)
    # North arrow and a 32-block scale bar, floating above the box corner nearest the camera (always in frame).
    mid_x = (b.x0 + b.x1) / 2
    # Seen square-on (overview_s) the near west corner projects to the frame edge, so move it inward.
    cx = b.x1 - 40 if position[0] > mid_x + 1 else b.x0 + 40 if abs(position[0] - mid_x) <= 1 else b.x0 + 8
    cy = -b.z0 - 40 if position[1] > -(b.z0 + b.z1) / 2 else -b.z1 + 8
    corner = np.array([cx, cy, b.y1 + 4], dtype=float)
    plotter.add_mesh(pv.Arrow(start=corner, direction=(0, 1, 0), scale=24), color="red", lighting=False)
    plotter.add_point_labels([corner + (0, 27, 0)], ["N"], font_size=20, text_color="red", shape=None, point_size=0, always_visible=True)
    bar = pv.Line(corner + (0, -6, 0), corner + (32, -6, 0))
    plotter.add_mesh(bar, color="black", line_width=5)
    plotter.add_point_labels([corner + (16, -10, 0)], ["32 blocks"], font_size=14, shape=None, point_size=0, always_visible=True)

    plotter.camera.position = position
    plotter.camera.focal_point = focus
    plotter.camera.up = (0, 0, 1)
    plotter.camera.view_angle = 35
    plotter.camera.clipping_range = (1, 20000)
    label = title or (f"close-up {feature.id}" if feature else view)
    plotter.add_text(label, position="upper_left", font_size=11, color="black")
    img = plotter.screenshot(return_img=True)
    plotter.close()

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()
