"""README image pairs: L2 previews plus matching in-game camera poses (manual stand-in for the L3 capture
script, ROADMAP "After applying").

    python -m scripts.readme_shots renders/valley.json@0,0 renders/island.json@300,0 renders/mesa.json@600,0

Each target is `<op stack JSON>@<world x>,<world z>` (the origin it was compiled at). Per target and view,
writes docs/images/<name>_l2_<view>.png and prints a /tp command whose camera matches that render, for
the "after" screenshot. The "before" pose is the same camera over a flat, unbuilt area (--before-origin).
In game: FOV 35 (= the L2 view angle), render distance 32, F1 to hide the HUD, F2 to take the screenshot,
then crop it to its centre square to match the (square) L2 render. Lines marked "also in frame" name other
targets visible in that square from that pose.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from terrain.opstack import OpStack
from terrain.render.l2 import _camera, render_volume

FOV_DEG = 35.0  # vertical; matches plotter.camera.view_angle in terrain/render/l2.py
ASPECT = 1.0  # the L2 render is square: crop the screenshot to its centre square to match
RENDER_DIST = 512  # 32 chunks, the game's maximum
EYE = 1.62  # /tp sets the feet position


def world_pose(position: tuple, focus: tuple) -> tuple[np.ndarray, np.ndarray]:
    """L2 camera coordinates are (x, -z, y); convert camera and focus to world (x, y, z)."""
    to_world = lambda p: np.array([p[0], p[2], -p[1]], dtype=float)  # noqa: E731
    return to_world(position), to_world(focus)


def tp_command(eye: np.ndarray, focus: np.ndarray) -> str:
    dx, dy, dz = focus - eye
    yaw = math.degrees(math.atan2(-dx, dz))
    pitch = -math.degrees(math.atan2(dy, math.hypot(dx, dz)))
    return f"/tp @s {eye[0]:.1f} {eye[1] - EYE:.1f} {eye[2]:.1f} {yaw:.1f} {pitch:.1f}"


def in_frame(eye: np.ndarray, focus: np.ndarray, box: tuple[float, float, float, float, float, float]) -> bool:
    """True if any sample point of the world box (x0, x1, y0, y1, z0, z1) is inside the view and render distance."""
    fwd = (focus - eye) / np.linalg.norm(focus - eye)
    right = np.cross(fwd, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    tv = math.tan(math.radians(FOV_DEG / 2))
    x0, x1, y0, y1, z0, z1 = box
    for x in np.linspace(x0, x1, 9):
        for z in np.linspace(z0, z1, 9):
            for y in (y0, y1):
                d = np.array([x, y, z]) - eye
                depth = d @ fwd
                if 0 < depth and np.linalg.norm(d) <= RENDER_DIST and abs(d @ up) <= depth * tv and abs(d @ right) <= depth * tv * ASPECT:
                    return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="+", help="<stack.json>@<x>,<z>")
    ap.add_argument("--views", nargs="+", default=["overview_ne", "overview_sw"])
    ap.add_argument("--before-origin", type=int, nargs=2, default=(-400, -400), metavar=("X", "Z"),
                    help="north-west corner of a flat, unbuilt area for the 'before' shots")
    ap.add_argument("--out", type=Path, default=Path("docs/images"))
    ap.add_argument("--no-render", action="store_true", help="print poses only")
    args = ap.parse_args()

    from terrain.compiler import compile_state

    loaded = []
    for t in args.targets:
        path, _, xz = t.rpartition("@")
        x, z = (int(v) for v in xz.split(","))
        name = Path(path).stem if Path(path).stem != "stack" else Path(path).parent.name
        state = OpStack.from_json(Path(path).read_text(encoding="utf-8")).state
        loaded.append((name, (x, z), compile_state(state, origin=(x, z))))
    boxes = {n: (v.box.x0, v.box.x1, 64, float(v.box.y1), v.box.z0, v.box.z1) for n, _, v in loaded}

    args.out.mkdir(parents=True, exist_ok=True)
    bx, bz = args.before_origin
    for name, (x, z), volume in loaded:
        print(f"{name} (origin {x}, {z})")
        for view in args.views:
            eye, focus = world_pose(*_camera(volume, view, None))
            shift = np.array([bx - x, 0, bz - z], dtype=float)
            if not args.no_render:
                png = args.out / f"{name}_l2_{view}.png"
                png.write_bytes(render_volume(volume, view, title=f"{name}: L2 preview ({view})"))
            for label, e, f in (("after ", eye, focus), ("before", eye + shift, focus + shift)):
                others = [n for n, b in boxes.items() if (label == "before" or n != name) and in_frame(e, f, b)]
                note = f"   also in frame: {', '.join(others)}" if others else ""
                print(f"  {view} {label}: {tp_command(e, f)}{note}")
    if not args.no_render:
        print(f"L2 previews in {args.out}/")


if __name__ == "__main__":
    main()
