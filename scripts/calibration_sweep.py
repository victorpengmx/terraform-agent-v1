"""Calibration sweep for the 0-1 scales (DESIGN §5.3): render each scale at 0.1 ... 0.9 so
a person can check by eye that ~0.2 reads gentle and ~0.8 reads jagged.

    python -m scripts.calibration_sweep [--out renders/calibration] [--size 256] [--only erode_thermal ...]

Per sweep, writes <sweep>_<value>_l1.png / _l2.png and a contact sheet <sweep>.png (L1 row over L2 row).
renders/ is not committed; record the verdict in docs/EVIDENCE.md.
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path
from typing import Any

from terrain.opstack import BaseSpec, OpStack

VALUES = (0.1, 0.3, 0.5, 0.7, 0.9)
PEAK = {"center": [0.5, 0.5], "radius": 0.35, "height": 100}
GROUND = ("ground", "noise_base", {"amplitude": 2, "feature_size": 0.3, "roughness": 0.3})

# name -> (scene ops before the swept op, swept op, its fixed params, swept param, values)
SWEEPS: dict[str, tuple[list[tuple[str, str, dict[str, Any]]], str, dict[str, Any], str, tuple[Any, ...]]] = {
    "noise_base_roughness": ([], "noise_base", {"amplitude": 8, "feature_size": 0.25}, "roughness", VALUES),
    "raise_peak_steepness": ([GROUND], "raise_peak", PEAK | {"roughness": 0.55}, "steepness", VALUES),
    "raise_peak_roughness": ([GROUND], "raise_peak", PEAK | {"steepness": 0.6}, "roughness", VALUES),
    "raise_peak_character": ([GROUND], "raise_peak", PEAK, "character", ("gentle", "rolling", "rugged", "jagged")),
    "erode_thermal_strength": (
        [GROUND, ("spire", "raise_peak", PEAK | {"character": "jagged"}),
         ("crags", "roughen", {"amount": 14, "scale": 0.08, "style": "ridged", "weight": "relief"})],
        "erode", {"kind": "thermal"}, "strength", (None, *VALUES)),
    "erode_hydraulic_strength": (
        [GROUND, ("mountain", "raise_peak", PEAK)], "erode", {"kind": "hydraulic"}, "strength", (None, *VALUES)),
}


def build(size: int, scene: list, op: str, fixed: dict[str, Any], param: str, value: Any) -> OpStack:
    stack = OpStack(BaseSpec(size, size, base_y=64, seed=1))
    for op_id, scene_op, params in scene:
        stack.add(scene_op, params, id=op_id)
    if value is not None:  # None = the scene before the swept op, for comparison
        stack.add(op, fixed | {param: value}, id="swept")
    return stack


def contact_sheet(rows: list[list[bytes]], labels: list[str], tile: int) -> bytes:
    from PIL import Image, ImageDraw

    sheet = Image.new("RGB", (tile * len(labels), tile * len(rows) + 24), "white")
    draw = ImageDraw.Draw(sheet)
    for c, label in enumerate(labels):
        draw.text((c * tile + 8, 6), label, fill="black")
        for r, row in enumerate(rows):
            sheet.paste(Image.open(io.BytesIO(row[c])).convert("RGB").resize((tile, tile)), (c * tile, 24 + r * tile))
    out = io.BytesIO()
    sheet.save(out, format="PNG")
    return out.getvalue()


def main() -> None:
    from terrain.compiler import compile_state
    from terrain.render.l1 import render_l1
    from terrain.render.l2 import render_volume

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("renders/calibration"))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--tile", type=int, default=384)
    ap.add_argument("--only", nargs="*", choices=list(SWEEPS))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.only or SWEEPS:
        scene, op, fixed, param, values = SWEEPS[name]
        l1s, l2s, labels = [], [], []
        for v in values:
            start = time.perf_counter()
            state = build(args.size, scene, op, fixed, param, v).state
            label = "before" if v is None else f"{param}={v}"
            stem = args.out / f"{name}_{'before' if v is None else v}"
            l1 = render_l1(state, title=f"{op} {label}", size_px=args.tile)
            l2 = render_volume(compile_state(state), "overview_sw", size_px=args.tile, title=f"{op} {label}")
            stem.with_name(stem.name + "_l1.png").write_bytes(l1)
            stem.with_name(stem.name + "_l2.png").write_bytes(l2)
            l1s.append(l1), l2s.append(l2), labels.append(label)
            print(f"{name} {label}: y {state.height.min():.0f}..{state.height.max():.0f} ({time.perf_counter() - start:.1f}s)")
        (args.out / f"{name}.png").write_bytes(contact_sheet([l1s, l2s], labels, args.tile))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
