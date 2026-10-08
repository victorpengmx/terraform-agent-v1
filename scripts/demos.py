"""Hand-scripted terrains (CLAUDE.md principle 5): tools must look good when called by a script.

    python -m scripts.demos [valley|island|mesa|all] [--seed 1] [--no-l2]

Each demo is an op stack built only through the public ops. Writes, per demo, to renders/:
<name>.json (op stack), <name>_l1.png (relief map) and <name>_l2_<view>.png (voxel previews).
Compile one into a world with `python -m scripts.compile_world --world <dir> --stack renders/<name>.json`.
All demos assume the surrounding world's ground is at y=64 (the deep superflat test world).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from terrain.opstack import BaseSpec, OpStack

Op = tuple[str, str, dict[str, Any]]
WHOLE = [[0, 0], [1, 0], [1, 1], [0, 1]]


def ring(cx: float, cy: float, r: float, n: int = 10, wobble: float = 0.0) -> list[list[float]]:
    """Polygon approximating a circle; `wobble` alternates the radius for a rougher outline."""
    import math

    return [[round(cx + r * (1 + wobble * (-1) ** k) * math.cos(2 * math.pi * k / n), 3),
             round(cy + r * (1 + wobble * (-1) ** k) * math.sin(2 * math.pi * k / n), 3)] for k in range(n)]


DEMOS: dict[str, tuple[int, list[Op]]] = {
    "valley": (256, [
        ("base_noise", "noise_base", {"amplitude": 6, "feature_size": 0.3, "roughness": 0.5}),
        ("mountain_west", "raise_peak", {"center": [0.22, 0.3], "radius": 0.3, "height": 110, "steepness": 0.63, "roughness": 0.6, "ridges": 0.7}),
        ("mountain_east", "raise_peak", {"center": [0.8, 0.25], "radius": 0.25, "height": 85, "steepness": 0.68, "roughness": 0.5, "ridges": 0.8}),
        ("foothills", "raise_region", {"region": {"polygon": [[0.05, 0.55], [0.45, 0.5], [0.4, 0.75], [0.05, 0.8]]}, "height": 14, "edge": 0.08}),
        # Detail passes: sculpted shapes first, then general noise and erosion.
        ("bend_west", "warp", {"region": {"feature": "mountain_west"}, "amount": 0.06, "scale": 0.2}),
        ("bend_east", "warp", {"region": {"feature": "mountain_east"}, "amount": 0.05, "scale": 0.18}),
        ("main_valley", "carve_valley", {"path": [[0.5, 0.0], [0.52, 0.45], [0.62, 0.7], [0.75, 1.0]], "width": 0.28, "depth": 18}),
        ("lake_basin", "raise_region", {"region": {"polygon": [[0.55, 0.55], [0.72, 0.55], [0.74, 0.7], [0.56, 0.72]]}, "height": -8, "edge": 0.04}),
        ("runoff", "erode", {"kind": "hydraulic", "strength": 0.75}),
        # Crags after erosion: sharp rock standing above the weathered slopes.
        ("crags_west", "roughen", {"region": {"feature": "mountain_west"}, "amount": 18, "scale": 0.08, "style": "ridged", "weight": "relief"}),
        ("crags_east", "roughen", {"region": {"feature": "mountain_east"}, "amount": 14, "scale": 0.07, "style": "ridged", "weight": "relief"}),
        ("lake", "fill_lake", {"at": [0.64, 0.63]}),
        ("river", "carve_river", {"path": [[0.5, 0.0], [0.53, 0.35], [0.62, 0.6]], "width": 0.03, "depth": 3, "meander": 0.5}),
        ("outflow", "carve_river", {"path": [[0.66, 0.7], [0.72, 0.85], [0.78, 1.0]], "width": 0.03, "depth": 3, "meander": 0.4}),
    ]),
    "island": (256, [
        ("seafloor", "raise_region", {"region": {"polygon": WHOLE}, "height": -26, "edge": 0}),
        ("seafloor_noise", "noise_base", {"amplitude": 4, "feature_size": 0.2, "roughness": 0.5}),
        ("island", "raise_region", {"region": {"polygon": ring(0.5, 0.5, 0.3, n=12, wobble=0.12)}, "height": 30, "edge": 0.1}),
        ("coastline", "warp", {"region": {"feature": "island"}, "amount": 0.08, "scale": 0.18}),
        ("volcano", "raise_peak", {"center": [0.52, 0.48], "radius": 0.26, "height": 95, "steepness": 0.54, "roughness": 0.5, "ridges": 0.8}),
        ("lava_ridges", "roughen", {"region": {"feature": "volcano"}, "amount": 9, "scale": 0.07, "style": "ridged", "weight": "relief"}),
        ("runoff", "erode", {"kind": "hydraulic", "strength": 0.5}),
        ("lava_crags", "roughen", {"region": {"feature": "volcano"}, "amount": 6, "scale": 0.05, "style": "ridged", "weight": "relief"}),
        ("crater", "raise_region", {"region": {"polygon": ring(0.52, 0.48, 0.05, n=10)}, "height": -10, "mode": "basin", "edge": 0.03}),
        ("crater_lake", "fill_lake", {"at": [0.52, 0.48]}),
        ("sea", "set_sea_level", {"level": 62}),
        ("palette", "set_palette_rules", {"rules": [
            {"when": {"y_min": 135}, "use": {"kind": "mix", "blocks": {"basalt": 0.5, "smooth_basalt": 0.3, "tuff": 0.2}}},
            {"when": {"slope_min": 48}, "use": {"kind": "banded", "blocks": ["basalt", "tuff", "blackstone", "tuff"], "band_min": 2, "band_max": 4}},
            {"when": {"water_dist_max": 3, "y_max": 66}, "use": {"kind": "mix", "blocks": {"sand": 0.85, "gravel": 0.15}}},
            {"when": {"y_max": 58}, "use": {"kind": "mix", "blocks": {"sand": 0.6, "gravel": 0.25, "clay": 0.15}}},
            {"when": {"slope_min": 32}, "use": {"kind": "mix", "blocks": {"coarse_dirt": 0.4, "tuff": 0.3, "grass_block": 0.3}}},
            {"use": {"kind": "mix", "blocks": {"grass_block": 0.85, "podzol": 0.15}, "patch_size": 0.02}},
        ]}),
    ]),
    "mesa": (256, [
        ("base_noise", "noise_base", {"amplitude": 3, "feature_size": 0.25, "roughness": 0.5}),
        ("mesa_north", "raise_region", {"region": {"polygon": [[0.08, 0.1], [0.45, 0.06], [0.5, 0.3], [0.3, 0.42], [0.1, 0.35]]}, "height": 42, "mode": "plateau", "edge": 0.015}),
        ("mesa_east", "raise_region", {"region": {"polygon": [[0.62, 0.45], [0.92, 0.4], [0.95, 0.75], [0.7, 0.8]]}, "height": 30, "mode": "plateau", "edge": 0.015}),
        ("terrace", "raise_region", {"region": {"polygon": [[0.68, 0.5], [0.88, 0.47], [0.9, 0.68], [0.73, 0.72]]}, "height": 14, "mode": "add", "edge": 0.01}),
        ("butte", "raise_region", {"region": {"polygon": ring(0.3, 0.72, 0.06, n=8, wobble=0.1)}, "height": 55, "mode": "plateau", "edge": 0.01}),
        ("canyon", "carve_valley", {"path": [[0.0, 0.55], [0.35, 0.5], [0.55, 0.6], [1.0, 0.92]], "width": 0.1, "depth": 12, "profile": "u"}),
        # Warp after the canyon so its walls bend too (straight banks looked artificial in game).
        ("scallops", "warp", {"amount": 0.05, "scale": 0.1}),
        ("broken_walls", "roughen", {"amount": 3, "scale": 0.04, "style": "fbm", "weight": "slope"}),
        ("weathering", "erode", {"strength": 0.14}),
        ("creek", "carve_river", {"path": [[0.0, 0.55], [0.35, 0.5], [0.55, 0.6], [1.0, 0.92]], "width": 0.02, "depth": 3, "meander": 0.3}),
        ("palette", "set_palette_rules", {"rules": [
            {"when": {"slope_min": 45}, "use": {"kind": "banded", "blocks": ["terracotta", "orange_terracotta", "red_terracotta", "yellow_terracotta", "white_terracotta", "brown_terracotta", "light_gray_terracotta"], "band_min": 1, "band_max": 4, "noise": 0.15}},
            {"when": {"water_dist_max": 2}, "use": {"kind": "mix", "blocks": {"red_sand": 0.7, "gravel": 0.3}}},
            {"when": {"y_min": 90}, "use": {"kind": "mix", "blocks": {"coarse_dirt": 0.5, "grass_block": 0.3, "red_sand": 0.2}, "patch_size": 0.02}},
            {"when": {"slope_min": 25}, "use": {"kind": "mix", "blocks": {"orange_terracotta": 0.5, "red_sand": 0.5}}},
            {"use": {"kind": "mix", "blocks": {"red_sand": 0.8, "coarse_dirt": 0.2}}},
        ]}),
    ]),
}


def build(name: str, seed: int = 1) -> OpStack:
    size, ops = DEMOS[name]
    stack = OpStack(BaseSpec(size, size, base_y=64, seed=seed))
    for op_id, op, params in ops:
        stack.add(op, params, id=op_id)
    return stack


def run(name: str, seed: int, l2: bool, out: Path) -> None:
    from terrain.render.l1 import render_l1

    start = time.perf_counter()
    stack = build(name, seed)
    print(f"[{name}] built {len(stack)} ops in {time.perf_counter() - start:.2f}s")
    s = stack.state
    for e in stack.entries:
        f = s.features.get(e.id)
        where = "(no height/water change)" if f is None else f"u={f.bounds[0]:.2f}..{f.bounds[2]:.2f} v={f.bounds[1]:.2f}..{f.bounds[3]:.2f} y={f.meta['bottom_y']:.0f}..{f.meta['top_y']:.0f}"
        print(f"  {e.id:14s} {e.op:17s} {where}")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(stack.to_json(), encoding="utf-8")
    (out / f"{name}_l1.png").write_bytes(render_l1(s, title=f"{name} (seed {seed})"))
    if l2:
        from terrain.compiler import compile_state
        from terrain.render.l2 import render_volume

        vol = compile_state(s)
        for view in ("overview_ne", "overview_sw"):
            (out / f"{name}_l2_{view}.png").write_bytes(render_volume(vol, view, title=f"{name} {view}"))
    print(f"  wrote {out / name}.json and renders")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("demo", nargs="?", default="all", choices=[*DEMOS, "all"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-l2", action="store_true", help="skip the voxel previews (no PyVista needed)")
    ap.add_argument("--out", type=Path, default=Path("renders"))
    args = ap.parse_args()
    for name in DEMOS if args.demo == "all" else [args.demo]:
        run(name, args.seed, not args.no_l2, args.out)


if __name__ == "__main__":
    main()
