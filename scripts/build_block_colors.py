"""Build the block colour table for the L2 renderer from a local Minecraft 26.3 client jar.

    python -m scripts.build_block_colors [--jar PATH]

Resolves each block's model (following `parent`), averages the opaque pixels of its top and side
textures, applies the fixed biome tints for tinted faces, and writes
terrain/render/block_colors.json. Only these averaged numbers are committed, never textures
(CLAUDE.md conventions).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from terrain.materials import SURFACE_BLOCKS

OUT = Path(__file__).resolve().parents[1] / "terrain" / "render" / "block_colors.json"
DEFAULT_JAR = Path(os.environ.get("APPDATA", "")) / ".minecraft" / "versions" / "26.3" / "26.3.jar"
EXTRA_BLOCKS = ("water", "bedrock", "dirt", "stone", "lava", "oak_leaves", "spruce_leaves", "short_grass")

# Plains-biome tints for grey "tintindex" textures (vanilla default grass/foliage/water colours).
TINTS = {"grass": (145, 189, 89), "foliage": (119, 171, 47), "water": (63, 118, 228)}
TINTED = {"grass_block": {"top": "grass"}, "short_grass": {"top": "grass", "side": "grass"},
          "oak_leaves": {"top": "foliage", "side": "foliage"}, "spruce_leaves": {"top": "foliage", "side": "foliage"},
          "water": {"top": "water", "side": "water"}}  # fmt: skip
SPECIAL_TEXTURES = {"water": "block/water_still", "lava": "block/lava_still"}  # fluid models have no textures


def resolve_model(jar: zipfile.ZipFile, name: str) -> dict[str, str]:
    """Texture variables for a block model, merged down its parent chain."""
    textures: dict[str, str] = {}
    model = f"block/{name}"
    for _ in range(16):
        path = f"assets/minecraft/models/{model.removeprefix('minecraft:')}.json"
        if path not in jar.namelist():
            break
        data = json.loads(jar.read(path))
        for k, v in data.get("textures", {}).items():
            textures.setdefault(k, v)
        if "parent" not in data:
            break
        model = data["parent"]

    def deref(v: str) -> str:
        for _ in range(8):
            if not v.startswith("#"):
                return v.removeprefix("minecraft:")
            v = textures.get(v[1:], v)
        return v

    return {k: deref(v) for k, v in textures.items()}


def average_rgb(jar: zipfile.ZipFile, texture: str) -> list[int]:
    img = Image.open(io.BytesIO(jar.read(f"assets/minecraft/textures/{texture}.png"))).convert("RGBA")
    px = np.asarray(img, dtype=np.float64)
    px = px[: px.shape[1]]  # animated textures are vertical strips: use the first frame
    opaque = px[..., 3] > 127
    rgb = px[..., :3][opaque] if opaque.any() else px[..., :3].reshape(-1, 3)
    return [int(round(c)) for c in rgb.mean(axis=0)]


def face_textures(name: str, tex: dict[str, str]) -> dict[str, str]:
    if name in SPECIAL_TEXTURES:
        return {"top": SPECIAL_TEXTURES[name], "side": SPECIAL_TEXTURES[name]}
    top = tex.get("top") or tex.get("end") or tex.get("all") or tex.get("particle")
    side = tex.get("side") or tex.get("all") or tex.get("particle") or top
    if top is None:
        raise KeyError(f"no texture found for {name}: {tex}")
    return {"top": top, "side": side}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jar", type=Path, default=DEFAULT_JAR)
    args = ap.parse_args()
    if not args.jar.exists():
        raise SystemExit(f"client jar not found: {args.jar} (pass --jar)")

    table: dict[str, dict[str, list[int]]] = {}
    with zipfile.ZipFile(args.jar) as jar:
        for name in sorted(set(SURFACE_BLOCKS) | set(EXTRA_BLOCKS)):
            faces = face_textures(name, resolve_model(jar, name))
            entry = {}
            for face, texture in faces.items():
                rgb = average_rgb(jar, texture)
                tint = TINTED.get(name, {}).get(face)
                if tint:
                    rgb = [int(round(c * t / 255)) for c, t in zip(rgb, TINTS[tint])]
                entry[face] = rgb
            table[name] = entry
    OUT.write_text(json.dumps({"source": args.jar.name, "colors": table}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(table)} blocks to {OUT}")


if __name__ == "__main__":
    main()
