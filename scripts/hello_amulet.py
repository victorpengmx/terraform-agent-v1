r"""M0 Amulet hello-world: write a 64x64 sine-wave hill into a copy of a flat 26.3 world.

Usage (the source world must be closed in Minecraft):
    python -m scripts.hello_amulet "$env:APPDATA\.minecraft\saves\HelloFlat"

Writes to a sibling copy (<world>_hello) so the original is never touched, then prints what Amulet
believes the world height limits are. Also places two marker blocks to check those limits in game:
  - diamond_block at the top limit (max_y - 1) above the hill centre
  - gold_block at the bottom limit (min_y) under the hill centre (replaces bedrock)
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import amulet
from amulet.api.block import Block

DIM = "minecraft:overworld"


def assert_world_closed(world: Path) -> None:
    # A running game holds an OS lock on session.lock; reading it then fails on Windows.
    lock = world / "session.lock"
    if not lock.exists():
        return
    try:
        lock.read_bytes()
    except OSError as e:
        raise SystemExit(f"{world} appears to be open in Minecraft ({e}). Close it first.")


def find_ground_y(level, x: int, z: int, min_y: int, max_y: int) -> int:
    for y in range(max_y - 1, min_y - 1, -1):
        if level.get_block(x, y, z, DIM).base_name != "air":
            return y
    raise SystemExit(f"No ground found in column ({x}, {z}).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("world", type=Path, help="path to a closed flat 26.3 world folder")
    ap.add_argument("--out", type=Path, help="output world folder (default: <world>_hello)")
    ap.add_argument("--origin", type=int, nargs=2, default=(0, 0), metavar=("X", "Z"))
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--amplitude", type=float, default=24.0, help="hill height in blocks")
    args = ap.parse_args()

    src: Path = args.world.resolve()
    out: Path = (args.out or src.with_name(src.name + "_hello")).resolve()
    if not (src / "level.dat").exists():
        raise SystemExit(f"{src} is not a Minecraft world (no level.dat).")
    if out.exists():
        raise SystemExit(f"{out} already exists; delete it or pass --out.")
    assert_world_closed(src)

    shutil.copytree(src, out, ignore=shutil.ignore_patterns("session.lock"))
    print(f"copied {src.name} -> {out}")

    level = amulet.load_level(str(out))
    try:
        platform, version = level.level_wrapper.max_world_version
        bounds = level.bounds(DIM)
        min_y, max_y = bounds.min_y, bounds.max_y
        print(f"world version: {platform} {version}")
        print(f"amulet overworld bounds: min_y={min_y} max_y={max_y} (exclusive)")

        # Translate each block to Amulet's universal format once, then write palette ids in bulk.
        translator = level.translation_manager.get_version(platform, version).block

        def block_id(name: str) -> int:
            universal, _, _ = translator.to_universal(Block("minecraft", name))
            return level.block_palette.get_add_block(universal)

        stone, dirt, grass = block_id("stone"), block_id("dirt"), block_id("grass_block")
        diamond, gold = block_id("diamond_block"), block_id("gold_block")

        ox, oz = args.origin
        n = args.size
        ground_y = find_ground_y(level, ox, oz, min_y, max_y)
        print(f"ground surface at y={ground_y}")

        def write_column(x: int, z: int, ids_by_y: dict[int, int]) -> None:
            cx, lx = divmod(x, 16)
            cz, lz = divmod(z, 16)
            chunk = level.get_chunk(cx, cz, DIM)
            for y, bid in ids_by_y.items():
                chunk.blocks[lx, y, lz] = bid
            chunk.changed = True

        written = 0
        for i in range(n):
            for j in range(n):
                # Heights are floats until the moment of writing (CLAUDE.md conventions).
                h = args.amplitude * math.sin(math.pi * i / (n - 1)) * math.sin(math.pi * j / (n - 1))
                top = ground_y + round(h)
                column = {y: (stone if y < top - 3 else dirt) for y in range(ground_y, top)}
                column[top] = grass
                write_column(ox + i, oz + j, column)
                written += len(column)

        cx, cz = ox + n // 2, oz + n // 2
        write_column(cx, cz, {max_y - 1: diamond, min_y: gold})

        level.save()
        print(f"wrote {written} hill blocks + 2 markers at ({cx}, {max_y - 1}, {cz}) and ({cx}, {min_y}, {cz})")
    finally:
        level.close()

    print(f"\nOpen '{out.name}' in Minecraft 26.3 and check with F3:")
    print(f"  1. hill around x={ox}..{ox + n - 1}, z={oz}..{oz + n - 1}, peak near y={ground_y + round(args.amplitude)}")
    print(f"  2. diamond_block at y={max_y - 1}; /setblock {cx} {max_y} {cz} stone should fail (above limit)")
    print(f"  3. gold_block at y={min_y}; /setblock {cx} {min_y - 1} {cz} stone should fail (below limit)")


if __name__ == "__main__":
    main()
