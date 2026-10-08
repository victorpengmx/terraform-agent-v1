r"""Compile an op stack into a Minecraft world through the AmuletBackend (copy-and-swap; never in place).

    python -m scripts.compile_world --world "$env:APPDATA\.minecraft\saves\TerraformTest" --stack renders/valley.json --origin 0 0
    python -m scripts.compile_world --world ... --stack ... --dry-run          # count changes, write nothing
    python -m scripts.compile_world --world ... --list-snapshots
    python -m scripts.compile_world --world ... --restore 0003                 # roll the world back

The world must be closed in Minecraft and the area's chunks generated (fly over it once).
Each write keeps the previous version as a snapshot under <.minecraft>/terraform_snapshots/<world>/.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from terrain.backends import BackendError
from terrain.backends.amulet_backend import AmuletBackend
from terrain.compiler.pipeline import build_patch
from terrain.opstack import OpStack


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", type=Path, required=True)
    ap.add_argument("--stack", type=Path, help="op stack JSON (e.g. renders/valley.json from scripts.demos)")
    ap.add_argument("--origin", type=int, nargs=2, default=(0, 0), metavar=("X", "Z"), help="world x, z of the area's north-west corner")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list-snapshots", action="store_true")
    ap.add_argument("--restore", metavar="SNAPSHOT_ID")
    args = ap.parse_args()

    try:
        backend = AmuletBackend(args.world)
        if args.list_snapshots:
            print("\n".join(backend.list_snapshots()) or "no snapshots")
            return
        if args.restore:
            backend.restore(args.restore)
            print(f"restored {args.restore}; the replaced version was kept as snapshot {backend.last_snapshot}")
            return
        if args.stack is None:
            raise SystemExit("--stack is required (or use --list-snapshots / --restore)")

        stack = OpStack.from_json(args.stack.read_text(encoding="utf-8"))
        s = stack.state
        x0, z0 = args.origin
        print(f"area {s.width}x{s.depth} at x={x0}..{x0 + s.width - 1}, z={z0}..{z0 + s.depth - 1}; {len(stack)} ops")

        ground, _ = backend.read_surface(x0, z0, x0 + s.width, z0 + s.depth)
        median = float(np.median(ground))
        print(f"world ground in the area: y={ground.min():.0f}..{ground.max():.0f} (median {median:.0f}); stack base y={stack.base.base_y:.0f}")
        if abs(median - stack.base.base_y) > 2:
            print("WARNING: the stack's base height differs from the world's ground; the area edges will show steps.")

        start = time.perf_counter()
        patch, volume = build_patch(s, backend, (x0, z0))
        print(f"{len(patch):,} blocks change (compile + diff {time.perf_counter() - start:.1f}s); y range {volume.box.y0}..{volume.box.y1 - 1}")
        for name, count in sorted(patch.summary().items(), key=lambda kv: -kv[1])[:12]:
            print(f"  {name:22s} {count:>9,}")
        if args.dry_run:
            print("dry run: nothing written")
            return

        start = time.perf_counter()
        backend.apply_patch(patch)
        print(f"written in {time.perf_counter() - start:.1f}s. Previous version kept as snapshot {backend.last_snapshot} "
              f"(undo: --restore {backend.last_snapshot}).")
        print(f"In game: /tp @s {x0 + s.width // 2} {int(s.height.max()) + 40} {z0 + s.depth + 40}")
    except BackendError as e:
        raise SystemExit(f"error: {e}") from None


if __name__ == "__main__":
    main()
