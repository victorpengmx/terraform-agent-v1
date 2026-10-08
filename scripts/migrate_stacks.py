"""Migrate saved op stacks to the schemas without raw knobs (DESIGN §5.1 principle 3; schema change of 2026-10-21).

    python -m scripts.migrate_stacks renders/valley.json renders/*/stack.json [--backup-dir renders/_pre_m2]

Rewrites each file in place (the original goes to the backup dir) with old values mapped to the nearest
new params, then checks that the result loads. Already-migrated ops are left alone, so it is safe to rerun.

- noise_base: `octaves` dropped (derived from feature_size); `roughness` was the octave persistence
  (0.1-0.9) and becomes the 0-1 scale with the same persistence.
- roughen: `octaves` dropped (derived from scale).
- raise_peak: `sharpness` -> the `steepness` that gives the same exponent (unrounded); irregularity and
  ridges stay explicit (old default 0), roughness = irregularity.
- erode: `iterations`/`talus_deg`/`deposition` -> `strength`. thermal: mean of the strengths that match
  the talus angle and the iteration count (they map to one strength only on the old default line);
  hydraulic: by total cutting (iterations x old strength). `deposition` was never varied by
  the demos and is dropped (fixed at the old default 0.5).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

from terrain.ops.erode import HYDRAULIC_ITERATIONS, HYDRAULIC_RATE, THERMAL_ITERATIONS, THERMAL_TALUS_DEG
from terrain.ops.noise import PERSISTENCE
from terrain.ops.peaks import SHARPNESS, SHARPNESS_CURVE
from terrain.opstack import OpStack


def _clip01(x: float, digits: int | None = 4) -> float:
    x = min(1.0, max(0.0, x))
    return x if digits is None else round(x, digits)


def migrate_op(op: str, params: dict[str, Any]) -> dict[str, Any]:
    """Old-schema params -> nearest new-schema params. New-schema params come back unchanged."""
    p = dict(params)
    if op == "noise_base" and "octaves" in p:
        p.pop("octaves")
        lo, hi = PERSISTENCE
        p["roughness"] = _clip01((p.get("roughness", 0.5) - lo) / (hi - lo))
        p.pop("character", None)
    elif op == "roughen":
        p.pop("octaves", None)
    elif op == "raise_peak" and "sharpness" in p:
        lo, hi = SHARPNESS
        exponent = min(hi, max(lo, p.pop("sharpness")))
        # Unrounded: the same exponent to the last bit, so migrated stacks replay bit-identical.
        p["steepness"] = _clip01((math.log(exponent / lo) / math.log(hi / lo)) ** (1 / SHARPNESS_CURVE), digits=None)
        p.setdefault("irregularity", 0.0)
        p.setdefault("ridges", 0.0)
        p["roughness"] = p["irregularity"]
        p.pop("character", None)
    elif op == "erode" and ("iterations" in p or "talus_deg" in p):
        iterations, talus_deg = p.pop("iterations", 30), p.pop("talus_deg", 40.0)
        rate = p.pop("strength", 0.5)
        if p.get("kind", "thermal") == "thermal":  # mean of the strengths matching the angle and the pass count
            (t0, t1), (i0, i1) = THERMAL_TALUS_DEG, THERMAL_ITERATIONS
            p["strength"] = _clip01(((talus_deg - t0) / (t1 - t0) + (iterations - i0) / (i1 - i0)) / 2)
        else:
            lo, hi = HYDRAULIC_ITERATIONS
            p["strength"] = _clip01((iterations * rate / HYDRAULIC_RATE - lo) / (hi - lo))
        p.pop("deposition", None)
    elif op == "erode":
        p.pop("deposition", None)  # stacks from the reverted 2026-10-21 strength experiment
    return p


def migrate_stack(data: dict[str, Any]) -> list[str]:
    """Migrate in place; returns one line per changed op."""
    changes = []
    for o in data["ops"]:
        new = migrate_op(o["op"], o["params"])
        if new != o["params"]:
            gone = {k: o["params"][k] for k in o["params"].keys() - new.keys()}
            changed = {k: v for k, v in new.items() if o["params"].get(k) != v}
            changes.append(f"{o['id']} ({o['op']}): {gone} -> {changed}")
            o["params"] = new
    return changes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stacks", nargs="+", type=Path)
    ap.add_argument("--backup-dir", type=Path, default=Path("renders/_pre_m2"))
    args = ap.parse_args()
    for path in args.stacks:
        data = json.loads(path.read_text(encoding="utf-8"))
        changes = migrate_stack(data)
        if not changes:
            print(f"{path}: nothing to migrate")
            continue
        backup = args.backup_dir / path.parent.name / path.name
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        print(f"{path}: migrated {len(changes)} ops (original: {backup})")
        for c in changes:
            print(f"  {c}")
        try:
            OpStack.from_json(path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001  (other schema changes, e.g. reverted ops, are reported, not fixed)
            print(f"  still does not load: {str(e)[:160]}")


if __name__ == "__main__":
    main()
