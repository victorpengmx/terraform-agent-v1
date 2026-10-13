"""Ranked list of missing tools (DESIGN §5.9, §11.5): critic issues whose fix needs a tool that doesn't exist.

    python -m scripts.missing_tools [traces/*.jsonl ...]

Collects every critic issue with tool "none" (critiques since the tool catalogue) and lists the missing
capabilities by how often they come up, so new tools are picked by evidence, not by appeal.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def missing(paths: list[Path]) -> Counter:
    found: Counter = Counter()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if not r.get("stage", "").endswith("/critique"):
                continue
            for i in (r.get("scores") or {}).get("issues") or []:
                if i.get("tool") == "none":
                    found[i["fix_hint"].strip()] += 1
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="*", type=Path)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    found = missing(args.traces or sorted(Path("traces").glob("*.jsonl")))
    print(f"{sum(found.values())} issues need a missing tool" + (":" if found else "."))
    for hint, n in found.most_common():
        print(f"  {n}x {hint}")


if __name__ == "__main__":
    main()
