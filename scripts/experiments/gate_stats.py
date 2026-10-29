"""Acceptance-gate validation from traces (DESIGN §11.4): is the blind comparison biased?

    python -m scripts.experiments.gate_stats [traces/*.jsonl ...]

Reads every blind-comparison record (stage ".../compare") and reports the revision win rate and the
"Terrain 1" (position) win rate. Healthy: the revision wins clearly less than always (> ~90% over 20+
comparisons means the gate still favours the newer attempt) and Terrain 1 wins near half of the decisive
verdicts (otherwise enable --compare-both-orders, DESIGN §18).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

MIN_N = 20
MAX_REVISION_RATE = 0.9
POSITION_BAND = (0.3, 0.7)


def comparisons(paths: list[Path]) -> list[dict[str, Any]]:
    """Gate comparisons (revision vs. previous) from traces; other comparisons (e.g. gate_check) are skipped."""
    out = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            s = r.get("scores") or {}
            if r.get("stage", "").endswith("/compare") and "revision" in (s.get("order") or {}).values():
                out.append({"run_id": r["run_id"], **s})
    return out


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    winners = Counter(r["winner"] for r in records)
    decisive = [r for r in records if r["better"] != "same"]
    n = len(records)
    revision_rate = winners["revision"] / n if n else None
    first_rate = sum(r["better"] == "1" for r in decisive) / len(decisive) if decisive else None
    flags = []
    if n >= MIN_N and revision_rate is not None and revision_rate > MAX_REVISION_RATE:
        flags.append(f"revision wins {revision_rate:.0%} > {MAX_REVISION_RATE:.0%}: still biased toward the newer attempt")
    if len(decisive) >= MIN_N // 2 and first_rate is not None and not POSITION_BAND[0] <= first_rate <= POSITION_BAND[1]:
        flags.append(f"Terrain 1 wins {first_rate:.0%} of decisive verdicts: position bias, use --compare-both-orders")
    if n < MIN_N:
        flags.append(f"only {n} comparisons; need {MIN_N}+ for a verdict")
    return {"n": n, "winners": dict(winners), "revision_rate": revision_rate, "decisive": len(decisive),
            "terrain1_rate": first_rate, "flags": flags}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="*", type=Path)
    args = ap.parse_args()
    paths = args.traces or sorted(Path("traces").glob("*.jsonl"))
    records = comparisons(paths)
    s = summarize(records)
    pct = lambda x: "n/a" if x is None else f"{x:.0%}"  # noqa: E731
    print(f"{s['n']} comparisons from {len({r['run_id'] for r in records})} runs: {s['winners']}")
    print(f"revision win rate {pct(s['revision_rate'])}; Terrain 1 wins {pct(s['terrain1_rate'])} of {s['decisive']} decisive verdicts")
    for f in s["flags"]:
        print(f"FLAG: {f}")
    if not s["flags"]:
        print("OK: no bias detected")


if __name__ == "__main__":
    main()
