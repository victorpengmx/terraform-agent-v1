"""Regression check (E7, DESIGN §16): the regression prompts with a fixed config, compared to a baseline.

    python -m eval.regression [--config eval/configs/regression.yaml] [--yes] [--update-baseline]
    python -m eval.regression --compare eval/results/<id> [--update-baseline]

Runs the config into eval/results/regression-<commit> (cost estimate first; needs --yes), then compares each
prompt with eval/baselines/regression.json and flags drops: success true -> false, a checklist pass rate
lower by more than 0.10, or cost up by more than 50% (a warning). Exits 1 when anything dropped.
--update-baseline accepts the results as the new baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from agent.run import ENV_FILE, git_state
from eval.benchmark import SUITE, load
from eval.config import load_config
from eval.rows import read_rows
from eval.run_eval import RESULTS, estimate, evaluate, pending, plan

CONFIG = Path("eval/configs/regression.yaml")
BASELINE = Path("eval/baselines/regression.json")
PASS_RATE_DROP = 0.10
COST_RISE = 0.5


def summary(rows: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """prompt id -> {success, pass_rate, cost_usd} (one run per prompt: seed 1, one condition)."""
    out = {}
    for r in rows.values():
        if r.get("status") == "done":
            out[r["prompt_id"]] = {"success": (r.get("success") or {}).get("overall"),
                                   "pass_rate": (r.get("answer_key") or {}).get("pass_rate"), "cost_usd": r.get("cost_usd")}
    return out


def compare(current: dict[str, dict[str, Any]], baseline: dict[str, dict[str, Any]]) -> tuple[list[str], list[str]]:
    """(drops, warnings)."""
    drops, warnings = [], []
    for pid in sorted(baseline.keys() | current.keys()):
        b, c = baseline.get(pid), current.get(pid)
        if b is None:
            warnings.append(f"{pid}: not in the baseline")
            continue
        if c is None:
            drops.append(f"{pid}: no finished run")
            continue
        if b["success"] is True and c["success"] is not True:
            drops.append(f"{pid}: success {b['success']} -> {c['success']}")
        if b["pass_rate"] is not None and c["pass_rate"] is not None and b["pass_rate"] - c["pass_rate"] > PASS_RATE_DROP:
            drops.append(f"{pid}: checklist pass rate {b['pass_rate']:.0%} -> {c['pass_rate']:.0%}")
        if b["cost_usd"] and c["cost_usd"] is not None and c["cost_usd"] > b["cost_usd"] * (1 + COST_RISE):
            warnings.append(f"{pid}: cost ${b['cost_usd']:.2f} -> ${c['cost_usd']:.2f}")
    return drops, warnings


def write_baseline(current: dict[str, dict[str, Any]], results: Path, path: Path = BASELINE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"results": results.as_posix(), "git": git_state(), "date": time.strftime("%Y-%m-%d"),
                                "prompts": current}, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--suite", type=Path, default=SUITE)
    ap.add_argument("--compare", type=Path, default=None, help="compare an existing results folder; no runs")
    ap.add_argument("--yes", action="store_true", help="spend: run the regression prompts")
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    results = args.compare
    if results is None:
        config = load_config(args.config)
        entries = load(args.suite)
        g = git_state() or {}
        results = RESULTS / f"regression-{g.get('commit', 'nogit')[:7]}{'-dirty' if g.get('dirty') else ''}"
        todo = pending(plan(config, entries), read_rows(results / "runs.jsonl"))
        print(estimate(config, todo)[0])
        if todo:
            if not args.yes:
                sys.exit("not run: add --yes to spend this")
            from dotenv import load_dotenv

            load_dotenv(ENV_FILE, override=False)
            evaluate(config, entries, results, args.suite.parent)
    current = summary(read_rows(results / "runs.jsonl"))
    if args.update_baseline:
        write_baseline(current, results)
        print(f"baseline updated from {results}")
        return
    if not BASELINE.exists():
        sys.exit(f"no baseline yet: run with --update-baseline to accept {results}")
    drops, warnings = compare(current, json.loads(BASELINE.read_text(encoding="utf-8"))["prompts"])
    for w in warnings:
        print(f"WARNING: {w}")
    for d in drops:
        print(f"DROP: {d}")
    print("no drops" if not drops else f"{len(drops)} drop(s) vs the baseline")
    sys.exit(1 if drops else 0)


if __name__ == "__main__":
    main()
