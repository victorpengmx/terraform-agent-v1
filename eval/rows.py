"""Result rows (`eval/results/<id>/runs.jsonl`, one JSON object per run) and what a run's trace says about it.

Rows are appended as runs finish; the last row per key wins, so a retried run (or a rescore) replaces the
earlier one. `trace_stats` reads the fields a report needs from the trace: tokens and cost per role, model
calls, tool errors, critic fix validity and the acceptance gate's decisions.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

ROLES = {"spec": "spec", "critique": "critic", "compare": "comparison"}  # stage tail -> role; anything else = builder
GATE_NOTE = re.compile(r"^(accepted|rolled back) round (\d+): (.*)$", re.S)


def read_rows(path: Path) -> dict[str, dict[str, Any]]:
    """key -> its latest row."""
    rows: dict[str, dict[str, Any]] = {}
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["key"]] = row
    return rows


def append_row(path: Path, row: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Replace the file atomically; the previous version is kept as <name>.bak."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    if path.exists():
        os.replace(path, path.with_suffix(path.suffix + ".bak"))
    os.replace(tmp, path)


def role_of(stage: str) -> str:
    return ROLES.get(stage.rsplit("/", 1)[-1], "builder")


def trace_stats(path: Path) -> dict[str, Any]:
    """Per-run figures from a trace: model calls, tokens and cost per role, tool use, critic fixes, gate events."""
    tokens: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0})
    cost: dict[str, float] = defaultdict(float)
    calls: dict[str, int] = defaultdict(int)
    unpriced = tool_calls = tool_errors = 0
    fixes = {"issues": 0, "with_fix": 0, "valid": 0, "none": 0}
    critiques = failed_critiques = 0
    gate: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        role = role_of(r["stage"])
        if r.get("usage"):
            calls[role] += 1
            for k, v in r["usage"].items():
                tokens[role][k] += v
            if r.get("cost_usd") is None:
                unpriced += 1
            else:
                cost[role] += r["cost_usd"]
        for c in r.get("tool_calls") or []:
            tool_calls += 1
            tool_errors += bool(c.get("is_error"))
        scores = r.get("scores") or {}
        if role == "critic" and "verdict" in scores:
            critiques += 1
            failed_critiques += bool(scores.get("failed"))
            for k in fixes:
                fixes[k] += (scores.get("fixes") or {}).get(k, 0)
        if role == "comparison" and "winner" in scores:
            comparisons.append({k: scores.get(k) for k in ("round", "order", "winner", "better")})
        if m := GATE_NOTE.match(r.get("note") or ""):
            gate.append({"round": int(m[2]), "decision": "accepted" if m[1] == "accepted" else "rolled_back", "why": m[3]})
    return {
        "calls": dict(calls), "steps": sum(calls.values()), "tokens": {k: dict(v) for k, v in tokens.items()},
        "cost_by_role": {k: round(v, 6) for k, v in cost.items()}, "unpriced_calls": unpriced,
        "tool_calls": tool_calls, "tool_errors": tool_errors, "critiques": critiques,
        "failed_critiques": failed_critiques, "fixes": fixes, "gate": gate, "comparisons": comparisons,
    }
