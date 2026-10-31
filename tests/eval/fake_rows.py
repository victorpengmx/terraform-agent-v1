"""Fake result rows in the runs.jsonl format (eval/run_eval.py), for report, calibration and regression tests."""

from __future__ import annotations

import random
from typing import Any

PROMPTS = [("t01", "text"), ("t02", "text"), ("t03", "text"), ("t04", "text"), ("i01", "image"), ("i02", "image")]


def row(prompt_id: str, kind: str, condition: str, seed: int, *, overall: bool | None, pass_rate: float,
        cost: float | None = 0.5, latency: float = 300.0, spec_graph: dict | None = None, **extra: Any) -> dict[str, Any]:
    total = 10
    passed = round(pass_rate * total)
    judge_passed = overall if overall is not None else None
    return {
        "key": f"{prompt_id}-{condition}-s{seed}", "prompt_id": prompt_id, "kind": kind, "condition": condition, "seed": seed,
        "status": "done", "config": "fake", "git": {"commit": "abcdef1234", "dirty": False},
        "models": {"builder": "claude-haiku-5-5", "critic": None if condition == "critic_off" else "claude-opus-5-5",
                   "judge": "gemini-3.1-flash-lite", "judge_same_family": False},
        "stop": "max_rounds", "rounds": 3, "accepted_round": 2, "cost_usd": cost, "cost_text": f"${cost}",
        "latency_s": latency, "spec_graph": spec_graph,
        "answer_key": {"mode": "matcher", "items": [], "passed": passed, "total": total, "pass_rate": passed / total,
                       "major_failures": [] if overall else ["x.exists"], "matched": {}},
        "checks": {"results": [], "failed": []},
        "judge": {"model": "gemini-3.1-flash-lite", "passed": judge_passed, "mean": 3.5 if judge_passed else 2.5,
                  "output": None if judge_passed is None else {"verdict": "pass" if judge_passed else "fail"},
                  "cost_usd": 0.003, "cached": False},
        "success": {"checklist": bool(overall), "checks": True, "judge": judge_passed, "overall": overall},
        "trace_stats": {"calls": {"builder": 20, "critic": 3, "comparison": 2}, "steps": 25,
                        "tokens": {"builder": {"input": 100, "output": 50, "cache_read": 1000, "cache_write": 0}},
                        "tool_calls": 10, "tool_errors": 1, "fixes": {"issues": 4, "with_fix": 3, "valid": 3, "none": 1},
                        "gate": [{"round": 1, "decision": "accepted", "why": "first"}, {"round": 2, "decision": "accepted", "why": "won"},
                                 {"round": 3, "decision": "rolled_back", "why": "lost"}],
                        "comparisons": [{"round": 2, "order": {}, "winner": "revision", "better": "1"},
                                        {"round": 3, "order": {}, "winner": "previous", "better": "1"}]},
        "trace": "x.jsonl", "stack": "x/stack.json",
    } | extra


def e1_rows(seed: int = 0) -> list[dict[str, Any]]:
    """critic_on clearly better than critic_off on pass rate; success mixed."""
    rng = random.Random(seed)
    out = []
    for pid, kind in PROMPTS:
        for s in (1, 2):
            on = rng.random() < 0.7
            out.append(row(pid, kind, "critic_on", s, overall=on, pass_rate=0.8 + rng.random() * 0.2))
            out.append(row(pid, kind, "critic_off", s, overall=rng.random() < 0.4, pass_rate=0.4 + rng.random() * 0.2,
                           cost=0.03, latency=60.0))
    return out
