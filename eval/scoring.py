"""Grading a final op stack (DESIGN §16): the answer-key checklist, the programmatic checks and the success rule.

Success = no failed major item on the answer-key checklist (matcher mode, `eval.match.match_checklist`) AND no
failed blocking or major programmatic check AND the eval judge passes. Each part is stored on its own; when
the judge gave no usable verdict, overall success is unknown (None), never a failure. The checklist pass rate
(passed / all items; skipped items count as not passed) is also reported as a continuous score.
"""

from __future__ import annotations

from typing import Any

from agent.checks import run_checks
from agent.session import Session
from agent.spec import FeatureGraph
from eval.match import match, match_checklist
from terrain.opstack import OpStack


def answer_key(graph: FeatureGraph, stack: OpStack) -> dict[str, Any]:
    items = match_checklist(graph, stack)
    passed = sum(i.status == "pass" for i in items)
    return {
        "mode": "matcher",
        "items": [{"id": i.id, "status": i.status, "severity": i.severity, "detail": i.detail} for i in items],
        "passed": passed,
        "total": len(items),
        "pass_rate": passed / len(items) if items else None,
        "major_failures": [i.id for i in items if i.status == "fail" and i.severity == "major"],
        "matched": match(graph, stack).ops,
    }


def checks(stack: OpStack) -> dict[str, Any]:
    results = run_checks(Session(stack))
    return {
        "results": [{"name": c.name, "severity": c.severity, "passed": c.passed, "detail": c.detail} for c in results],
        "failed": [c.name for c in results if not c.passed and c.severity in ("blocking", "major")],
    }


def success(answer: dict[str, Any], checked: dict[str, Any], judge: dict[str, Any] | None) -> dict[str, bool | None]:
    checklist_ok = not answer["major_failures"]
    checks_ok = not checked["failed"]
    judge_ok = None if judge is None or judge.get("passed") is None else bool(judge["passed"])
    overall = None if judge_ok is None else checklist_ok and checks_ok and judge_ok
    return {"checklist": checklist_ok, "checks": checks_ok, "judge": judge_ok, "overall": overall}
