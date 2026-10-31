"""A/B for the critic's tool catalogue (DESIGN §11.5): same terrains, same renders, prompt with and without it.

    python -m scripts.experiments.catalogue_ab [--model claude-haiku-5-5] [--limit N]

Re-critiques every saved final op stack (renders/<run_id>/stack.json, prompt from traces/<run_id>.jsonl) once
per arm and checks each structured fix with `hint_problems`: does it name a real tool, real parameters and
values they accept? Fixes with tool "none" are the critic's list of missing tools. Records go to
traces/catalogue_ab-<run_id>.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from agent.catalogue import hint_problems
from agent.critic import Critic, CriticError
from agent.session import Session
from agent.tracing import Tracer, TraceRecord
from terrain.opstack import OpStack


def saved_runs(limit: int | None) -> list[tuple[str, str, OpStack]]:
    runs = []
    for stack_file in sorted(Path("renders").glob("2*/stack.json")):
        run_id = stack_file.parent.name
        trace = Path("traces") / f"{run_id}.jsonl"
        if not trace.exists():
            continue
        header = json.loads(trace.read_text(encoding="utf-8").splitlines()[0]).get("header") or {}
        runs.append((run_id, header.get("prompt", ""), OpStack.from_json(stack_file.read_text(encoding="utf-8"))))
    return runs[:limit]


def summarize(rows: list[dict]) -> dict:
    fixes = [r for r in rows if r["tool"] != "none"]
    kinds = Counter()
    for r in fixes:
        for p in r["problems"]:
            kinds["unknown tool" if p.startswith("unknown tool") else "no such op id" if "no op with id" in p
                  else "no such parameter" if "has no parameter" in p else "invalid value"] += 1
    return {"issues": len(rows), "tool_none": len(rows) - len(fixes),
            "fixes_valid": sum(not r["problems"] for r in fixes), "fixes": len(fixes), "problem_kinds": dict(kinds),
            "missing": Counter(r["fix_hint"].strip()[:90] for r in rows if r["tool"] == "none").most_common()}


def main() -> None:
    from agent.models import ModelConfig, make_adapter
    from agent.run import ENV_FILE, make_renderer

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--effort", default=None, help="default: the provider's default effort")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--arms", nargs="+", choices=["without", "with"], default=["without", "with"])
    args = ap.parse_args()
    load_dotenv(ENV_FILE, override=False)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    tracer = Tracer()
    tracer.run_id = f"catalogue_ab-{tracer.run_id}"
    tracer.path = tracer.path.with_name(f"{tracer.run_id}.jsonl")
    render = make_renderer("l2", Path("renders") / tracer.run_id)
    arms = {"without": Critic(make_adapter(ModelConfig(model=args.model, effort=args.effort)), catalogue=False),
            "with": Critic(make_adapter(ModelConfig(model=args.model, effort=args.effort)), catalogue=True)}
    arms = {arm: critic for arm, critic in arms.items() if arm in args.arms}
    rows: dict[str, list[dict]] = {arm: [] for arm in arms}
    cost, unpriced, step = 0.0, 0, 0
    for run_id, prompt, stack in saved_runs(args.limit):
        session = Session(stack)
        renders = render(stack.state, f"{len(stack)} ops")
        for arm, critic in arms.items():
            for attempt in (1, 2):
                try:
                    critique, response = critic.critique({"prompt": prompt}, session, renders=renders)
                    break
                except CriticError as e:
                    print(f"{run_id} {arm}: critic failed (try {attempt}): {e}")
                    critique, response = None, e.response
            if response is not None:
                call_cost = response.usage.cost_usd(response.model)
                cost, unpriced = cost + (call_cost or 0.0), unpriced + (call_cost is None)
            if critique is None:
                continue
            for i in critique.issues:
                row = {"run": run_id, "tool": i.tool, "op_id": i.op_id, "location": i.location, "changes": [c.model_dump() for c in i.changes],
                       "fix_hint": i.fix_hint, "problems": hint_problems(i, stack)}
                rows[arm].append(row)
            step += 1
            tracer.log(TraceRecord(run_id=tracer.run_id, stage=f"catalogue_ab/{arm}", step=step,
                                   model=response.model if response else None,
                                   cost_usd=response.usage.cost_usd(response.model) if response else None,
                                   scores={"source_run": run_id, "issues": rows[arm][-len(critique.issues):] if critique.issues else []}))
            print(f"{run_id} {arm}: {len(critique.issues)} issues")
    for arm, r in rows.items():
        s = summarize(r)
        print(f"\n== {arm} catalogue: {s['issues']} issues; tool 'none' {s['tool_none']}; "
              f"fixes valid as written {s['fixes_valid']}/{s['fixes']}; problems {s['problem_kinds']}")
        for hint, n in s["missing"]:
            print(f"   missing tool ({n}x): {hint}")
    cost_text = f"unknown ({unpriced} unpriced calls; ${cost:.3f} priced)" if unpriced else f"${cost:.3f}"
    print(f"\ncost {cost_text}; trace {tracer.path}")


if __name__ == "__main__":
    main()
