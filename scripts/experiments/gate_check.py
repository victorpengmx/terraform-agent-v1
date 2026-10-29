"""Broken-copy test for the blind comparison (DESIGN §11.4 validation).

    python -m scripts.experiments.gate_check [--model claude-haiku-5-5] [--render l1|l2]

Intact terrain: the valley demo. Broken copy: the same stack with its water removed and a heavy, fine,
uniform ridged roughen on top (spikes, walls, no river or lake). The comparison runs in both orders and
passes only if the intact terrain wins both times. Records go to traces/gate_check-<run_id>.jsonl.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from agent.critic import AttemptView, Critic
from agent.session import terrain_stats
from agent.tracing import Tracer, TraceRecord
from terrain.opstack import OpStack

SPEC = {"prompt": "a valley between two rugged mountains, with a lake in the valley and a river flowing through it"}


def broken_copy(intact: OpStack) -> OpStack:
    broken = OpStack.from_json(intact.to_json())
    for e in reversed([e for e in broken.entries if e.op in ("carve_river", "fill_lake")]):
        broken.remove(e.id)  # rivers first: one ends in the lake
    broken.add("roughen", {"amount": 30, "scale": 0.02, "style": "ridged", "weight": "uniform"}, id="wreck")
    return broken


def check_pair(critic: Critic, spec: dict, intact: AttemptView, broken: AttemptView, tracer: Tracer | None = None) -> tuple[bool, list[dict]]:
    """Compare in both orders; passes only if the intact terrain wins both. Returns (passed, one row per order)."""
    rows = []
    for intact_first in (True, False):
        order = {"1": "intact", "2": "broken"} if intact_first else {"1": "broken", "2": "intact"}
        first, second = (intact, broken) if intact_first else (broken, intact)
        out, response = critic.compare(spec, first, second)
        winner = "same" if out.better == "same" else order[out.better]
        row = {"order": order, "winner": winner, **out.model_dump()}
        rows.append(row)
        if tracer is not None:
            tracer.log(TraceRecord(run_id=tracer.run_id, stage="gate_check/compare", step=len(rows), model=response.model,
                                   request_id=response.request_id, cost_usd=response.usage.cost_usd(response.model),
                                   scores=row, note=f"winner: {winner}"))
    return all(r["winner"] == "intact" for r in rows), rows


def main() -> None:
    from agent.models import AnthropicAdapter, ModelConfig
    from agent.run import ENV_FILE, make_renderer
    from scripts.demos import build

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--render", choices=["l1", "l2"], default="l2")  # same default as agent.run
    args = ap.parse_args()
    load_dotenv(ENV_FILE, override=False)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    tracer = Tracer()
    tracer.run_id = f"gate_check-{tracer.run_id}"
    tracer.path = tracer.path.with_name(f"{tracer.run_id}.jsonl")
    render = make_renderer(args.render, Path("renders") / tracer.run_id)
    intact = build("valley")
    broken = broken_copy(intact)
    views = [AttemptView(render(s.state, None), terrain_stats(s.state)) for s in (intact, broken)]
    critic = Critic(AnthropicAdapter(ModelConfig(model=args.model, effort=args.effort)))
    passed, rows = check_pair(critic, SPEC, *views, tracer=tracer)
    for r in rows:
        print(f"Terrain 1 = {r['order']['1']}: winner {r['winner']} ({r['better']}). why: {r['why']}")
    print(f"{'PASS' if passed else 'FAIL'}: trace {tracer.path}")


if __name__ == "__main__":
    main()
