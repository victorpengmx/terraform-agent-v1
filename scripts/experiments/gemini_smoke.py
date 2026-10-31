"""Live smoke check of the Gemini adapter (paid; never run by pytest).

    python -m scripts.experiments.gemini_smoke --model gemini-3.8-flash [--critic-model ...] [--render l1|l2|none]

One SPEC call, one builder round with one critique (`run_task` with max_rounds=1), then one blind comparison of
the result against the untouched flat canvas (the built terrain should win). Needs the gemini extra and
GEMINI_API_KEY in .env. Prints every model step (stop reason, tokens, cost or "unknown") and the trace path
(traces/gemini_smoke-<run_id>.jsonl); renders and the final stack go to renders/<run_id>/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from agent.budget import Budget
from agent.critic import AttemptView, Critic
from agent.harness import run_task
from agent.models import ModelConfig, make_adapter
from agent.run import ENV_FILE, EFFORTS, make_renderer, save_final
from agent.session import Session, terrain_stats
from agent.tools import control_tools, observe_tools, op_tools
from agent.tracing import Tracer, TraceRecord


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="builder and SPEC model, e.g. a gemini-... id")
    ap.add_argument("--critic-model", default=None, help="critic and comparison model (default: --model)")
    ap.add_argument("--effort", default=None, choices=EFFORTS, help="default: the provider's default effort")
    ap.add_argument("--prompt", default="a lone volcanic peak with a crater lake")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--render", choices=["l1", "l2", "none"], default="l1")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--max-cost", type=float, default=1.0, help="USD; only enforced for priced models")
    ap.add_argument("--max-seconds", type=float, default=1800.0,
                    help="wall-clock budget for SPEC, build and critique (the first run hit the 900 s default before "
                         "the critique: Gemini calls averaged 55 s)")
    args = ap.parse_args()
    load_dotenv(ENV_FILE, override=False)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    tracer = Tracer()
    tracer.run_id = f"gemini_smoke-{tracer.run_id}"
    tracer.path = tracer.path.with_name(f"{tracer.run_id}.jsonl")
    out_dir = Path("renders") / tracer.run_id
    builder = make_adapter(ModelConfig(model=args.model, effort=args.effort))
    critic = Critic(make_adapter(ModelConfig(model=args.critic_model or args.model, effort=args.effort)))
    session = Session.flat(args.size, args.size, base_y=64, seed=1)
    observe = [] if args.render == "none" else observe_tools(session, out_dir)
    tools = {t.name: t for t in [*op_tools(session), *control_tools(session), *observe]}
    render = None if args.render == "none" else make_renderer(args.render, out_dir)
    models = {role: {"provider": a.config.provider, "model": a.config.model}
              for role, a in (("builder", builder), ("spec", builder), ("critic", critic.model), ("comparison", critic.model))}
    tracer.log(TraceRecord(run_id=tracer.run_id, stage="run", step=0, note="gemini smoke header",
                           header={"prompt": args.prompt, "models": models, "effort": args.effort, "render": args.render,
                                   "canvas": {"width": args.size, "depth": args.size, "base_y": 64, "seed": 1},
                                   "budget": {"max_steps": args.max_steps, "max_cost_usd": args.max_cost,
                                              "max_seconds": args.max_seconds}}))
    print(f"run {tracer.run_id}: {json.dumps(models)}")

    try:
        result = run_task(prompt=args.prompt, images=[], session=session, tools=tools, model=builder, tracer=tracer,
                          budget=Budget(max_steps=args.max_steps, max_cost_usd=args.max_cost, max_seconds=args.max_seconds), critic=critic,
                          render=render, max_rounds=1)
        print(result)

        spec = {"prompt": args.prompt, "graph": result.graph}
        built = AttemptView(render(session.state, None) if render else [], terrain_stats(session.state))
        flat = Session.flat(args.size, args.size, base_y=64, seed=1)
        empty = AttemptView(render(flat.state, None) if render else [], terrain_stats(flat.state))
        out, response = critic.compare(spec, built, empty)
        u = response.usage
        tracer.log(TraceRecord(run_id=tracer.run_id, stage="smoke/compare", step=10_000, model=response.model,
                               request_id=response.request_id, stop_reason=response.stop_reason,
                               usage={"input": u.input_tokens, "output": u.output_tokens, "cache_read": u.cache_read_tokens,
                                      "cache_write": u.cache_write_tokens},
                               cost_usd=u.cost_usd(response.model), latency_s=round(response.latency_s, 3),
                               retries=response.retries or None,
                               scores={"order": {"1": "built", "2": "flat"}, **out.model_dump()}))
        print(f"blind comparison (1 = built, 2 = flat): better={out.better} "
              f"({'as expected' if out.better == '1' else 'UNEXPECTED'}): {out.why}")
    finally:
        save_final(session, out_dir, l2_views=("overview_sw",) if args.render == "l2" else ())
        print_steps(tracer.path)
        print(f"trace {tracer.path}")


def print_steps(path: Path) -> None:
    """One line per model step in the trace: stage, stop reason, tokens, cost (or unknown), latency and the
    adapter's retries (error codes and backoff seconds, which latency includes)."""
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("usage") is None:
            continue
        cost = "unknown" if r["cost_usd"] is None else f"${r['cost_usd']:.4f}"
        u = r["usage"]
        retries = r.get("retries") or []
        waits = f" retries={[x['error'] for x in retries]} waited={sum(x['wait_s'] for x in retries):.1f}s" if retries else ""
        print(f"  {r['stage']:<20} {r['model']:<24} stop={r['stop_reason']:<10} in={u['input']} out={u['output']} "
              f"cache_read={u['cache_read']} cost={cost} {r['latency_s']}s{waits}")


if __name__ == "__main__":
    main()
