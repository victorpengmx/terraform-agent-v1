"""Run an evaluation (DESIGN §16): every (prompt, seed, condition) of a config, in-process, graded and judged.

    python -m eval.run_eval --config eval/configs/e1_critic.yaml [--suite eval/benchmark.yaml]
                            [--results-id ID] [--dry-run] [--yes]

Prints the cost estimate first (pending runs x their per-run budget cap, plus judge calls) and spends nothing
without --yes. Results go to eval/results/<id>/ (default id: the config name, so re-running resumes):
runs.jsonl (one row per run; the last row per key wins), config.yaml, and per run <key>/ with its trace,
final stack and renders. Finished runs are skipped; runs that ended in an error are retried.

The agent never sees the answer key: it writes its own SPEC (or none, with use_spec false). The answer key
only grades the final stack (eval/scoring.py), and the judge (eval/judge.py) sees the request and renders.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agent.models import ModelAdapter, ModelConfig, make_adapter, price_known
from agent.run import ENV_FILE, RunConfig, execute, git_state
from agent.tracing import Tracer
from eval import judge as judging
from eval.benchmark import SUITE, BenchmarkEntry, load
from eval.config import EvalConfig, load_config, same_family
from eval.rows import append_row, read_rows, trace_stats
from eval.scoring import answer_key, checks, success

RESULTS = Path("eval/results")
MakeModel = Callable[[ModelConfig], ModelAdapter]


@dataclass(frozen=True)
class PlannedRun:
    key: str
    entry: BenchmarkEntry
    condition: str
    seed: int


def plan(config: EvalConfig, entries: list[BenchmarkEntry]) -> list[PlannedRun]:
    """prompts x seeds x conditions, in benchmark, seed, condition order."""
    return [PlannedRun(f"{e.id}-{c}-s{s}", e, c, s) for e in config.select(entries) for s in config.seeds for c in config.conditions]


def pending(runs: list[PlannedRun], rows: dict[str, dict[str, Any]]) -> list[PlannedRun]:
    return [r for r in runs if rows.get(r.key, {}).get("status") != "done"]


def run_config(config: EvalConfig, run: PlannedRun, suite_dir: Path) -> RunConfig:
    cond = config.conditions[run.condition]
    return RunConfig(prompt=run.entry.prompt or "", images=tuple(suite_dir / r for r in run.entry.references),
                     model=config.models.builder, effort=config.effort, critic=cond.critic,
                     critic_model=config.models.critic, critic_max_tokens=config.critic_max_tokens,
                     rounds=cond.rounds if cond.critic else 1, size=tuple(config.size),
                     base_y=config.base_y, seed=run.seed, budget=config.budget_for(run.condition), render=config.render,
                     use_spec=cond.use_spec, compare_both_orders=config.compare_both_orders)


def condition_roles(config: EvalConfig, condition: str) -> dict[str, str]:
    """The model each role uses in a condition: builder (and SPEC), critic and comparison when the critic is on."""
    cond = config.conditions[condition]
    roles = {"builder": config.models.builder}
    if cond.use_spec:
        roles["spec"] = config.models.builder
    if cond.critic:
        roles |= {"critic": config.models.critic, "comparison": config.models.critic}
    return roles


def estimate(config: EvalConfig, runs: list[PlannedRun]) -> tuple[str, float | None]:
    """The cost gate's text and upper bound: each pending run's budget cap plus one judge call per run.

    A condition with a model that has no price is "unknown", never $0: the budget cap only counts priced
    calls, so it can't bound that condition's spend."""
    lines, known, unknown = [], 0.0, False
    for cond in config.conditions:
        n = sum(r.condition == cond for r in runs)
        if not n:
            continue
        cap = config.budget_for(cond).max_cost_usd
        unpriced = [f"{role} {m}" for role, m in condition_roles(config, cond).items() if not price_known(m)]
        if unpriced:
            lines.append(f"  {cond}: {n} runs x ${cap:.2f} cap = unknown (no price for {', '.join(unpriced)}; "
                         "the cap only counts priced calls)")
            unknown = True
        else:
            lines.append(f"  {cond}: {n} runs x ${cap:.2f} cap = ${n * cap:.2f}")
            known += n * cap
    judge_model = config.judge_model
    per_call = [judging.call_estimate(judge_model, len(r.entry.references)) for r in runs]
    if any(c is None for c in per_call):
        lines.append(f"  judge: {len(runs)} calls on {judge_model} = unknown (no price for judge {judge_model})")
        unknown = True
    elif runs:
        judge_total = sum(per_call)  # type: ignore[arg-type]
        lines.append(f"  judge: {len(runs)} calls on {judge_model} = ${judge_total:.2f} (estimate; cached verdicts are free)")
        known += judge_total
    upper = None if unknown else known
    total = f"unknown (priced part up to ${known:.2f})" if unknown else f"${known:.2f}"
    return "\n".join([f"{len(runs)} pending runs; estimated cost up to {total}", *lines]), upper


def run_one(run: PlannedRun, config: EvalConfig, results: Path, suite_dir: Path, make_model: MakeModel,
            judge_model: ModelAdapter | None, cache: Path | None = judging.CACHE,
            render: judging.Renderer = judging.render_views) -> dict[str, Any]:
    """Build, grade and judge one run; returns its row (status done or error)."""
    run_dir = results / run.key
    if run_dir.exists():
        shutil.rmtree(run_dir)  # a retried run starts clean (traces append)
    cfg = run_config(config, run, suite_dir)
    tracer = Tracer(run_id=f"{results.name}-{run.key}", root=run_dir)
    row: dict[str, Any] = {
        "key": run.key, "prompt_id": run.entry.id, "kind": run.entry.kind, "condition": run.condition, "seed": run.seed,
        "config": config.name, "git": git_state(),
        "models": {"builder": cfg.model, "critic": cfg.critic_model if cfg.critic else None,
                   "judge": judge_model.config.model if judge_model else config.judge_model,
                   "judge_same_family": same_family(config.judge_model, cfg.model, cfg.critic_model if cfg.critic else None)},
        "settings": {"critic": cfg.critic, "critic_max_tokens": cfg.critic_max_tokens if cfg.critic else None, "rounds": cfg.rounds, "use_spec": cfg.use_spec, "render": cfg.render,
                     "compare_both_orders": cfg.compare_both_orders, "budget": cfg.budget.__dict__},
        "trace": tracer.path.as_posix(), "stack": (run_dir / "stack.json").as_posix(),
    }
    started = time.perf_counter()
    try:
        result = execute(cfg, tracer, run_dir, make_model=make_model)
    except Exception as e:  # noqa: BLE001 - recorded and retried on the next invocation
        row |= {"status": "error", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(),
                "latency_s": round(time.perf_counter() - started, 1)}
        if tracer.path.exists():
            row["trace_stats"] = trace_stats(tracer.path)
        return row
    row |= {"latency_s": round(time.perf_counter() - started, 1), "stop": result.stop, "rounds": result.rounds,
            "accepted_round": result.accepted_round, "cost_usd": result.cost_usd, "cost_text": result.cost_text,
            "spec_graph": None if result.graph is None else result.graph.model_dump(mode="json"),
            "trace_stats": trace_stats(tracer.path)}
    stack = result.session.stack
    row |= {"answer_key": answer_key(run.entry.graph, stack), "checks": checks(stack)}
    row["judge"] = judge_run(judge_model, stack, run.entry, suite_dir, cache, render, run_dir)
    row["success"] = success(row["answer_key"], row["checks"], row["judge"])
    row["status"] = "done"
    return row


def judge_run(model: ModelAdapter | None, stack, entry: BenchmarkEntry, suite_dir: Path, cache: Path | None,
              render: judging.Renderer, run_dir: Path | None) -> dict[str, Any]:
    if model is None:
        return {"failed": "no judge", "passed": None}
    refs = [(suite_dir / r).read_bytes() for r in entry.references]
    try:
        return judging.judge(model, stack, entry.prompt, refs, cache=cache, render=render, save_renders=run_dir)
    except Exception as e:  # noqa: BLE001 - the run is still graded; the judge can be redone by rescore --judge
        return {"model": model.config.model, "failed": f"{type(e).__name__}: {e}", "passed": None}


def evaluate(config: EvalConfig, entries: list[BenchmarkEntry], results: Path, suite_dir: Path,
             make_model: MakeModel = make_adapter, cache: Path | None = judging.CACHE,
             render: judging.Renderer = judging.render_views, log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    """Run every pending run of `config` and append its row; returns the new rows."""
    results.mkdir(parents=True, exist_ok=True)
    rows_path = results / "runs.jsonl"
    todo = pending(plan(config, entries), read_rows(rows_path))
    judge_model = make_model(ModelConfig(model=config.judge_model, effort=config.judge_effort)) if todo else None
    new = []
    for k, run in enumerate(todo, 1):
        log(f"[{k}/{len(todo)}] {run.key}")
        row = run_one(run, config, results, suite_dir, make_model, judge_model, cache, render)
        append_row(rows_path, row)
        new.append(row)
        s = row.get("success") or {}
        log(f"  {row['status']}: success {s.get('overall')}, checklist "
            f"{(row.get('answer_key') or {}).get('passed')}/{(row.get('answer_key') or {}).get('total')}, {row.get('cost_text', row.get('error'))}")
    return new


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--suite", type=Path, default=SUITE)
    ap.add_argument("--results-id", default=None, help="results folder name (default: the config name)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and cost estimate only")
    ap.add_argument("--yes", action="store_true", help="spend: run the pending runs")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    config = load_config(args.config)
    entries = load(args.suite)
    results = RESULTS / (args.results_id or config.name)
    runs = plan(config, entries)
    todo = pending(runs, read_rows(results / "runs.jsonl"))
    print(f"{config.name}: {len(runs)} runs planned, {len(runs) - len(todo)} finished, results in {results}")
    text, _ = estimate(config, todo)
    print(text)
    if args.dry_run or not todo:
        return
    if not args.yes:
        sys.exit("not run: add --yes to spend this")
    from dotenv import load_dotenv

    load_dotenv(ENV_FILE, override=False)
    results.mkdir(parents=True, exist_ok=True)
    (results / "config.yaml").write_text(args.config.read_text(encoding="utf-8"), encoding="utf-8")
    evaluate(config, entries, results, args.suite.parent)


if __name__ == "__main__":
    main()
