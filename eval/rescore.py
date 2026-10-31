"""Re-score finished runs from their saved stacks, without builder calls (DESIGN §16).

    python -m eval.rescore eval/results/<id> [--suite eval/benchmark.yaml] [--judge] [--judge-model ID] [--yes]

Recomputes the answer-key checklist, the programmatic checks and success for every finished row, for when a
check, the matcher or the answer key changes. With --judge it also re-judges: cached verdicts are free, and
the cost of the uncached ones is printed first and needs --yes. runs.jsonl is rewritten atomically (the
previous version is kept as runs.jsonl.bak); each row records when and with which code it was scored.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from agent.models import ModelAdapter, ModelConfig, make_adapter
from agent.run import ENV_FILE, git_state
from eval import judge as judging
from eval.benchmark import SUITE, BenchmarkEntry, load
from eval.config import default_judge
from eval.rows import read_rows, write_rows
from eval.run_eval import judge_run
from eval.scoring import answer_key, checks, success
from terrain.opstack import OpStack


def load_stack(row: dict[str, Any]) -> OpStack:
    return OpStack.from_json(Path(row["stack"]).read_text(encoding="utf-8"))


def uncached(rows: list[dict[str, Any]], entries: dict[str, BenchmarkEntry], suite_dir: Path, model: str,
             cache: Path | None) -> list[dict[str, Any]]:
    """Finished rows whose judge verdict would need a new (paid) call."""
    out = []
    for row in rows:
        e = entries[row["prompt_id"]]
        refs = [(suite_dir / r).read_bytes() for r in e.references]
        key = judging.cache_key(load_stack(row), e.prompt, refs, model)
        if cache is None or not (cache / f"{key}.json").exists():
            out.append(row)
    return out


def rescore(rows: dict[str, dict[str, Any]], entries: dict[str, BenchmarkEntry], suite_dir: Path,
            judge_model: ModelAdapter | None = None, cache: Path | None = judging.CACHE,
            render: judging.Renderer = judging.render_views) -> list[dict[str, Any]]:
    out = []
    for row in rows.values():
        if row.get("status") != "done":
            out.append(row)
            continue
        e = entries[row["prompt_id"]]
        stack = load_stack(row)
        row = row | {"answer_key": answer_key(e.graph, stack), "checks": checks(stack)}
        if judge_model is not None:
            row["judge"] = judge_run(judge_model, stack, e, suite_dir, cache, render, Path(row["stack"]).parent)
            row["models"] = row["models"] | {"judge": judge_model.config.model}
        row["success"] = success(row["answer_key"], row["checks"], row.get("judge"))
        refs = [(suite_dir / r).read_bytes() for r in e.references]
        row["scored"] = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_state(),
                         "judge_prompt": judging.prompt_hash(e.prompt, refs, judging.views_for(bool(refs)))}
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("--suite", type=Path, default=SUITE)
    ap.add_argument("--judge", action="store_true", help="also re-judge (cached verdicts are free)")
    ap.add_argument("--judge-model", default=None, help="default: the model the rows were judged with, else the default judge")
    ap.add_argument("--yes", action="store_true", help="spend: make the uncached judge calls")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = read_rows(args.results / "runs.jsonl")
    if not rows:
        sys.exit(f"no rows in {args.results / 'runs.jsonl'}")
    entries = {e.id: e for e in load(args.suite)}
    judge_model = None
    if args.judge:
        model = args.judge_model or next((r["models"]["judge"] for r in rows.values() if r.get("models", {}).get("judge")),
                                         default_judge())
        done = [r for r in rows.values() if r.get("status") == "done"]
        todo = uncached(done, entries, args.suite.parent, model, judging.CACHE)
        per_call = [judging.call_estimate(model, len(entries[r["prompt_id"]].references)) for r in todo]
        cost = "unknown" if any(c is None for c in per_call) else f"${sum(per_call):.2f}"  # type: ignore[arg-type]
        print(f"judge {model}: {len(done) - len(todo)} cached, {len(todo)} new calls, estimated {cost}")
        if todo and not args.yes:
            sys.exit("not re-judged: add --yes to spend this (or drop --judge)")
        from dotenv import load_dotenv

        load_dotenv(ENV_FILE, override=False)
        judge_model = make_adapter(ModelConfig(model=model, effort="low"))
    new = rescore(rows, entries, args.suite.parent, judge_model)
    write_rows(args.results / "runs.jsonl", new)
    changed = sum((a.get("success") or {}).get("overall") != (b.get("success") or {}).get("overall")
                  for a, b in zip(rows.values(), new))
    print(f"rescored {sum(r.get('status') == 'done' for r in new)} runs; overall success changed on {changed}")


if __name__ == "__main__":
    main()
