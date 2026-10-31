"""Evaluation report (DESIGN §16): results folders in, eval/reports/<date>-<name>.md and a runs CSV out.

    python -m eval.report eval/results/e1_critic eval/results/e2_spec --name m6 [--calibration NAME]

- E1 critic_on vs critic_off and E2 critic_on vs spec_off: success rate and mean answer-key pass rate per
  condition (all, text, image); paired win/loss/tie per (prompt, seed); exact sign test; bootstrap 95% CI of
  the paired difference (prompts resampled); a per-prompt table. "No clear difference at this N" whenever the
  CI includes 0.
- E4 per condition: cost and tokens per successful build, p50/p95 latency, steps, tool-error rate, critic
  fix validity, rollback rate, blind-comparison revision and Terrain 1 win rates.
- E5 SPEC quality; E3 judge calibration when `--calibration` names a scored calibration set.
- Limitations computed from the rows.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from eval import spec_quality, stats
from eval.benchmark import SUITE, load
from eval.config import shared_family
from eval.rows import read_rows

REPORTS = Path("eval/reports")
EXPERIMENTS = [("E1", "Critic on vs off", "critic_on", "critic_off"), ("E2", "SPEC on vs off", "critic_on", "spec_off")]
KINDS = (None, "text", "image")
MIN_PROMPTS = 5  # fewer prompts: the bootstrap CI collapses onto the few values there are, so no verdict


def load_rows(folders: Iterable[Path]) -> list[dict[str, Any]]:
    """Latest row per key from each folder, in folder order."""
    out = []
    for folder in folders:
        for row in read_rows(Path(folder) / "runs.jsonl").values():
            out.append(row | {"folder": Path(folder).name})
    return out


def done(rows: list[dict[str, Any]], condition: str | None = None, kind: str | None = None) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("status") == "done" and condition in (None, r["condition"]) and kind in (None, r["kind"])]


def overall(r: dict[str, Any]) -> bool | None:
    return (r.get("success") or {}).get("overall")


def judge_paid(r: dict[str, Any]) -> float | None:
    """What the run's verdict cost when it was made (a rescore served from the cache spends $0 but keeps this)."""
    j = r.get("judge") or {}
    return j.get("paid_usd", j.get("cost_usd"))


def pass_rate(r: dict[str, Any]) -> float | None:
    return (r.get("answer_key") or {}).get("pass_rate")


def pairs(rows: list[dict[str, Any]], a: str, b: str, kind: str | None = None) -> list[tuple[dict, dict]]:
    """(row a, row b) per (prompt, seed) present in both conditions; the first folder wins on duplicates."""
    def index(cond: str) -> dict[tuple[str, int], dict]:
        out: dict[tuple[str, int], dict] = {}
        for r in done(rows, cond, kind):
            out.setdefault((r["prompt_id"], r["seed"]), r)
        return out

    ia, ib = index(a), index(b)
    return [(ia[k], ib[k]) for k in sorted(ia.keys() & ib.keys())]


def compare(ps: list[tuple[dict, dict]], metric: str) -> dict[str, Any]:
    """Paired comparison of a minus b on `success` (pairs with an unknown side dropped) or `pass_rate`."""
    get = overall if metric == "success" else pass_rate
    wins = losses = ties = unknown = 0
    diffs: dict[str, list[float]] = defaultdict(list)
    for ra, rb in ps:
        va, vb = get(ra), get(rb)
        if va is None or vb is None:
            unknown += 1
            continue
        d = float(va) - float(vb)
        diffs[ra["prompt_id"]].append(d)
        wins += d > 0
        losses += d < 0
        ties += d == 0
    ci = stats.bootstrap_ci(diffs)
    return {"pairs": len(ps), "wins": wins, "losses": losses, "ties": ties, "unknown": unknown,
            "p": stats.sign_test(wins, losses), "ci": ci, "prompts": len(diffs)}


def condition_summary(rows: list[dict[str, Any]], cond: str, kind: str | None) -> dict[str, Any]:
    rs = done(rows, cond, kind)
    known = [overall(r) for r in rs if overall(r) is not None]
    rates = [pass_rate(r) for r in rs if pass_rate(r) is not None]
    return {"n": len(rs), "success": mean(known) if known else None, "unknown": len(rs) - len(known),
            "pass_rate": mean(rates) if rates else None}


def efficiency(rows: list[dict[str, Any]], cond: str) -> dict[str, Any]:
    """E4 for one condition."""
    rs = done(rows, cond)
    successes = sum(overall(r) is True for r in rs)
    costs = [r.get("cost_usd") for r in rs]
    cost_known = all(c is not None for c in costs)
    judge_costs = [judge_paid(r) or 0.0 for r in rs]
    ts = [r.get("trace_stats") or {} for r in rs]
    tokens = sum(sum(v.values()) for t in ts for v in (t.get("tokens") or {}).values())
    tool_calls = sum(t.get("tool_calls", 0) for t in ts)
    fixes = Counter()
    for t in ts:
        fixes.update(t.get("fixes") or {})
    decisions = [g for t in ts for g in (t.get("gate") or []) if g["round"] > 1]  # round 1 is the baseline
    comps = [c for t in ts for c in (t.get("comparisons") or [])]
    decisive = [c for c in comps if c.get("better") in ("1", "2")]
    latency = [r["latency_s"] for r in rs if r.get("latency_s") is not None]

    def rate(a: float, b: float) -> float | None:
        return a / b if b else None

    return {
        "n": len(rs), "successes": successes,
        "cost": sum(costs) if cost_known else None, "judge_cost": sum(judge_costs),
        "cost_per_success": (sum(costs) / successes if successes else None) if cost_known else None,
        "tokens_per_success": tokens / successes if successes else None,
        "p50_latency": stats.percentile(latency, 50), "p95_latency": stats.percentile(latency, 95),
        "steps": mean([t.get("steps", 0) for t in ts]) if ts else None,
        "tool_error_rate": rate(sum(t.get("tool_errors", 0) for t in ts), tool_calls),
        "fix_validity": rate(fixes["valid"], fixes["with_fix"]),
        "rollback_rate": rate(sum(g["decision"] == "rolled_back" for g in decisions), len(decisions)),
        "comparisons": len(comps),
        "revision_win_rate": rate(sum(c.get("winner") == "revision" for c in comps), len(comps)),
        "terrain1_win_rate": rate(sum(c.get("better") == "1" for c in decisive), len(decisive)),
        "stops": dict(Counter(r.get("stop", "?").split(":")[0] for r in rs)),
    }


# ---------------------------------------------------------------- formatting


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def num(x: float | None, fmt: str = "{:.2f}") -> str:
    return "n/a" if x is None else fmt.format(x)


def money(x: float | None) -> str:
    """Sub-dollar amounts get 3 decimals: a pilot's judge costs $0.003, not $0.00."""
    return "unknown" if x is None else f"${x:.2f}" if abs(x) >= 1 else f"${x:.3f}"


def ci_text(ci: tuple[float, float, float] | None, as_pct: bool) -> str:
    if ci is None:
        return "n/a"
    f = (lambda v: f"{v * 100:+.0f} pts") if as_pct else (lambda v: f"{v:+.3f}")
    return f"{f(ci[0])} [{f(ci[1])}, {f(ci[2])}]"


def table(head: list[str], body: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head), *("| " + " | ".join(r) + " |" for r in body)]


def experiment_section(rows: list[dict[str, Any]], code: str, title: str, a: str, b: str) -> list[str]:
    out = [f"## {code}: {title}", ""]
    if not pairs(rows, a, b):
        return out + [f"No paired runs of `{a}` and `{b}` in these results.", ""]
    prompts = {r["prompt_id"] for ra, rb in pairs(rows, a, b) for r in (ra, rb)}
    rs = [r for r in rows if r["prompt_id"] in prompts]  # E2: only the spec_ablation prompts
    body = []
    for kind in KINDS:
        for cond in (a, b):
            s = condition_summary(rs, cond, kind)
            body.append([kind or "all", f"`{cond}`", str(s["n"]), pct(s["success"]), str(s["unknown"]), pct(s["pass_rate"])])
    out += table(["prompts", "condition", "runs", "success", "judge unknown", "mean checklist pass rate"], body) + [""]
    body = []
    for kind in KINDS:
        ps = pairs(rs, a, b, kind)
        for metric in ("success", "pass_rate"):
            c = compare(ps, metric)
            verdict = (stats.label(c["ci"], a, b) if c["prompts"] >= MIN_PROMPTS
                       else f"too few prompts to tell ({c['prompts']})")
            body.append([kind or "all", metric.replace("_", " "), f"{c['wins']} / {c['losses']} / {c['ties']}",
                         str(c["unknown"]), num(c["p"], "{:.3f}"), ci_text(c["ci"], True), verdict])
    out += [f"Paired by (prompt, seed): `{a}` minus `{b}`. Wins/losses/ties count pairs where `{a}` is better/worse/equal; "
            "p is the exact two-sided sign test (ties dropped); the CI resamples prompts (10 000 bootstrap samples). "
            f"No verdict below {MIN_PROMPTS} prompts.", ""]
    out += table(["prompts", "metric", "W / L / T", "unknown", "sign-test p", "difference [95% CI]", "verdict"], body) + [""]
    out += ["Per prompt (success per seed: ✓ pass, ✗ fail, ? judge unknown; pass rate is the mean over seeds):", ""]
    seeds = sorted({r["seed"] for r in done(rs)})
    body = []
    for pid in sorted(prompts):
        cells = [pid, next(r["kind"] for r in rs if r["prompt_id"] == pid)]
        for cond in (a, b):
            mine = {r["seed"]: r for r in done(rs, cond) if r["prompt_id"] == pid}
            marks = "".join({True: "✓", False: "✗", None: "?"}[overall(mine[s])] if s in mine else "·" for s in seeds)
            rates = [pass_rate(r) for r in mine.values() if pass_rate(r) is not None]
            cells += [marks, pct(mean(rates)) if rates else "n/a"]
        body.append(cells)
    out += table(["prompt", "kind", f"`{a}` success", f"`{a}` pass rate", f"`{b}` success", f"`{b}` pass rate"], body) + [""]
    return out


def e4_section(rows: list[dict[str, Any]]) -> list[str]:
    conds = sorted({r["condition"] for r in done(rows)})
    effs = {c: efficiency(rows, c) for c in conds}
    metrics = [
        ("runs / successes", lambda e: f"{e['n']} / {e['successes']}"),
        ("agent cost (builder, SPEC, critic)", lambda e: money(e["cost"])),
        ("judge cost", lambda e: money(e["judge_cost"])),
        ("cost per successful build", lambda e: "n/a (0 successes)" if not e["successes"] else money(e["cost_per_success"])),
        ("tokens per successful build", lambda e: "n/a (0 successes)" if not e["successes"] else num(e["tokens_per_success"], "{:,.0f}")),
        ("latency p50 / p95 (s)", lambda e: f"{num(e['p50_latency'], '{:.0f}')} / {num(e['p95_latency'], '{:.0f}')}"),
        ("model calls per run", lambda e: num(e["steps"], "{:.1f}")),
        ("tool-error rate", lambda e: pct(e["tool_error_rate"])),
        ("critic fix validity", lambda e: pct(e["fix_validity"])),
        ("rollback rate (revisions)", lambda e: pct(e["rollback_rate"])),
        ("comparisons", lambda e: str(e["comparisons"])),
        ("revision win rate", lambda e: pct(e["revision_win_rate"])),
        ("Terrain 1 win rate (decisive)", lambda e: pct(e["terrain1_win_rate"])),
        ("stop reasons", lambda e: ", ".join(f"{k} {v}" for k, v in sorted(e["stops"].items()))),
    ]
    out = ["## E4: Cost and efficiency", ""]
    out += table(["metric", *(f"`{c}`" for c in conds)], [[m, *(f(effs[c]) for c in conds)] for m, f in metrics])
    out += ["", "Tokens include cache reads and writes. Cost per successful build divides all of a condition's agent cost "
            "by its successes. The revision win rate should stay clearly below 100% and Terrain 1 near 50% (DESIGN §11.4).", ""]
    return out


def e5_section(rows: list[dict[str, Any]], keys: dict) -> list[str]:
    s = spec_quality.for_rows(rows, keys)
    out = ["## E5: SPEC quality (agent graph vs answer key)", ""]
    if not s["all"]["runs"]:
        return out + ["No runs with an agent graph.", ""]
    body = []
    for kind in ("all", "text", "image"):
        q = s[kind]
        cats = "; ".join(f"{a} {pct(c['exact'])} exact / {pct(c['within_one'])} ±1 (n={c['n']})"
                         for a, c in q["categories"].items() if c["n"])
        body.append([kind, str(q["runs"]), pct(q["recall"]), pct(q["precision"]), pct(q["relation_recall"]), cats or "n/a"])
    out += table(["prompts", "runs", "feature recall", "precision", "relation recall", "category accuracy (matched features)"], body)
    return out + ["", "Features match one-to-one on compatible type and overlapping position bands (eval/spec_quality.py). "
                  "Category accuracy counts only categories the answer key states; one the agent left unspecified counts as wrong.", ""]


def limitations(rows: list[dict[str, Any]], calibrated: bool) -> list[str]:
    d = done(rows)
    prompts = {r["prompt_id"]: r["kind"] for r in d}
    seeds = sorted({r["seed"] for r in d})
    models = Counter((r["models"]["builder"], r["models"].get("critic"), r["models"]["judge"]) for r in d)
    shared = Counter(role for r in d for role in shared_family(r["models"]))
    judge_unknown = sum(overall(r) is None for r in d)
    errors = [r for r in rows if r.get("status") == "error"]
    budget = sum(str(r.get("stop", "")).startswith("budget") for r in d)
    unpriced = sum(r.get("cost_usd") is None for r in d)
    out = ["## Limitations", ""]
    out.append(f"- Small N: {len(prompts)} prompts ({sum(k == 'text' for k in prompts.values())} text, "
               f"{sum(k == 'image' for k in prompts.values())} image), seeds {seeds}, {len(d)} finished runs. "
               "Differences smaller than the CIs above can't be told apart from noise.")
    for (b, c, j), n in sorted(models.items(), key=lambda kv: -kv[1]):
        out.append(f"- Models ({n} runs): builder `{b}`, critic `{c or 'off'}`, judge `{j}`.")
    if shared["builder"]:
        out.append(f"- The judge shares the builder's model family in {shared['builder']} runs. Acceptable for E1 and E2: "
                   "both arms of each comparison use the same builder, so a family preference affects them alike.")
    if shared["critic"]:
        out.append(f"- The judge shares the critic's model family in {shared['critic']} runs: it may share the taste the "
                   "critic optimized for, which favours critic_on in E1.")
    out.append("- The judge is human-calibrated (E3 below)." if calibrated else
               "- The judge is not yet calibrated against human ratings (E3 pending); its pass/fail is part of success.")
    if judge_unknown:
        out.append(f"- {judge_unknown} runs have no usable judge verdict; their success is unknown and left out of the success rates.")
    if budget:
        out.append(f"- {budget} runs stopped on a budget limit; they are graded on the latest accepted terrain.")
    if errors:
        out.append(f"- {len(errors)} runs ended in an error and are not in the tables: {', '.join(r['key'] for r in errors)}.")
    if unpriced:
        out.append(f"- {unpriced} runs used a model without a price; their cost is unknown.")
    out.append("- The answer key grades layout and categories only; looks rest on one judge model.")
    return out + [""]


def render(rows: list[dict[str, Any]], name: str, folders: list[Path], keys: dict, calibration: str | None = None,
           sample: bool = False) -> str:
    d = done(rows)
    head = [f"# Evaluation report: {name}", ""]
    if sample:
        head += ["> **SAMPLE: generated from fake data to show the format. These numbers mean nothing.**", ""]
    commits = sorted({(r.get("git") or {}).get("commit", "?")[:7] for r in d})
    head += [f"Generated {time.strftime('%Y-%m-%d')} from {', '.join(f'`{f.as_posix()}`' for f in folders)}: {len(d)} finished runs, "
             f"code {', '.join(commits)}.", "",
             "Success = no failed major answer-key item (matcher mode) AND no failed blocking/major programmatic check "
             "AND judge pass. The checklist pass rate is reported as a continuous score.", ""]
    body = head
    for code, title, a, b in EXPERIMENTS:
        body += experiment_section(rows, code, title, a, b)
    body += e4_section(rows) + e5_section(rows, keys)
    calibrated = False
    if calibration:
        path = Path("eval/calibration") / calibration / "score.md"
        if path.exists():
            body += [path.read_text(encoding="utf-8").strip(), ""]
            calibrated = True
    if not calibrated:
        body += ["## E3: Judge calibration", "", "Not scored yet (`python -m eval.calibrate score NAME`).", ""]
    body += limitations(rows, calibrated)
    return "\n".join(body)


CSV_FIELDS = ["folder", "key", "prompt_id", "kind", "condition", "seed", "status", "stop", "rounds", "accepted_round",
              "success", "success_checklist", "success_checks", "success_judge", "pass_rate", "passed", "total",
              "major_failures", "checks_failed", "judge_mean", "judge_verdict", "cost_usd", "judge_cost_usd", "latency_s",
              "steps", "tool_calls", "tool_errors", "builder", "critic", "judge", "judge_same_family"]


def csv_row(r: dict[str, Any]) -> dict[str, Any]:
    s, a, j, t = r.get("success") or {}, r.get("answer_key") or {}, r.get("judge") or {}, r.get("trace_stats") or {}
    return {
        "folder": r.get("folder"), "key": r["key"], "prompt_id": r["prompt_id"], "kind": r["kind"], "condition": r["condition"],
        "seed": r["seed"], "status": r.get("status"), "stop": r.get("stop"), "rounds": r.get("rounds"),
        "accepted_round": r.get("accepted_round"), "success": s.get("overall"), "success_checklist": s.get("checklist"),
        "success_checks": s.get("checks"), "success_judge": s.get("judge"), "pass_rate": a.get("pass_rate"),
        "passed": a.get("passed"), "total": a.get("total"), "major_failures": " ".join(a.get("major_failures") or []),
        "checks_failed": " ".join((r.get("checks") or {}).get("failed") or []), "judge_mean": j.get("mean"),
        "judge_verdict": (j.get("output") or {}).get("verdict"), "cost_usd": r.get("cost_usd"), "judge_cost_usd": judge_paid(r),
        "latency_s": r.get("latency_s"), "steps": t.get("steps"), "tool_calls": t.get("tool_calls"), "tool_errors": t.get("tool_errors"),
        "builder": r["models"]["builder"], "critic": r["models"].get("critic"), "judge": r["models"].get("judge"),
        "judge_same_family": r["models"].get("judge_same_family"),
    }


def write(rows: list[dict[str, Any]], name: str, folders: list[Path], keys: dict, out_dir: Path = REPORTS,
          calibration: str | None = None, sample: bool = False) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{time.strftime('%Y-%m-%d')}-{name}"
    md, runs_csv = out_dir / f"{stem}.md", out_dir / f"{stem}-runs.csv"
    md.write_text(render(rows, name, folders, keys, calibration, sample), encoding="utf-8")
    with runs_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(csv_row(r) for r in rows)
    return md, runs_csv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", type=Path, nargs="+")
    ap.add_argument("--name", required=True)
    ap.add_argument("--suite", type=Path, default=SUITE)
    ap.add_argument("--calibration", default=None, help="eval/calibration/<NAME> with a score.md to include as E3")
    args = ap.parse_args()
    keys = {e.id: e.graph for e in load(args.suite)}
    md, runs_csv = write(load_rows(args.folders), args.name, args.folders, keys, calibration=args.calibration)
    print(f"wrote {md} and {runs_csv}")


if __name__ == "__main__":
    main()
