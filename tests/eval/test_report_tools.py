"""Report (eval/report.py), rescore, calibration (E3) and regression (E7) on fake rows and small real stacks."""

import csv
import json
from pathlib import Path

from agent.session import Session
from eval import calibrate, regression, report
from eval.benchmark import load
from eval.rescore import rescore
from eval.rows import append_row, read_rows, trace_stats, write_rows
from tests.agent.fakes import PEAK
from tests.eval.fake_rows import e1_rows, row

KEYS = {e.id: e.graph for e in load()}


def folder(tmp_path, name, rows):
    d = tmp_path / name
    d.mkdir()
    for r in rows:
        append_row(d / "runs.jsonl", r)
    return d


def test_report_from_fake_rows(tmp_path):
    e1 = folder(tmp_path, "e1_critic", e1_rows())
    e2 = folder(tmp_path, "e2_spec", [row(pid, kind, "spec_off", s, overall=False, pass_rate=0.5)
                                       for pid, kind in [("t01", "text"), ("i01", "image")] for s in (1, 2)])
    md, runs_csv = report.write(report.load_rows([e1, e2]), "fake", [e1, e2], KEYS, out_dir=tmp_path / "reports", sample=True)
    text = md.read_text(encoding="utf-8")
    for heading in ("## E1: Critic on vs off", "## E2: SPEC on vs off", "## E4: Cost and efficiency",
                    "## E5: SPEC quality", "## E3: Judge calibration", "## Limitations", "SAMPLE"):
        assert heading in text
    # critic_on's pass rate is 0.8-1.0 and critic_off's 0.4-0.6 in every pair: a clear, one-sided difference.
    assert "| all | pass rate | 12 / 0 / 0 |" in text and "critic_on higher" in text
    assert "| image | pass rate | 4 / 0 / 0 |" in text and "too few prompts to tell (2)" in text  # 2 image prompts
    # E2 pairs only the prompts that have spec_off runs (t01, i01 x 2 seeds).
    e2_part = text.split("## E2")[1].split("## E4")[0]
    assert "| all | `spec_off` | 4 |" in e2_part and "| all | `critic_on` | 4 |" in e2_part
    # Columns: critic_off, critic_on, spec_off. Each fake run has two comparisons, one won by the revision,
    # and two revision rounds, one rolled back.
    assert "| revision win rate | 50% | 50% | 50% |" in text and "| Terrain 1 win rate (decisive) | 100% | 100% | 100% |" in text
    assert "| rollback rate (revisions) | 50% | 50% | 50% |" in text
    assert "seeds [1, 2], 28 finished runs" in text
    assert len(list(csv.DictReader(runs_csv.open(encoding="utf-8")))) == 28


def test_small_n_reads_as_no_clear_difference(tmp_path):
    rows = [row(f"t0{p}", "text", c, s, overall=v, pass_rate=0.5) for p in range(1, 6)
            for c, vals in (("critic_on", [True, False]), ("critic_off", [False, True])) for s, v in zip((1, 2), vals)]
    text = report.render(rows, "x", [tmp_path], KEYS)
    assert "no clear difference at this N" in text and "critic_on higher" not in text


def test_a_judge_failure_leaves_success_unknown_not_failed(tmp_path):
    rows = [row("t01", "text", "critic_on", 1, overall=None, pass_rate=0.9), row("t01", "text", "critic_off", 1, overall=True, pass_rate=0.5)]
    s = report.condition_summary(rows, "critic_on", None)
    assert s["success"] is None and s["unknown"] == 1
    assert report.compare(report.pairs(rows, "critic_on", "critic_off"), "success")["unknown"] == 1
    assert "no usable judge verdict" in report.render(rows, "x", [tmp_path], KEYS)


def real_run(tmp_path, key="t01-critic_on-s1", prompt="t01", kind="text"):
    """A done row whose stack.json is a real (small) stack, plus fake judge renders next to it."""
    s = Session.flat(33, 33, base_y=64, seed=1)
    s.apply("raise_peak", PEAK | {"center": [0.25, 0.3]}, id="west")
    d = tmp_path / "res" / key
    d.mkdir(parents=True)
    (d / "stack.json").write_text(s.stack.to_json(), encoding="utf-8")
    for v in ("overview_sw", "overview_ne", "overview_s"):
        (d / f"judge_{v}.png").write_bytes(b"\x89PNG")
    return row(prompt, kind, key.split("-")[1], int(key[-1]), overall=True, pass_rate=1.0, stack=(d / "stack.json").as_posix())


def test_rescore_recomputes_from_saved_stacks(tmp_path):
    r = real_run(tmp_path)
    write_rows(tmp_path / "runs.jsonl", [r])
    [new] = rescore(read_rows(tmp_path / "runs.jsonl"), {e.id: e for e in load()}, Path("eval"))
    # The fake row claimed a full pass; the real stack has one peak, so most of t01's items fail.
    assert new["answer_key"]["pass_rate"] < 0.5 and new["success"]["overall"] is False
    assert new["judge"] == r["judge"] and "scored" in new
    write_rows(tmp_path / "runs.jsonl", [new])
    assert (tmp_path / "runs.jsonl.bak").exists()


def test_calibration_sample_hides_the_condition_and_scores_ratings(tmp_path):
    rows = []
    for k, (pid, kind, cond) in enumerate([("t01", "text", "critic_on"), ("t02", "text", "critic_off"),
                                            ("i01", "image", "critic_on"), ("t03", "text", "critic_off")]):
        r = real_run(tmp_path, f"{pid}-{cond}-s1", pid, kind)
        r["judge"] |= {"passed": k % 2 == 0, "mean": 2.0 + k}
        rows.append(r)
    res = folder(tmp_path, "e1", rows)
    out = calibrate.sample([res], "cal", n=4, screenshots=1, root=tmp_path / "cal")
    page = (out / "sheet.html").read_text(encoding="utf-8")
    assert "critic_on" not in page and "critic_off" not in page and "verdict" not in page
    assert page.count("<section>") == 4 and page.count("Rate from in-game screenshots") == 1
    assert "--origin 0 0" in page and "TerraformEval" in page
    ratings = list(csv.DictReader((out / "ratings.csv").open(encoding="utf-8")))
    assert [r["source"] for r in ratings].count("screenshot") == 1
    key = json.loads((out / "key.json").read_text(encoding="utf-8"))
    # Rate in line with the judge: your mean tracks its mean, pass where it passes.
    for r in ratings:
        j = key[r["item"]]["judge"]
        r.update(naturalism=str(j["mean"]), minecraft_quality=str(j["mean"]), materials=str(j["mean"]),
                 **{"pass": "y" if j["passed"] else "n"})
        if r["reference_match"] != "n/a":
            r["reference_match"] = str(j["mean"])
    with (out / "ratings.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=calibrate.FIELDS)
        w.writeheader()
        w.writerows(ratings)
    results, md = calibrate.score("cal", root=tmp_path / "cal")
    assert results["all"]["n"] == 4 and results["all"]["spearman"] == 1.0
    assert results["all"]["agreement"] == 1.0 and results["all"]["kappa"] == 1.0
    assert results["render"]["n"] + results["screenshot"]["n"] == 4
    assert "## E3: Judge calibration" in md.read_text(encoding="utf-8")


def test_kappa_and_confusion():
    r = calibrate.agreement([(4, 4, True, True), (2, 3, False, True), (1, 1, False, False), (5, 4, True, True)])
    assert r["agreement"] == 0.75 and r["confusion"]["human fail, judge pass"] == 1
    assert round(r["kappa"], 3) == 0.5  # po 0.75, pe 0.5


def test_regression_flags_drops_and_warns_on_cost():
    base = {"t01": {"success": True, "pass_rate": 1.0, "cost_usd": 0.5}, "t03": {"success": False, "pass_rate": 0.8, "cost_usd": 0.5},
            "t05": {"success": True, "pass_rate": 0.9, "cost_usd": 0.5}}
    now = {"t01": {"success": False, "pass_rate": 0.95, "cost_usd": 0.5}, "t03": {"success": False, "pass_rate": 0.6, "cost_usd": 0.9},
           "t05": {"success": True, "pass_rate": 0.9, "cost_usd": 0.5}, "i01": {"success": True, "pass_rate": 1.0, "cost_usd": 0.4}}
    drops, warnings = regression.compare(now, base)
    assert drops == ["t01: success True -> False", "t03: checklist pass rate 80% -> 60%"]
    assert warnings == ["i01: not in the baseline", "t03: cost $0.50 -> $0.90"]


def test_trace_stats_reads_roles_gate_and_comparisons(tmp_path):
    recs = [
        {"stage": "run", "note": "run header"},
        {"stage": "spec", "usage": {"input": 1, "output": 2, "cache_read": 0, "cache_write": 0}, "cost_usd": 0.01},
        {"stage": "terrain", "usage": {"input": 5, "output": 5, "cache_read": 10, "cache_write": 0}, "cost_usd": 0.02,
         "tool_calls": [{"name": "raise_peak", "is_error": False}, {"name": "carve_river", "is_error": True}]},
        {"stage": "terrain/critique", "usage": {"input": 1, "output": 1, "cache_read": 0, "cache_write": 0}, "cost_usd": None,
         "scores": {"verdict": "revise", "failed": None, "fixes": {"issues": 2, "with_fix": 2, "valid": 1, "none": 0}}},
        {"stage": "terrain", "note": "accepted round 1: first attempt (baseline)"},
        {"stage": "terrain/compare", "usage": {"input": 1, "output": 1, "cache_read": 0, "cache_write": 0}, "cost_usd": 0.03,
         "scores": {"round": 2, "order": {"1": "revision", "2": "previous"}, "winner": "previous", "better": "2"}},
        {"stage": "terrain", "note": "rolled back round 2: a blind comparison preferred the accepted attempt"},
    ]
    path = tmp_path / "t.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    s = trace_stats(path)
    assert s["calls"] == {"spec": 1, "builder": 1, "critic": 1, "comparison": 1} and s["steps"] == 4
    assert s["tokens"]["builder"]["cache_read"] == 10 and s["unpriced_calls"] == 1
    assert (s["tool_calls"], s["tool_errors"]) == (2, 1) and s["fixes"]["valid"] == 1
    assert [g["decision"] for g in s["gate"]] == ["accepted", "rolled_back"]
    assert s["comparisons"] == [{"round": 2, "order": {"1": "revision", "2": "previous"}, "winner": "previous", "better": "2"}]
