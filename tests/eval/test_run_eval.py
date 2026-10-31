"""The eval runner (eval/run_eval.py) end to end with fake adapters: rows, the answer key kept from the agent,
resumability and the cost gate. No network (tests/conftest.py)."""

import json
import sys

import pytest

import eval.run_eval as run_eval
from agent.models import ModelConfig
from eval.benchmark import load
from eval.config import EvalConfig
from eval.rows import read_rows
from tests.agent.fakes import PEAK, ScriptedModel, critique_json, feature, graph, response, spec_json, text, tool_use

ENTRIES = load()
KEY_IDS = {f.id for e in ENTRIES if e.id == "t01" for f in e.graph.features}  # mountain_west, valley_1, ...


def config(**kw) -> EvalConfig:
    return EvalConfig.model_validate({
        "name": "test", "prompts": ["t01"], "seeds": [1], "size": [33, 33], "render": "none",
        "models": {"builder": "claude-haiku-5-5", "critic": "claude-opus-5-5", "judge": "claude-sonnet-5-5"},
        "conditions": {"critic_on": {"critic": True, "rounds": 1}, "critic_off": {"critic": False}}} | kw)


def judge_reply():
    return response(text(json.dumps({"reasons": ["fine"], "verdict": "pass", "scores": {
        "naturalism": 4, "minecraft_quality": 4, "materials": 4, "reference_match": None}})))


class Models:
    """make_model stand-in: a fresh scripted adapter per role, every one recorded."""

    def __init__(self, fail_builder: bool = False):
        self.made: list[tuple[str, ScriptedModel]] = []
        self.fail_builder = fail_builder

    def __call__(self, cfg: ModelConfig) -> ScriptedModel:
        if cfg.model == "claude-haiku-5-5":  # builder: its own SPEC, one peak, done
            if self.fail_builder:
                m = ScriptedModel()  # runs out at once: the run crashes
            else:
                m = ScriptedModel(spec_json(graph(feature("big_peak", x="left"), feature("stream", "river"))),
                                  response(tool_use("raise_peak", PEAK | {"center": [0.25, 0.4]}), stop="tool_use"),
                                  response(text("done")))
        elif cfg.model == "claude-opus-5-5":
            m = ScriptedModel(critique_json(4, "pass"))
        else:
            m = ScriptedModel(*[judge_reply() for _ in range(4)])
        self.made.append((cfg.model, m))
        return m


def fake_render(stack, views):
    return [(v, b"\x89PNG " + v.encode()) for v in views]


def run(tmp_path, models, cfg=None):
    return run_eval.evaluate(cfg or config(), ENTRIES, tmp_path / "res", run_eval.SUITE.parent,
                             make_model=models, cache=tmp_path / "cache", render=fake_render, log=lambda s: None)


def test_rows_are_complete_and_the_agent_never_sees_the_answer_key(tmp_path):
    models = Models()
    new = run(tmp_path, models)
    assert [r["key"] for r in new] == ["t01-critic_on-s1", "t01-critic_off-s1"]
    on, off = new
    assert on["status"] == off["status"] == "done"
    assert on["kind"] == "text" and on["seed"] == 1 and on["models"]["judge"] == "claude-haiku-5-5"  # fakes report haiku
    assert on["settings"]["critic"] is True and off["settings"]["critic"] is False and off["models"]["critic"] is None
    assert on["answer_key"]["total"] == 11 and 0 < on["answer_key"]["pass_rate"] < 1  # one peak: most items fail
    assert on["success"]["checklist"] is False and on["success"]["judge"] is True and on["success"]["overall"] is False
    assert on["spec_graph"]["features"][0]["id"] == "big_peak"  # the agent's own SPEC
    assert on["trace_stats"]["calls"] == {"spec": 1, "builder": 2, "critic": 1} and off["trace_stats"]["calls"] == {"spec": 1, "builder": 2}
    assert on["trace_stats"]["gate"] == [{"round": 1, "decision": "accepted", "why": "first attempt (baseline)"}]
    assert on["trace_stats"]["tool_calls"] == 1 and on["trace_stats"]["fixes"]["issues"] == 1
    assert (tmp_path / "res" / "t01-critic_on-s1" / "stack.json").exists()
    assert (tmp_path / "res" / "t01-critic_on-s1" / "judge_overview_sw.png").exists()
    # Graph ids from the answer key never reach any model, in any role.
    for _, m in models.made:
        sent = json.dumps([c["messages"] for c in m.calls], default=str)
        assert not [i for i in KEY_IDS if i in sent], [i for i in KEY_IDS if i in sent]


def test_spec_off_builds_from_the_raw_prompt(tmp_path):
    cfg = config(conditions={"spec_off": {"critic": False, "use_spec": False}})
    # No SPEC reply scripted: a SPEC call would run out.
    builder = ScriptedModel(response(tool_use("raise_peak", PEAK), stop="tool_use"), response(text("done")))
    judge = ScriptedModel(judge_reply())
    make = lambda c: builder if c.model == "claude-haiku-5-5" else judge  # noqa: E731
    [row] = run(tmp_path, make, cfg)
    assert row["status"] == "done" and row["spec_graph"] is None and row["settings"]["use_spec"] is False
    assert "spec" not in row["trace_stats"]["calls"]


def test_finished_runs_are_skipped_and_errors_retried(tmp_path):
    failing = Models(fail_builder=True)
    first = run(tmp_path, failing)
    assert [r["status"] for r in first] == ["error", "error"] and "ran out" in first[0]["error"]
    fixed = Models()
    second = run(tmp_path, fixed)
    assert [r["status"] for r in second] == ["done", "done"]
    rows = read_rows(tmp_path / "res" / "runs.jsonl")
    assert {k: r["status"] for k, r in rows.items()} == {"t01-critic_on-s1": "done", "t01-critic_off-s1": "done"}
    again = Models()
    assert run(tmp_path, again) == [] and again.made == []  # nothing pending: no adapter is even built


def test_the_cost_gate_spends_nothing_without_yes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_eval, "RESULTS", tmp_path)
    monkeypatch.setattr(run_eval, "evaluate", lambda *a, **k: pytest.fail("evaluate ran without --yes"))
    monkeypatch.setattr(sys, "argv", ["run_eval", "--config", "eval/configs/pilot_haiku.yaml"])
    with pytest.raises(SystemExit, match="--yes"):
        run_eval.main()
    out = capsys.readouterr().out
    assert "4 pending runs; estimated cost up to $2.51" in out
    assert "critic_on: 2 runs x $1.00 cap = $2.00" in out and "critic_off: 2 runs x $0.25 cap = $0.50" in out
    monkeypatch.setattr(sys, "argv", ["run_eval", "--config", "eval/configs/pilot_haiku.yaml", "--dry-run", "--yes"])
    run_eval.main()  # a dry run never spends, even with --yes


def test_a_role_without_a_price_makes_its_condition_unknown_never_zero():
    from eval.config import load_config

    e1 = load_config("eval/configs/e1_critic.yaml")  # critic gemini-3.8-flash: no price yet
    text, upper = run_eval.estimate(e1, run_eval.plan(e1, ENTRIES))
    assert upper is None and "$0.00" not in text
    assert "critic_on: 30 runs x $1.00 cap = unknown (no price for critic gemini-3.8-flash, comparison gemini-3.8-flash" in text
    assert "critic_off: 30 runs x $0.25 cap = $7.50" in text  # Haiku builder only: priced
    assert "estimated cost up to unknown (priced part up to $" in text
    opus = load_config("eval/configs/opus.yaml")  # judge gemini-3.8-flash: no price yet
    text, upper = run_eval.estimate(opus, run_eval.plan(opus, ENTRIES))
    assert upper is None and "judge: 15 calls on gemini-3.8-flash = unknown (no price for judge gemini-3.8-flash)" in text
    assert "critic_on: 15 runs x $3.00 cap = $45.00" in text


def test_pairs_share_prompt_and_seed_across_conditions():
    runs = run_eval.plan(config(seeds=[1, 2]), ENTRIES)
    cfgs = {r.key: run_eval.run_config(config(seeds=[1, 2]), r, run_eval.SUITE.parent) for r in runs}
    assert cfgs["t01-critic_on-s2"].seed == cfgs["t01-critic_off-s2"].seed == 2
    assert cfgs["t01-critic_on-s1"].critic and not cfgs["t01-critic_off-s1"].critic
    assert cfgs["t01-critic_off-s1"].budget.max_cost_usd == 1.0  # this test config has no override
