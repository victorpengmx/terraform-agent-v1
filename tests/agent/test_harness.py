import json

import numpy as np
import pytest

from agent.budget import Budget
from agent.critic import Critic
from agent.harness import comparison_order, run_task
from agent.models import ModelConfig
from agent.session import Session
from agent.tools import control_tools, observe_tools, op_tools
from agent.tracing import Tracer
from fakes import PEAK, ScriptedModel, compare_json, critique_json, response, text, tool_use


@pytest.fixture
def session():
    return Session.flat(33, 33, base_y=64)


def run(session, model, tmp_path, **kw):
    tools = {t.name: t for t in op_tools(session)}
    kw.setdefault("budget", Budget())
    kw.setdefault("use_spec", False)  # SPEC behaviour is tested separately (test_spec_step.py)
    return run_task(prompt="a hill", images=[], session=session, tools=tools, model=model,
                    tracer=Tracer(run_id="t", root=tmp_path), **kw)


def test_single_pass_applies_tools_and_finishes(session, tmp_path):
    model = ScriptedModel(
        response(tool_use("raise_peak", PEAK), stop="tool_use"),
        response(text("Built one hill.")),
    )
    result = run(session, model, tmp_path)
    assert result.stop == "done" and result.summary == "Built one hill."
    assert [r.id for r in session.ops] == ["raise_peak_1"]
    # The tool result went back in one user message after the assistant turn.
    second = model.calls[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert second[2]["content"][0]["is_error"] is False
    assert "write_note" in [t["name"] for t in model.calls[0]["tools"]]
    steps = [json.loads(l) for l in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert steps[0]["tool_calls"][0]["name"] == "raise_peak" and steps[-1]["note"] == "run ended: done"


def test_tool_errors_go_back_to_the_model(session, tmp_path):
    model = ScriptedModel(
        response(tool_use("raise_peak", PEAK | {"radius": 5}), stop="tool_use"),
        response(text("ok")),
    )
    run(session, model, tmp_path)
    result_block = model.calls[1]["messages"][2]["content"][0]
    assert result_block["is_error"] is True and "radius" in result_block["content"]
    assert session.ops == []


def test_truncated_tool_call_is_not_executed(session, tmp_path):
    model = ScriptedModel(
        response(tool_use("raise_peak", PEAK), stop="max_tokens"),
        response(text("done")),
    )
    run(session, model, tmp_path)
    assert session.ops == []
    block = model.calls[1]["messages"][2]["content"][0]
    assert block["is_error"] is True and "cut off" in block["content"]


def test_malformed_tool_call_is_fed_back_then_two_in_a_row_stop(session, tmp_path):
    model = ScriptedModel(response(stop="malformed_tool_call"), response(tool_use("raise_peak", PEAK), stop="tool_use"),
                          response(text("done")))
    assert run(session, model, tmp_path).stop == "done" and [r.id for r in session.ops] == ["raise_peak_1"]
    assert "malformed and was not run" in model.calls[1]["messages"][2]["content"]
    again = ScriptedModel(response(stop="malformed_tool_call"), response(stop="malformed_tool_call"))
    assert run(Session.flat(33, 33, base_y=64), again, tmp_path).stop == "malformed tool calls: 2 in a row"


def test_refusal_ends_the_run(session, tmp_path):
    result = run(session, ScriptedModel(response(stop="refusal")), tmp_path)
    assert result.stop.startswith("refused")


def test_step_budget_stops_a_looping_model(session, tmp_path):
    loop = [response(tool_use("raise_peak", PEAK | {"height": 1}, id=f"tu_{i}"), stop="tool_use") for i in range(5)]
    result = run(session, ScriptedModel(*loop), tmp_path, budget=Budget(max_steps=3))
    assert result.stop.startswith("budget: step budget") and len(session.ops) == 3


def test_a_budget_spent_by_the_build_skips_the_critic_and_ends_cleanly(session, tmp_path):
    builder = ScriptedModel(response(tool_use("raise_peak", PEAK), stop="tool_use"), response(text("built")))
    critic_model = ScriptedModel()  # no responses: any critic call would fail the test
    result = run(session, builder, tmp_path, critic=Critic(critic_model), budget=Budget(max_steps=2), max_rounds=3)
    assert result.stop == "budget: step budget reached (2/2 model calls)" and critic_model.calls == []
    assert result.critiques == [] and result.accepted_round is None and [e.id for e in session.ops] == ["raise_peak_1"]
    notes = [json.loads(l)["note"] or "" for l in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert "critique skipped: budget: step budget reached (2/2 model calls)" in notes
    assert notes[-1] == "run ended: budget: step budget reached (2/2 model calls)"


def test_a_budget_spent_by_the_critique_skips_the_comparison_and_keeps_the_accepted_attempt(session, tmp_path):
    # Steps: build 2 + critique 1 + build 2 + critique 1 = 6, so the comparison must not run.
    critic_model = ScriptedModel(critique_json(3), critique_json(4))
    result = run(session, two_rounds(("raise_peak", HILL_2)), tmp_path, critic=Critic(critic_model),
                 budget=Budget(max_steps=6), max_rounds=3)
    assert result.stop.startswith("budget: step budget") and len(critic_model.calls) == 2 and len(result.critiques) == 2
    assert result.accepted_round == 1 and [e.id for e in session.ops] == ["raise_peak_1"]  # the unjudged revision is undone
    assert "comparison skipped: budget" in (tmp_path / "t.jsonl").read_text()


def test_episode_rotates_when_context_is_long(session, tmp_path):
    model = ScriptedModel(
        response(tool_use("raise_peak", PEAK), stop="tool_use", input_tokens=200_000),
        response(text("done")),
    )
    run(session, model, tmp_path)
    fresh = model.calls[1]["messages"]
    assert len(fresh) == 1 and "continues earlier work" in fresh[0]["content"][-1]["text"]


@pytest.mark.parametrize("model_id, rotates", [("claude-haiku-5-5", True), ("claude-opus-5-5", False)])
def test_rotation_stays_below_a_long_prompt_tier(session, tmp_path, model_id, rotates):
    """Haiku has a 100K long-prompt tier, so its episodes rotate at 90K prompt tokens; Opus keeps 150K."""
    model = ScriptedModel(response(tool_use("raise_peak", PEAK), stop="tool_use", input_tokens=95_000), response(text("done")))
    model.config = ModelConfig(model=model_id)
    run(session, model, tmp_path)
    assert (len(model.calls[1]["messages"]) == 1) is rotates
    assert ("episode rotated" in (tmp_path / "t.jsonl").read_text()) is rotates


def test_a_rotated_episode_keeps_the_rounds_critique_and_rollback_feedback(session, tmp_path):
    # Round 1 accepted, round 2 rolled back by the comparison; round 3's first reply overflows the context.
    builder = ScriptedModel(
        response(tool_use("raise_peak", PEAK), stop="tool_use"), response(text("r1")),
        response(tool_use("raise_peak", HILL_2, id="tu_2"), stop="tool_use"), response(text("r2")),
        response(tool_use("update_op", {"id": "raise_peak_1", "params": {"height": 14}}, id="tu_3"), stop="tool_use",
                 input_tokens=200_000),
        response(text("r3")),
    )
    critic_model = ScriptedModel(critique_json(3), critique_json(3), compare_json(previous_wins(0, 2), "the peak got worse"),
                                 critique_json(3), compare_json("same"))
    tools = {t.name: t for t in [*op_tools(session), *control_tools(session)]}
    run_task(prompt="a hill", images=[], session=session, tools=tools, model=builder, critic=Critic(critic_model),
             tracer=Tracer(run_id="t", root=tmp_path), budget=Budget(), max_rounds=3, use_spec=False)
    rotated = builder.calls[5]["messages"]
    opening = rotated[0]["content"][-1]["text"]
    assert len(rotated) == 1 and "continues earlier work" in opening
    assert "# Critique of the current terrain" in opening and "too round" in opening
    assert "Round 2 was not kept" in opening and "the peak got worse" in opening


# ---------------------------------------------------------------- acceptance gate (DESIGN §11.4)

HILL_2 = PEAK | {"center": [0.25, 0.25], "radius": 0.2, "height": 10}  # passes every check
THIN_WALL = {"region": {"polygon": [[0.2, 0.5], [0.8, 0.5], [0.8, 0.53], [0.2, 0.53]]}, "height": 20, "edge": 0}


def two_rounds(*round2_ops, rounds=2):
    """Builder script: round 1 raises PEAK, round 2 applies `round2_ops`, later rounds change nothing."""
    calls = [response(tool_use("raise_peak", PEAK), stop="tool_use"), response(text("r1"))]
    calls += [response(*[tool_use(op, params, id=f"tu_2_{i}") for i, (op, params) in enumerate(round2_ops)], stop="tool_use"),
              response(text("r2"))]
    calls += [response(text(f"r{n}")) for n in range(3, rounds + 1)]
    return ScriptedModel(*calls)


revision_first = comparison_order


def previous_wins(seed: int, round_no: int) -> str:
    return "2" if revision_first(seed, round_no) else "1"


def test_first_attempt_is_the_baseline_and_a_pass_ends_the_run(session, tmp_path):
    critic_model = ScriptedModel(critique_json(5, "pass"))
    result = run(session, two_rounds(("raise_peak", HILL_2)), tmp_path, critic=Critic(critic_model), max_rounds=2)
    assert result.stop == "accepted" and result.accepted_round == 1 and result.score == 5
    assert len(critic_model.calls) == 1  # nothing to compare against in round 1


def test_check_regression_rolls_back_without_a_comparison(session, tmp_path):
    builder = two_rounds(("raise_region", THIN_WALL), rounds=3)
    critic_model = ScriptedModel(critique_json(3), critique_json(4), critique_json(3), compare_json("same"))
    result = run(session, builder, tmp_path, critic=Critic(critic_model), max_rounds=3)

    assert result.critiques[1].regressions(result.critiques[0])[0].name == "thin_features"
    # Round 2 was rolled back before any comparison: critique, critique, critique, then round 3's comparison.
    assert [c["response_schema"]["title"] for c in critic_model.calls] == ["CritiqueOutput"] * 3 + ["CompareOutput"]
    round3 = builder.calls[4]["messages"][0]["content"][-1]["text"]
    assert "Round 2 was not kept" in round3 and "thin_features" in round3 and "added raise_region_1" in round3
    assert [e.op for e in session.ops] == ["raise_peak"] and result.accepted_round == 3


@pytest.mark.parametrize("seed", range(6))
def test_blind_comparison_rolls_back_when_the_previous_attempt_wins(tmp_path, seed):
    session = Session.flat(33, 33, base_y=64, seed=seed)
    critic_model = ScriptedModel(critique_json(3), critique_json(5), compare_json(previous_wins(seed, 2), "the peak got worse"))
    result = run(session, two_rounds(("raise_peak", HILL_2)), tmp_path, critic=Critic(critic_model), max_rounds=2)
    # A higher rubric score does not matter: the comparison decides, and the run ends on round 1.
    assert result.accepted_round == 1 and [e.id for e in session.ops] == ["raise_peak_1"]
    trace = [json.loads(l) for l in (tmp_path / "t.jsonl").read_text().splitlines()]
    compare = next(r for r in trace if r["stage"] == "terrain/compare")
    assert compare["scores"]["winner"] == "previous" and compare["scores"]["order"]["1"] == ("revision" if revision_first(seed, 2) else "previous")
    assert any("rolled back round 2: a blind comparison preferred the accepted attempt: the peak got worse" in (r["note"] or "")
               for r in trace)


def test_both_comparison_orders_occur_across_seeds():
    assert {revision_first(seed, 2) for seed in range(6)} == {True, False}


@pytest.mark.parametrize("seed", [0, 7, 123_456_789, 2**31 - 1])
def test_comparison_order_varies_across_rounds_and_reproduces_per_seed(seed):
    orders = [comparison_order(seed, r) for r in range(2, 12)]
    assert set(orders) == {True, False}  # both orders across one run's rounds
    assert orders == [comparison_order(seed, r) for r in range(2, 12)]  # the same seed reproduces them


def test_different_seeds_give_different_order_sequences():
    sequences = {tuple(comparison_order(seed, r) for r in range(2, 6)) for seed in range(20)}
    assert len(sequences) > 1


def test_a_tie_keeps_the_revision(session, tmp_path):
    critic_model = ScriptedModel(critique_json(4), critique_json(3), compare_json("same"))
    result = run(session, two_rounds(("raise_peak", HILL_2)), tmp_path, critic=Critic(critic_model), max_rounds=2)
    assert result.stop == "max_rounds" and result.accepted_round == 2 and len(session.ops) == 2


@pytest.mark.parametrize("answers, kept", [(("1", "1"), True), ("previous", False)])
def test_both_orders_must_agree_to_roll_back(session, tmp_path, answers, kept):
    if answers == "previous":  # the accepted attempt wins in both orders
        first = previous_wins(0, 2)
        answers = (first, "1" if first == "2" else "2")
    critic_model = ScriptedModel(critique_json(4), critique_json(4), *[compare_json(a) for a in answers])
    result = run(session, two_rounds(("raise_peak", HILL_2)), tmp_path, critic=Critic(critic_model), max_rounds=2,
                 compare_both_orders=True)
    assert len(critic_model.calls) == 4 and (len(session.ops) == 2) is kept


def test_an_unusable_critic_reply_is_retried_then_recorded_as_a_failed_round(session, tmp_path):
    bad = response(text("not json"))
    critic_model = ScriptedModel(bad, bad)
    result = run(session, two_rounds(("raise_peak", HILL_2)), tmp_path, critic=Critic(critic_model), max_rounds=1)
    critique = result.critiques[0]
    assert critique.failed and "invalid JSON" in critique.failed and critique.checks and critique.verdict == "revise"
    assert result.stop == "max_rounds" and result.accepted_round == 1
    assert "critic failed twice" in (tmp_path / "t.jsonl").read_text()


def test_one_retry_recovers_the_critique(session, tmp_path):
    critic_model = ScriptedModel(response(text("not json")), critique_json(5, "pass"))
    result = run(session, two_rounds(("raise_peak", HILL_2)), tmp_path, critic=Critic(critic_model), max_rounds=1)
    assert result.stop == "accepted" and result.critiques[0].failed is None


def test_the_comparison_never_reveals_which_attempt_is_newer(session, tmp_path):
    titles = []

    def render(state, title):
        titles.append(title)
        return [bytes([0x89]) + b"PNG fake"]

    critic_model = ScriptedModel(critique_json(3), critique_json(3), compare_json("same"))
    run(session, two_rounds(("raise_peak", HILL_2)), tmp_path, critic=Critic(critic_model), max_rounds=2, render=render)
    blocks = critic_model.calls[2]["messages"][0]["content"]
    texts = " ".join(b["text"] for b in blocks if b["type"] == "text")
    assert "Terrain 1" in texts and "Terrain 2" in texts
    assert "raise_peak" not in texts and " ops" not in texts and "round" not in texts.lower()
    assert titles[-2:] == [None, None]  # comparison renders use the renderers' neutral titles


def test_render_view_image_reaches_the_model_but_not_the_trace(session, tmp_path):
    model = ScriptedModel(
        response(tool_use("raise_peak", PEAK), stop="tool_use"),
        response(tool_use("render_view", {"view": "map"}, id="tu_2"), stop="tool_use"),
        response(text("Looked at it; done.")),
    )
    tools = {t.name: t for t in [*op_tools(session), *observe_tools(session, tmp_path / "views")]}
    run_task(prompt="a hill", images=[], session=session, tools=tools, model=model,
             tracer=Tracer(run_id="t", root=tmp_path), budget=Budget(), use_spec=False)
    last = model.calls[2]["messages"][-1]["content"][0]  # tool_result for render_view
    assert last["is_error"] is False and [b["type"] for b in last["content"]] == ["text", "image"]
    trace = (tmp_path / "t.jsonl").read_text()
    assert '"result": "[content blocks]"' in trace and "base64" not in trace and len(trace) < 20_000
