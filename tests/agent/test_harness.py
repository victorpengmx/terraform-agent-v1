import json

import numpy as np
import pytest

from agent.budget import Budget
from agent.critic import Critic
from agent.harness import run_task
from agent.session import Session
from agent.tools import observe_tools, op_tools
from agent.tracing import Tracer
from fakes import PEAK, ScriptedModel, critique_json, response, text, tool_use


@pytest.fixture
def session():
    return Session.flat(33, 33, base_y=64)


def run(session, model, tmp_path, **kw):
    tools = {t.name: t for t in op_tools(session)}
    kw.setdefault("budget", Budget())
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


def test_refusal_ends_the_run(session, tmp_path):
    result = run(session, ScriptedModel(response(stop="refusal")), tmp_path)
    assert result.stop.startswith("refused")


def test_step_budget_stops_a_looping_model(session, tmp_path):
    loop = [response(tool_use("raise_peak", PEAK | {"height": 1}, id=f"tu_{i}"), stop="tool_use") for i in range(5)]
    result = run(session, ScriptedModel(*loop), tmp_path, budget=Budget(max_steps=3))
    assert result.stop.startswith("budget: step budget") and len(session.ops) == 3


def test_episode_rotates_when_context_is_long(session, tmp_path):
    model = ScriptedModel(
        response(tool_use("raise_peak", PEAK), stop="tool_use", input_tokens=200_000),
        response(text("done")),
    )
    run(session, model, tmp_path)
    fresh = model.calls[1]["messages"]
    assert len(fresh) == 1 and "continues earlier work" in fresh[0]["content"][-1]["text"]


def test_score_gated_rollback_then_accept(session, tmp_path):
    builder = ScriptedModel(
        response(tool_use("raise_peak", PEAK), stop="tool_use"), response(text("round 1")),
        response(tool_use("raise_peak", PEAK | {"height": 100}, id="tu_2"), stop="tool_use"), response(text("round 2")),
        response(tool_use("raise_peak", PEAK | {"height": 5}, id="tu_3"), stop="tool_use"), response(text("round 3")),
    )
    critic = Critic(ScriptedModel(critique_json(3), critique_json(2), critique_json(5, "pass")))
    result = run(session, builder, tmp_path, critic=critic, max_rounds=3)

    assert result.stop == "accepted" and result.rounds == 3 and result.best_score == 5
    # Round 2 scored lower, so it was rolled back: round 3 built on round 1's state.
    assert [r.params["height"] for r in session.ops] == [40, 5]
    assert np.isclose(session.state.height.max(), 64 + 45)
    round3_prompt = builder.calls[4]["messages"][0]["content"][-1]["text"]
    assert "rolled back" in round3_prompt and "too round" in round3_prompt


def test_max_rounds_keeps_best_attempt(session, tmp_path):
    builder = ScriptedModel(
        response(tool_use("raise_peak", PEAK), stop="tool_use"), response(text("r1")),
        response(tool_use("raise_peak", PEAK, id="tu_2"), stop="tool_use"), response(text("r2")),
    )
    critic = Critic(ScriptedModel(critique_json(4), critique_json(3)))
    result = run(session, builder, tmp_path, critic=critic, max_rounds=2)
    assert result.stop == "max_rounds" and result.best_score == 4 and len(session.ops) == 1


def test_render_view_image_reaches_the_model_but_not_the_trace(session, tmp_path):
    model = ScriptedModel(
        response(tool_use("raise_peak", PEAK), stop="tool_use"),
        response(tool_use("render_view", {"view": "map"}, id="tu_2"), stop="tool_use"),
        response(text("Looked at it; done.")),
    )
    tools = {t.name: t for t in [*op_tools(session), *observe_tools(session, tmp_path / "views")]}
    run_task(prompt="a hill", images=[], session=session, tools=tools, model=model,
             tracer=Tracer(run_id="t", root=tmp_path), budget=Budget())
    last = model.calls[2]["messages"][-1]["content"][0]  # tool_result for render_view
    assert last["is_error"] is False and [b["type"] for b in last["content"]] == ["text", "image"]
    trace = (tmp_path / "t.jsonl").read_text()
    assert '"result": "[content blocks]"' in trace and "base64" not in trace and len(trace) < 20_000
