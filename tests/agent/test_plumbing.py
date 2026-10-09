"""Agent plumbing tests. No network: the Anthropic client is faked."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from agent.budget import Budget, BudgetTracker
from agent.models import AnthropicAdapter, ModelConfig, Usage
from agent.session import Session
from agent.tools import control_tools, execute, image_block, observe_tools, op_tools
from agent.tracing import TraceRecord, Tracer

PEAK = {"center": [0.5, 0.5], "radius": 0.25, "height": 40}


def tool_use(name, inp, id="tu_1"):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=inp)


@pytest.fixture
def session():
    return Session.flat(33, 33, base_y=64)


def test_op_tools_mirror_registry(session):
    tools = op_tools(session)
    peak = next(t for t in tools if t.name == "raise_peak")
    assert peak.to_param()["input_schema"]["additionalProperties"] is False


def test_execute_applies_op_and_records_it(session):
    tools = {t.name: t for t in op_tools(session)}
    result = execute(tools, tool_use("raise_peak", PEAK))
    assert result["is_error"] is False and "raise_peak_1" in result["content"]
    assert session.state.height.max() == pytest.approx(104)
    assert [r.id for r in session.ops] == ["raise_peak_1"]


def test_execute_returns_op_errors_to_the_agent(session):
    tools = {t.name: t for t in op_tools(session)}
    result = execute(tools, tool_use("raise_peak", PEAK | {"radius": 2}))
    assert result["is_error"] is True and "radius" in result["content"]
    assert session.ops == [] and np.all(session.state.height == 64)  # failed op leaves no trace in state


def test_material_ops_say_what_they_changed(session):
    tools = {t.name: t for t in op_tools(session)}
    region = {"circle": {"center": [0.5, 0.5], "radius": 0.25}}
    paint = execute(tools, tool_use("paint_region", {"region": region, "block": "gravel", "id": "scree"}))
    palette = execute(tools, tool_use("set_palette_rules", {"rules": [{"use": {"kind": "single", "block": "sand"}}]}))
    for result in (paint, palette):
        assert result["is_error"] is False and "surface blocks only, no height or water change" in result["content"]
    assert "columns set to gravel" in paint["content"] and "1 palette rule;" in palette["content"]
    assert "no visible change" not in session.summary() and "scree: paint_region" in session.summary()


def test_execute_unknown_tool(session):
    result = execute({}, tool_use("nope", {}))
    assert result["is_error"] is True and "unknown tool" in result["content"]


def test_op_tool_accepts_custom_id(session):
    tools = {t.name: t for t in op_tools(session)}
    assert "id" in tools["raise_peak"].input_schema["properties"]
    result = execute(tools, tool_use("raise_peak", PEAK | {"id": "mountain_1"}))
    assert result["is_error"] is False and [r.id for r in session.ops] == ["mountain_1"]


def test_control_tools_list_update_remove(session):
    tools = {t.name: t for t in [*op_tools(session), *control_tools(session)]}
    execute(tools, tool_use("raise_peak", PEAK | {"id": "m"}))
    listing = execute(tools, tool_use("list_ops", {}))["content"]
    assert "m: raise_peak" in listing and "footprint" in listing

    r = execute(tools, tool_use("update_op", {"id": "m", "params": {"height": 80}}))
    assert r["is_error"] is False and session.state.height.max() == pytest.approx(144)
    bad = execute(tools, tool_use("update_op", {"id": "m", "params": {"radius": 3}}))
    assert bad["is_error"] is True and session.state.height.max() == pytest.approx(144)
    # JSON-encoded params (seen from Haiku) are accepted; anything else is an agent-facing error, not a crash.
    r = execute(tools, tool_use("update_op", {"id": "m", "params": '{"height": 90}'}))
    assert r["is_error"] is False and session.state.height.max() == pytest.approx(154)
    bad = execute(tools, tool_use("update_op", {"id": "m", "params": "height=100"}))
    assert bad["is_error"] is True and "must be an object" in bad["content"]

    assert execute(tools, tool_use("remove_op", {"id": "nope"}))["is_error"] is True
    assert execute(tools, tool_use("remove_op", {"id": "m"}))["is_error"] is False
    assert session.ops == [] and np.all(session.state.height == 64)


def test_image_block_is_base64_png():
    block = image_block(b"\x89PNG")
    assert block["source"]["media_type"] == "image/png" and block["source"]["data"] == "iVBORw=="


def fake_client(stop_reason="end_turn"):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="hi")],
            stop_reason=stop_reason,
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200, cache_read_input_tokens=5000, cache_creation_input_tokens=0),
            model=kwargs["model"],
            _request_id="req_1",
        )

    messages = SimpleNamespace(create=create)
    return SimpleNamespace(messages=messages, beta=SimpleNamespace(messages=messages)), calls


def test_adapter_normalizes_response_and_sets_request_params():
    client, calls = fake_client()
    r = AnthropicAdapter(ModelConfig(model="claude-haiku-5-5", effort="low"), client=client).complete(
        [{"role": "user", "content": "x"}], system="sys"
    )
    assert r.text == "hi" and r.stop_reason == "end_turn" and r.request_id == "req_1"
    assert r.usage == Usage(1000, 200, 5000, 0)
    assert calls[0]["output_config"] == {"effort": "low"} and "fallbacks" not in calls[0]  # no fallback for Haiku


def test_adapter_uses_refusal_fallback_on_supported_models():
    client, calls = fake_client()
    AnthropicAdapter(ModelConfig(model="claude-opus-5-5"), client=client).complete([{"role": "user", "content": "x"}])
    assert calls[0]["fallbacks"] == "default" and calls[0]["betas"] == ["server-side-fallback-2026-07-01"]


def test_usage_cost():
    # Opus 5.5: 1M input @ $4 + 1M output @ $20
    assert Usage(1_000_000, 1_000_000).cost_usd("claude-opus-5-5") == pytest.approx(24.0)
    assert Usage(1).cost_usd("unknown-model") is None


def test_budget_trips_on_steps_and_cost():
    t = BudgetTracker(Budget(max_steps=2, max_cost_usd=1.0))
    t.charge(Usage(10, 10), 0.4)
    assert t.exceeded() is None
    t.charge(Usage(10, 10), 0.4)
    assert "step budget" in t.exceeded()
    t2 = BudgetTracker(Budget(max_steps=99, max_cost_usd=1.0))
    t2.charge(Usage(), 1.5)
    assert "cost budget" in t2.exceeded()


def test_tracer_writes_one_json_line_per_record(tmp_path):
    tracer = Tracer(run_id="t", root=tmp_path)
    tracer.log(TraceRecord(run_id="t", stage="terrain", step=1, usage={"input_tokens": 5}))
    tracer.log(TraceRecord(run_id="t", stage="terrain", step=2))
    lines = tracer.path.read_text().splitlines()
    assert [json.loads(l)["step"] for l in lines] == [1, 2]


# ---------------------------------------------------------------- feature stats and render_view


def test_op_result_has_feature_stats_but_summary_stays_compact(session):
    tools = {t.name: t for t in op_tools(session)}
    result = execute(tools, tool_use("raise_peak", PEAK | {"id": "hill"}))["content"]
    assert "top y=104 at (u=0.50, v=0.50)" in result and "slope max" in result and "columns" in result
    assert "slope max" not in session.summary() and "hill: raise_peak" in session.summary()


def test_update_op_reports_the_changed_feature(session):
    tools = {t.name: t for t in [*op_tools(session), *control_tools(session)]}
    execute(tools, tool_use("raise_peak", PEAK | {"id": "hill"}))
    result = execute(tools, tool_use("update_op", {"id": "hill", "params": {"height": 20}}))["content"]
    assert "hill now: footprint" in result and "top y=84" in result


def view(tools, **inp):
    return execute(tools, tool_use("render_view", inp))


@pytest.fixture
def viewer(session, tmp_path):
    tools = {t.name: t for t in [*op_tools(session), *observe_tools(session, tmp_path, max_views=3)]}
    return tools, tmp_path


def test_render_view_returns_text_and_png_and_saves_it(session, viewer):
    tools, out = viewer
    assert view(tools)["is_error"] is True  # nothing built yet
    execute(tools, tool_use("raise_peak", PEAK | {"id": "hill"}))
    r = view(tools)
    text_part, image_part = r["content"]
    assert r["is_error"] is False and "map view of the terrain after 1 ops" in text_part["text"]
    assert image_part["type"] == "image" and image_part["source"]["media_type"] == "image/png"
    assert (out / "view_001.png").read_bytes()[1:4] == b"PNG"


def test_render_view_3d_and_feature_close_up(session, viewer):
    tools, _ = viewer
    execute(tools, tool_use("raise_peak", PEAK | {"id": "hill"}))
    assert view(tools, view="3d_sw")["is_error"] is False
    close = view(tools, view="3d_ne", feature="hill")
    assert close["is_error"] is False and "3D close-up of hill" in close["content"][0]["text"]
    bad = view(tools, view="3d_ne", feature="nope")
    assert bad["is_error"] is True and "unknown feature 'nope'" in bad["content"] and "hill" in bad["content"]


def test_render_view_refuses_unchanged_repeats_and_caps_views(session, viewer):
    tools, _ = viewer
    execute(tools, tool_use("raise_peak", PEAK | {"id": "hill"}))
    assert view(tools)["is_error"] is False
    repeat = view(tools)
    assert repeat["is_error"] is True and "unchanged" in repeat["content"]
    execute(tools, tool_use("raise_peak", PEAK | {"id": "hill_2", "center": [0.2, 0.2]}))
    assert view(tools)["is_error"] is False  # allowed again after an edit
    assert view(tools, view="3d_sw")["is_error"] is False  # third and last view
    execute(tools, tool_use("smooth", {"strength": 0.5}))
    capped = view(tools)
    assert capped["is_error"] is True and "view limit reached (3 per run)" in capped["content"]
