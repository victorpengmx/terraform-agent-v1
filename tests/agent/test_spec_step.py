"""SPEC step in the harness (DESIGN §9.3): written first, pinned in every episode, given to the critic."""

import json

from agent.budget import Budget
from agent.critic import Critic
from agent.harness import run_task
from agent.session import Session
from agent.spec import FeatureGraph
from agent.tools import op_tools
from agent.tracing import Tracer
from fakes import PEAK, ScriptedModel, critique_json, feature, graph, response, spec_json, text, tool_use

GRAPH = graph(feature("peak_1", x="left", y="back", tallest=True), feature("river_1", "river", winding="winding"),
              relations=[{"subject": "river_1", "relation": "reaches", "object": "south_edge"}])


def run(model, tmp_path, session=None, **kw):
    session = session or Session.flat(33, 33, base_y=64)
    tools = {t.name: t for t in op_tools(session)}
    result = run_task(prompt="a peak and a river", images=[], session=session, tools=tools, model=model,
                      tracer=Tracer(run_id="t", root=tmp_path), budget=Budget(), **kw)
    return result, session, [json.loads(l) for l in (tmp_path / "t.jsonl").read_text().splitlines()]


def test_spec_is_written_first_and_pinned_for_builder_and_critic(tmp_path):
    builder = ScriptedModel(spec_json(GRAPH), response(tool_use("raise_peak", PEAK), stop="tool_use"), response(text("done")))
    critic_model = ScriptedModel(critique_json(4, "pass"))
    result, session, trace = run(builder, tmp_path, critic=Critic(critic_model), max_rounds=1)
    assert builder.calls[0]["tools"] is None and "a peak and a river" in builder.calls[0]["messages"][0]["content"][-1]["text"]
    episode = builder.calls[1]["messages"][0]["content"][-1]["text"]
    assert "peak_1 (peak): u 0.00-0.45, v 0.00-0.45; tallest" in episode and "river_1 reaches south_edge" in episode
    assert "peak_1 (peak)" in critic_model.calls[0]["messages"][0]["content"][-1]["text"]
    assert result.graph.ids() == {"peak_1", "river_1"} and session.spec_feature_ids == {"peak_1", "river_1"}
    assert trace[0]["stage"] == "spec" and trace[0]["scores"]["graph"]["features"][0]["id"] == "peak_1"


def test_spec_failing_twice_builds_from_the_raw_prompt(tmp_path):
    bad = response(text("not json"))
    builder = ScriptedModel(bad, bad, response(text("done")))
    result, _, trace = run(builder, tmp_path)
    assert result.graph is None and result.stop == "done"
    assert [r["note"].split(":")[0] for r in trace[:3]] == ["SPEC failed (try 1 of 2)", "SPEC failed (try 2 of 2)",
                                                            "SPEC failed twice; building from the raw prompt without a checklist"]
    assert "Features (" not in builder.calls[2]["messages"][0]["content"][-1]["text"]


def test_no_spec_skips_the_step(tmp_path):
    builder = ScriptedModel(response(text("done")))
    result, session, trace = run(builder, tmp_path, use_spec=False)
    assert result.graph is None and len(builder.calls) == 1 and session.spec_feature_ids is None
    assert not any(r["stage"] == "spec" for r in trace)


def test_a_given_graph_is_used_without_a_spec_call(tmp_path):
    builder = ScriptedModel(response(text("done")))
    result, _, _ = run(builder, tmp_path, graph=FeatureGraph.model_validate(GRAPH))
    assert len(builder.calls) == 1 and "river_1 (river)" in builder.calls[0]["messages"][0]["content"][-1]["text"]


def test_reference_image_spec_projects_image_positions(tmp_path):
    image_graph = graph(feature("peak_1", x=None, y=None, position={"image_x": "left", "depth": "far"}),
                        feature("lake_1", "lake", x=None, y=None, position={"image_x": "center", "depth": "near"}))
    builder = ScriptedModel(spec_json(image_graph), response(text("done")))
    session = Session.flat(33, 33, base_y=64)
    tools = {t.name: t for t in op_tools(session)}
    run_task(prompt="", images=[b"\x89PNG fake"], session=session, tools=tools, model=builder,
             tracer=Tracer(run_id="t", root=tmp_path), budget=Budget())
    assert [b["type"] for b in builder.calls[0]["messages"][0]["content"]] == ["image", "text"]  # the SPEC call
    episode = builder.calls[1]["messages"][0]["content"]
    assert episode[0]["type"] == "image"  # the reference stays in front of the builder
    assert "peak_1 (peak): u 0.00-0.35, v 0.00-0.40" in episode[1]["text"]  # far left: the wide end of the frame
    assert "lake_1 (lake): u 0.38-0.62, v 0.60-1.00" in episode[1]["text"]
