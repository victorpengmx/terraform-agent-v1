"""Spec checklist (DESIGN §11.3): code-checked items on synthetic terrains, critic wiring, gate rule 2."""

import json

import pytest

from agent.budget import Budget
from agent.checklist import checklist, summary
from agent.critic import Critic
from agent.harness import run_task
from agent.session import Session
from agent.spec import FeatureGraph
from agent.tools import control_tools, op_tools
from agent.tracing import Tracer
from fakes import PEAK, ScriptedModel, compare_json, critique_json, feature, graph, response, text, tool_use

RIVER = {"path": [[0.5, 0.0], [0.5, 1.0]], "width": 0.06, "depth": 3}


def items(g: dict, session: Session) -> dict:
    return {i.id: i for i in checklist(FeatureGraph.model_validate(g), session.stack)}


@pytest.fixture
def s():
    return Session.flat(65, 65, base_y=64)


def test_untagged_feature_fails_exists_and_skips_what_depends_on_it(s):
    s.apply("raise_peak", PEAK)
    got = items(graph(feature("peak_1", x="left", slope="steep"), feature("hill_1", "hills"),
                      relations=[{"subject": "peak_1", "relation": "taller_than", "object": "hill_1"}]), s)
    assert got["peak_1.exists"].status == "fail" and "pass feature='peak_1'" in got["peak_1.exists"].detail
    assert got["peak_1.position"].status == got["peak_1.slope"].status == "skipped"
    assert got["peak_1.taller_than.hill_1"].status == "skipped"


def test_position_band(s):
    s.apply("raise_peak", PEAK | {"center": [0.2, 0.2], "radius": 0.15}, feature="peak_1")
    assert items(graph(feature("peak_1", x="left", y="back")), s)["peak_1.position"].status == "pass"
    bad = items(graph(feature("peak_1", x="right", y="front")), s)["peak_1.position"]
    assert bad.status == "fail" and "centre at u=0.2" in bad.detail


@pytest.mark.parametrize("spec_slope, status, severity", [("gentle", "pass", "major"), ("moderate", "fail", "minor"),
                                                          ("steep", "fail", "major")])
def test_slope_category_with_one_category_of_slack(s, spec_slope, status, severity):
    s.apply("raise_peak", PEAK | {"radius": 0.45, "height": 4, "character": "gentle"}, feature="hill_1")
    item = items(graph(feature("hill_1", "hills", slope=spec_slope)), s)["hill_1.slope"]
    assert (item.status, item.severity if status == "fail" else "major") == (status, severity)


@pytest.mark.parametrize("meander, spec, status", [(0.0, "almost straight", "pass"), (0.0, "twisty", "fail"),
                                                   (0.6, "twisty", "pass")])
def test_winding(meander, spec, status):
    s = Session.flat(256, 256, base_y=64)
    s.apply("carve_river", RIVER | {"width": 0.03, "meander": meander}, feature="river_1")
    item = items(graph(feature("river_1", "river", winding=spec)), s)["river_1.winding"]
    assert item.status == status and ("raise meander" in item.detail) == (status == "fail")


def test_tallest_and_height_relations(s):
    s.apply("raise_peak", PEAK | {"center": [0.25, 0.5], "radius": 0.15, "height": 30}, feature="peak_1")
    s.apply("raise_peak", PEAK | {"center": [0.75, 0.5], "radius": 0.15, "height": 10}, feature="hill_1")
    g = graph(feature("peak_1", tallest=True), feature("hill_1", "hills"),
              relations=[{"subject": "peak_1", "relation": "taller_than", "object": "hill_1"},
                         {"subject": "peak_1", "relation": "left_of", "object": "hill_1"},
                         {"subject": "hill_1", "relation": "left_of", "object": "peak_1"}])
    got = items(g, s)
    assert got["peak_1.tallest"].status == got["peak_1.taller_than.hill_1"].status == got["peak_1.left_of.hill_1"].status == "pass"
    assert got["hill_1.left_of.peak_1"].status == "fail"


def test_river_relations(s):
    s.apply("raise_peak", PEAK | {"center": [0.5, 0.1], "radius": 0.15}, feature="peak_1")
    s.apply("carve_river", RIVER | {"path": [[0.5, 0.1], [0.5, 1.0]]}, feature="river_1")
    g = graph(feature("peak_1"), feature("river_1", "river"),
              relations=[{"subject": "river_1", "relation": "flows_from", "object": "peak_1"},
                         {"subject": "river_1", "relation": "reaches", "object": "south_edge"},
                         {"subject": "river_1", "relation": "reaches", "object": "north_edge"},
                         {"subject": "peak_1", "relation": "reaches", "object": "south_edge"}])
    got = items(g, s)
    assert got["river_1.flows_from.peak_1"].status == got["river_1.reaches.south_edge"].status == "pass"
    assert got["river_1.reaches.north_edge"].status == "fail"
    assert "no carve_river op" in got["peak_1.reaches.south_edge"].detail


def test_summary_counts_and_lists_failures(s):
    s.apply("raise_peak", PEAK, feature="peak_1")
    text_ = summary(checklist(FeatureGraph.model_validate(graph(feature("peak_1", x="right", y="front"), feature("lake_1", "lake"))), s.stack))
    assert text_.startswith("Spec checklist: 1 of 3 checked items pass") and "lake_1 (lake) exists" in text_


def test_critic_sees_the_checklist_and_major_failures_force_revise(s):
    s.apply("raise_peak", PEAK, feature="peak_1")
    model = ScriptedModel(critique_json(5, "pass"))
    spec = {"prompt": "x", "graph": FeatureGraph.model_validate(graph(feature("peak_1"), feature("lake_1", "lake")))}
    result, _ = Critic(model).critique(spec, s)
    assert result.verdict == "revise" and result.checklist_passes == 1
    assert "lake_1 (lake) exists" in model.calls[0]["messages"][0]["content"][-1]["text"]
    assert "Spec checklist: 1 of 2" in result.as_text()


def test_gate_rule_2_rolls_back_a_lower_checklist_pass_count(s, tmp_path):
    g = FeatureGraph.model_validate(graph(feature("peak_1")))
    builder = ScriptedModel(
        response(tool_use("raise_peak", PEAK | {"id": "m", "feature": "peak_1"}), stop="tool_use"), response(text("r1")),
        response(tool_use("update_op", {"id": "m", "params": {"feature": None}}, id="tu_2"), stop="tool_use"), response(text("r2")),
    )
    critic_model = ScriptedModel(critique_json(3), critique_json(5), compare_json("2"))
    tools = {t.name: t for t in [*op_tools(s), *control_tools(s)]}
    result = run_task(prompt="a peak", images=[], session=s, tools=tools, model=builder, graph=g, critic=Critic(critic_model),
                      tracer=Tracer(run_id="t", root=tmp_path), budget=Budget(), max_rounds=2)
    assert result.accepted_round == 1 and s.ops[0].feature == "peak_1"
    assert len(critic_model.calls) == 2  # rule 2 decided before any comparison
    notes = [json.loads(l).get("note") or "" for l in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert any("rolled back round 2: it passes 0 spec checklist items, the accepted attempt 1" in n for n in notes)


def test_wall_defined_features_are_judged_by_their_walls(s):
    s.apply("raise_region", {"region": {"polygon": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]}, "height": 25,
                             "mode": "plateau", "edge": 0.01}, feature="mesa_1")
    item = items(graph(feature("mesa_1", "mesa", slope="cliff")), s)["mesa_1.slope"]
    assert item.status == "pass" and "wall slope (90th percentile)" in item.detail


def test_a_valley_or_canyon_can_be_checked_for_winding(s):
    s.apply("carve_valley", {"path": [[0.5, 0.0], [0.3, 0.3], [0.7, 0.6], [0.4, 1.0]], "width": 0.15, "depth": 8},
            feature="canyon_1")
    assert items(graph(feature("canyon_1", "canyon", winding="twisty")), s)["canyon_1.winding"].status == "pass"
    straight = items(graph(feature("canyon_1", "canyon", winding="almost straight")), s)["canyon_1.winding"]
    assert straight.status == "fail" and straight.severity == "major" and "sinuosity 1.39 (twisty)" in straight.detail
    assert "straighten its carve_valley path" in straight.detail  # the hint points the right way
    more = items(graph(feature("canyon_1", "canyon", winding="meandering")), s)["canyon_1.winding"]
    assert "add bends" in more.detail


def test_gate_rule_2_keeps_a_higher_checklist_pass_count_without_a_comparison(s, tmp_path):
    g = FeatureGraph.model_validate(graph(feature("peak_1")))
    builder = ScriptedModel(
        response(tool_use("raise_peak", PEAK | {"id": "m"}), stop="tool_use"), response(text("r1")),
        response(tool_use("update_op", {"id": "m", "params": {"feature": "peak_1"}}, id="tu_2"), stop="tool_use"), response(text("r2")),
    )
    critic_model = ScriptedModel(critique_json(4), critique_json(3))  # no comparison scripted: it must not be asked
    tools = {t.name: t for t in [*op_tools(s), *control_tools(s)]}
    result = run_task(prompt="a peak", images=[], session=s, tools=tools, model=builder, graph=g, critic=Critic(critic_model),
                      tracer=Tracer(run_id="t", root=tmp_path), budget=Budget(), max_rounds=2)
    assert result.accepted_round == 2 and len(critic_model.calls) == 2
    notes = [json.loads(l).get("note") or "" for l in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert any("accepted round 2: it passes 1 spec checklist items, the accepted attempt 0; newly met: peak_1 (peak) exists" in n
               for n in notes)
