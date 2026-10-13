"""Spec feature tags on ops and per-feature measurement over the union of their footprints (DESIGN §4.3)."""

import pytest

from agent.session import Session
from agent.tools import control_tools, execute, op_tools
from fakes import PEAK, tool_use
from terrain.features import measure_spec_feature
from terrain.ops import OpError
from terrain.opstack import OpStack


@pytest.fixture
def session():
    return Session.flat(65, 65, base_y=64)


def test_feature_tag_is_stored_shown_and_replayed(session):
    tools = {t.name: t for t in [*op_tools(session), *control_tools(session)]}
    r = execute(tools, tool_use("raise_peak", PEAK | {"id": "m", "feature": "peak_1"}))
    assert r["is_error"] is False and session.ops[0].feature == "peak_1"
    assert "m [feature=peak_1]: raise_peak" in session.summary()
    assert "feature" not in session.ops[0].params  # a tag, not an op parameter
    again = OpStack.from_json(session.stack.to_json())
    assert again.entries[0].feature == "peak_1"


def test_untagged_stacks_serialize_as_before(session):
    session.apply("raise_peak", PEAK)
    assert '"feature"' not in session.stack.to_json()


def test_update_op_retags_and_keeps_the_terrain(session):
    tools = {t.name: t for t in [*op_tools(session), *control_tools(session)]}
    execute(tools, tool_use("raise_peak", PEAK | {"id": "m"}))
    before = session.state.height.copy()
    r = execute(tools, tool_use("update_op", {"id": "m", "params": {"feature": "hill_1"}}))
    assert r["is_error"] is False and session.ops[0].feature == "hill_1" and (session.state.height == before).all()
    execute(tools, tool_use("update_op", {"id": "m", "params": {"height": 20}}))
    assert session.ops[0].feature == "hill_1"  # keeps its tag through a parameter change


def test_unknown_spec_feature_is_an_agent_error(session):
    session.spec_feature_ids = {"peak_1", "river_1"}
    with pytest.raises(OpError, match="unknown spec feature 'peak_9'; the spec's features are: peak_1, river_1"):
        session.apply("raise_peak", PEAK, feature="peak_9")
    assert session.ops == []
    with pytest.raises(OpError, match="invalid feature"):
        Session.flat(9, 9).apply("raise_peak", PEAK, feature="Peak 1")


def test_spec_feature_is_measured_over_the_union_of_its_ops(session):
    session.apply("raise_peak", PEAK | {"center": [0.3, 0.5], "radius": 0.15, "height": 20}, id="west", feature="hills_1")
    session.apply("raise_peak", PEAK | {"center": [0.7, 0.5], "radius": 0.15, "height": 30}, id="east", feature="hills_1")
    a = measure_spec_feature(session.stack, "hills_1", "hills", ["west", "east"])
    assert a.top_y == pytest.approx(94, abs=0.5) and a.top_uv[0] > 0.6  # the east summit is the higher one
    assert 0.45 < a.centroid_uv[0] < 0.6 and a.bounds[0] < 0.2 and a.bounds[2] > 0.8
    assert a.prominence is not None and measure_spec_feature(session.stack, "x", "hills", ["missing"]) is None


def test_river_source_and_mouth(session):
    session.apply("carve_river", {"path": [[0.5, 0.0], [0.5, 1.0]], "width": 0.06, "depth": 3}, id="r", feature="river_1")
    a = measure_spec_feature(session.stack, "river_1", "river", ["r"])
    assert a.source_uv[1] == pytest.approx(0.0, abs=0.02) and a.mouth_uv[1] == pytest.approx(1.0, abs=0.02)
    assert a.winding == "almost straight" and set(a.edges) >= {"n", "s"}


def test_regions_and_palette_rules_can_name_a_spec_feature(session):
    session.apply("raise_peak", PEAK | {"center": [0.3, 0.5], "radius": 0.15}, id="west", feature="hills_1")
    session.apply("raise_peak", PEAK | {"center": [0.7, 0.5], "radius": 0.15}, id="east", feature="hills_1")
    session.apply("roughen", {"region": {"feature": "hills_1"}, "amount": 4}, id="crags")
    changed = session.state.masks["feature:crags"] > 0.02
    assert changed[32, 19] and changed[32, 45] and not changed[5, 5]  # both hills, nothing far away
    session.apply("set_palette_rules", {"rules": [{"when": {"feature": "hills_1"}, "use": {"kind": "single", "block": "tuff"}},
                                                  {"use": {"kind": "single", "block": "grass_block"}}]})
    with pytest.raises(OpError, match="known: crags, east, hills_1, west"):
        session.apply("smooth", {"region": {"feature": "nope"}, "strength": 0.5})
