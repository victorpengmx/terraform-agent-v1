"""Tag-independent matching for evaluation (eval/match.py, DESIGN §16): tags are ignored, landforms are found
by type, position and size, and the checklist runs on the matches."""

import pytest

from agent.checklist import checklist
from agent.session import Session
from agent.spec import FeatureGraph
from eval.match import match, match_checklist
from tests.agent.fakes import PEAK, feature, graph

RIVER = {"path": [[0.6, 0.1], [0.6, 1.0]], "width": 0.06, "depth": 3}


def by_id(items):
    return {i.id: i for i in items}


def g(*features, relations=None) -> FeatureGraph:
    return FeatureGraph.model_validate(graph(*features, relations=relations))


@pytest.fixture
def s():
    return Session.flat(65, 65, base_y=64)


def build_peak_river_lake(s, tags=(None, None, None)):
    s.apply("raise_peak", PEAK | {"center": [0.2, 0.3], "radius": 0.15}, id="cone", feature=tags[0])
    s.apply("carve_river", RIVER, id="stream", feature=tags[1])
    s.apply("raise_region", {"region": {"circle": {"center": [0.85, 0.6], "radius": 0.1}}, "height": -4, "mode": "basin"}, id="hole")
    s.apply("fill_lake", {"at": [0.85, 0.6]}, id="water", feature=tags[2])


PEAK_RIVER_LAKE = (feature("peak_1", x="left"), feature("river_1", "river"), feature("lake_1", "lake", x="right"))
REACHES_SOUTH = [{"subject": "river_1", "relation": "reaches", "object": "south_edge"}]


def test_an_untagged_correct_build_passes_in_matcher_mode(s):
    build_peak_river_lake(s)
    spec = g(*PEAK_RIVER_LAKE, relations=REACHES_SOUTH)
    assert match(spec, s.stack).ops == {"peak_1": ["cone"], "river_1": ["stream"], "lake_1": ["water"]}
    matched = by_id(match_checklist(spec, s.stack))
    assert all(i.status == "pass" for i in matched.values()), [i for i in matched.values() if i.status != "pass"]
    tagged = by_id(checklist(spec, s.stack))  # the in-loop mode needs tags
    assert {tagged[f"{f}.exists"].status for f in ("peak_1", "river_1", "lake_1")} == {"fail"}


def test_a_global_op_tagged_with_a_feature_does_not_satisfy_a_peak_or_river(s):
    s.apply("raise_region", {"region": {"polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}, "height": 10, "edge": 0}, feature="peak_1")
    s.apply("noise_base", {"amplitude": 4}, feature="river_1")
    spec = g(feature("peak_1"), feature("river_1", "river"))
    tagged = by_id(checklist(spec, s.stack))
    assert tagged["peak_1.exists"].status == tagged["river_1.exists"].status == "pass"  # tags can be gamed
    matched = by_id(match_checklist(spec, s.stack))
    assert matched["peak_1.exists"].status == matched["river_1.exists"].status == "fail"
    assert "cover most of the area" in matched["peak_1.exists"].detail
    assert "no op builds a river" in matched["river_1.exists"].detail


def test_a_hand_written_graph_with_other_ids_matches(s):
    build_peak_river_lake(s, tags=("peak_1", "river_1", None))  # the builder's SPEC ids
    spec = g(feature("summit", x="left"), feature("stream_a", "river"),
             relations=[{"subject": "stream_a", "relation": "reaches", "object": "south_edge"}])
    assert match(spec, s.stack).ops == {"summit": ["cone"], "stream_a": ["stream"]}
    items = by_id(match_checklist(spec, s.stack))
    assert items["summit.exists"].status == items["stream_a.reaches.south_edge"].status == "pass"


def test_two_peaks_are_assigned_by_position(s):
    s.apply("raise_peak", PEAK | {"center": [0.2, 0.5], "radius": 0.12}, id="a")
    s.apply("raise_peak", PEAK | {"center": [0.8, 0.5], "radius": 0.12}, id="b")
    assert match(g(feature("east", x="right"), feature("west", x="left")), s.stack).ops == {"east": ["b"], "west": ["a"]}


def test_a_full_width_range_of_three_cones_matches_as_a_union():
    s = Session.flat(129, 129, base_y=64)
    for i, u in enumerate((0.15, 0.5, 0.85)):
        s.apply("raise_peak", PEAK | {"center": [u, 0.2], "radius": 0.12, "height": 30}, id=f"cone_{i}")
    spec = g(feature("range_1", "mountain_range", y="back", span_x="full"))
    assert match(spec, s.stack).ops["range_1"] == ["cone_0", "cone_1", "cone_2"]
    assert by_id(match_checklist(spec, s.stack))["range_1.span_x"].status == "pass"


def test_a_parent_contains_its_parts():
    s = Session.flat(129, 129, base_y=64)
    s.apply("raise_ridge", {"path": [[0.05, 0.2], [0.95, 0.2]], "width": 0.3, "height": 50}, id="spine")
    s.apply("raise_peak", PEAK | {"center": [0.5, 0.2], "radius": 0.1, "height": 10}, id="top")
    ops = match(g(feature("range_1", "mountain_range"), feature("peak_1", part_of="range_1")), s.stack).ops
    assert ops == {"range_1": ["spine", "top"], "peak_1": ["top"]}


@pytest.mark.parametrize("u, exists, position", [(0.85, "fail", "skipped"), (0.55, "pass", "fail"), (0.25, "pass", "pass")])
def test_wrong_side_is_unmatched_and_a_near_miss_fails_position(s, u, exists, position):
    s.apply("raise_peak", PEAK | {"center": [u, 0.5], "radius": 0.1})
    items = by_id(match_checklist(g(feature("peak_1", x="left")), s.stack))
    assert (items["peak_1.exists"].status, items["peak_1.position"].status) == (exists, position)
    if exists == "fail":
        assert "no compatible landform has its centre near u 0.00-0.45" in items["peak_1.exists"].detail
