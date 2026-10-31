"""SPEC quality (E5): the agent's own graph vs the answer key (eval/spec_quality.py)."""

from agent.spec import FeatureGraph
from eval.spec_quality import compare, match_graphs, summarize
from tests.agent.fakes import feature, graph


def g(*features, relations=None) -> FeatureGraph:
    return FeatureGraph.model_validate(graph(*features, relations=relations))


KEY = g(feature("mountain_west", "peak", x="left"), feature("mountain_east", "peak", x="right"),
        feature("valley_1", "valley", x="center", slope="steep"), feature("river_1", "river", winding="winding"),
        relations=[{"subject": "mountain_west", "relation": "taller_than", "object": "mountain_east"},
                   {"subject": "river_1", "relation": "reaches", "object": "south_edge"}])


def test_a_renamed_but_equal_graph_matches_fully():
    agent = g(feature("a", "peak", x="left"), feature("b", "peak", x="right"), feature("c", "canyon", x="center", slope="steep"),
              feature("d", "river", winding="winding"),
              relations=[{"subject": "b", "relation": "lower_than", "object": "a"},  # the inverse form counts
                         {"subject": "d", "relation": "reaches", "object": "south_edge"}])
    r = compare(KEY, agent)
    assert r["matches"] == {"mountain_west": "a", "mountain_east": "b", "valley_1": "c", "river_1": "d"}  # valley ~ canyon
    assert (r["matched"], r["relations_found"]) == (4, 2)
    assert r["categories"]["slope"] == {"n": 1, "exact": 1, "within_one": 1}
    assert r["categories"]["winding"] == {"n": 1, "exact": 1, "within_one": 1}


def test_missing_features_wrong_sides_and_categories():
    agent = g(feature("a", "peak", x="right"), feature("c", "valley", x="center", slope="gentle"),
              feature("d", "river", winding="twisty"), feature("extra", "lake"))
    r = compare(KEY, agent)
    # "a" is on the right: it can only be the eastern peak; the western one is not found.
    assert r["matches"] == {"mountain_east": "a", "valley_1": "c", "river_1": "d"}
    assert r["categories"]["slope"] == {"n": 1, "exact": 0, "within_one": 0}  # steep vs gentle: two off
    assert r["categories"]["winding"] == {"n": 1, "exact": 0, "within_one": 1}
    assert r["relations_found"] == 0  # taller_than needs the western peak; reaches isn't stated
    s = summarize([r])
    assert s["recall"] == 3 / 4 and s["precision"] == 3 / 4 and s["relation_recall"] == 0


def test_incompatible_types_never_match():
    assert match_graphs(g(feature("l", "lake")), g(feature("p", "peak"))) == {}
    assert match_graphs(g(feature("m", "mesa")), g(feature("p", "plateau"))) == {"m": "p"}
    assert match_graphs(g(feature("s", "other")), g(feature("p", "peak"))) == {"s": "p"}


def test_image_positions_compare_through_their_bands():
    key = g(feature("near_lake", "lake", x=None, y=None, position={"image_x": "left", "depth": "near"}))
    near = g(feature("l", "lake", x=None, y=None, position={"image_x": "left", "depth": "near"}))
    far = g(feature("l", "lake", x=None, y=None, position={"image_x": "right", "depth": "far"}))
    assert match_graphs(key, near) == {"near_lake": "l"} and match_graphs(key, far) == {}
