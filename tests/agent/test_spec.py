"""Feature graph (DESIGN §9.3): schema, validation, projection to (u, v) bands, the SPEC call."""

import pytest

from agent.spec import FeatureGraph, Position, SpecError, bands, graph_text, write_spec
from fakes import ScriptedModel, feature, graph, response, spec_json, text
from test_critic_context import _walk


def test_graph_schema_fits_structured_output_limits():
    for path, node in _walk(FeatureGraph.model_json_schema()):
        assert not {"minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems", "pattern"} & node.keys(), path
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False, path


def test_problems_catch_bad_ids_unknown_references_and_size():
    good = FeatureGraph.model_validate(graph(feature("peak_1", tallest=True), feature("river_1", "river"),
                                             relations=[{"subject": "river_1", "relation": "reaches", "object": "south_edge"}]))
    assert good.problems() == []
    bad = FeatureGraph.model_validate(graph(
        feature("Peak 1"), feature("lake", "lake", part_of="nowhere"), feature("lake", "lake"),
        relations=[{"subject": "river_9", "relation": "left_of", "object": "south_edge"}]))
    problems = " | ".join(bad.problems())
    for fragment in ["invalid id 'Peak 1'", "duplicate id 'lake'", "part_of unknown feature 'nowhere'",
                     "subject 'river_9'", "object 'south_edge' is not a feature"]:
        assert fragment in problems
    too_many = FeatureGraph.model_validate(graph(*[feature(f"hill_{i}", "hills") for i in range(16)]))
    assert "16 features" in too_many.problems()[0]


def test_text_positions_follow_the_south_edge_viewpoint():
    assert bands(Position(x="left", y="back", image_x=None, depth=None)) == (0.0, 0.45, 0.0, 0.45)  # north-west
    assert bands(Position(x="right", y="front", image_x=None, depth=None)) == (0.55, 1.0, 0.55, 1.0)  # south-east
    assert bands(Position(x="any", y=None, image_x=None, depth=None)) == (0.0, 1.0, 0.0, 1.0)


def test_image_positions_widen_with_distance():
    near = bands(Position(x=None, y=None, image_x="left", depth="near"))
    far = bands(Position(x=None, y=None, image_x="left", depth="far"))
    assert near[2:] == (0.6, 1.0) and far[2:] == (0.0, 0.4)
    assert far[0] < near[0]  # "left" in the distance is further west than "left" up close
    assert far[1] - far[0] > near[1] - near[0]
    assert bands(Position(x=None, y=None, image_x="center", depth="near"))[:2] == (0.38, 0.62)


def test_graph_text_shows_bands_relations_and_unsupported():
    g = FeatureGraph.model_validate(graph(feature("peak_1", x="left", y="back", slope="steep", tallest=True),
                                          feature("river_1", "river", winding="winding"),
                                          relations=[{"subject": "river_1", "relation": "flows_from", "object": "peak_1"}])
                                    | {"unsupported": ["a castle"]})
    t = graph_text(g)
    assert "peak_1 (peak): u 0.00-0.45, v 0.00-0.45; slope steep, tallest" in t
    assert "river_1 (river): anywhere; winding winding" in t
    assert "river_1 flows_from peak_1" in t and "a castle" in t


def test_write_spec_parses_and_validates():
    g = graph(feature("peak_1"), feature("river_1", "river"))
    model = ScriptedModel(spec_json(g))
    parsed, _ = write_spec(model, "a peak and a river")
    assert parsed.ids() == {"peak_1", "river_1"}
    assert model.calls[0]["response_schema"]["additionalProperties"] is False


@pytest.mark.parametrize("bad, message", [
    (response(text("not json")), "invalid JSON"),
    (response(text("{}"), stop="max_tokens"), "max_tokens"),
    (spec_json(graph(feature("a"), feature("a"))), "duplicate id"),
])
def test_unusable_spec_fails_loudly_with_its_response(bad, message):
    with pytest.raises(SpecError, match=message) as e:
        write_spec(ScriptedModel(bad), "x")
    assert e.value.response is bad


def test_images_go_first_and_ask_for_image_positions():
    model = ScriptedModel(spec_json(graph(feature("peak_1", x=None, y=None, position={"image_x": "left", "depth": "far"}))))
    write_spec(model, "", images=[b"\x89PNG fake"])
    blocks = model.calls[0]["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["image", "text"] and "image positions" in blocks[1]["text"]


def test_image_spans_are_relative_to_the_frame_at_that_depth():
    from agent.spec import SpecFeature, span_classes

    def f(depth):
        return SpecFeature.model_validate(feature("r", "mountain_range", x=None, y=None, position={"image_x": "center", "depth": depth}))

    far, near = dict(span_classes(f("far"), "x")), dict(span_classes(f("near"), "x"))
    assert far["wide"] == pytest.approx(2 / 3) and near["wide"] == pytest.approx(1 / 3)  # near frame = half the area
    assert dict(span_classes(f("near"), "y"))["wide"] == pytest.approx(2 / 3)  # far-to-near is not scaled
    text = SpecFeature.model_validate(feature("r", "mountain_range", span_x="full"))
    assert dict(span_classes(text, "x"))["narrow"] == pytest.approx(1 / 3)


def test_graph_text_shows_spans():
    g = FeatureGraph.model_validate(graph(feature("range_1", "mountain_range", y="back", span_x="full")))
    assert "range_1 (mountain_range): u 0.00-1.00, v 0.00-0.45; spans full west-east" in graph_text(g)
