"""The shared `region` type (terrain/ops/region.py): selection shapes, fades, errors and schema."""

import numpy as np
import pytest

from terrain.ops import OpError, apply_op, get_op
from terrain.brushes import to_blocks
from terrain.ops.region import (
    PATH_WOBBLE,
    CircleRegion,
    FeatureRegion,
    PathRegion,
    PolygonRegion,
    _smooth_points,
    region_shape,
    region_weight,
)
from terrain.opstack import BaseSpec, OpStack
from terrain.state import WorldState

N = 65  # 64 blocks per normalized unit
BASE = 64.0
SQUARE = [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]


def flat():
    return WorldState.flat(N, N, base_y=BASE)


def raise_(region, **kw):
    return apply_op(flat(), "raise_region", {"region": region, "height": 10, "edge": 0, **kw})


def test_circle_radius_is_a_fraction_of_the_shorter_side():
    inside, _ = region_shape(flat(), CircleRegion(circle={"center": [0.5, 0.5], "radius": 0.25}))
    assert inside[32, 32 + 16] and not inside[32, 32 + 17]  # 0.25 * 64 = 16 blocks


def test_path_width_is_the_full_width_with_a_gentle_wobble():
    inside, _ = region_shape(flat(), PathRegion(path={"points": [[0, 0.5], [1, 0.5]], "width": 0.25}))
    widths = inside.sum(axis=0)  # per column across a horizontal path; nominal 16 blocks (0.25 * 64)
    assert widths.mean() == pytest.approx(16, abs=1.5)
    assert widths.min() >= 16 * (1 - PATH_WOBBLE) - 1 and widths.max() <= 16 * (1 + PATH_WOBBLE) + 2
    assert widths.max() - widths.min() >= 2  # edges are not ruler-straight


def test_path_is_a_smooth_curve_through_its_points():
    pts = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.9, 0.6]]  # two right-angle corners
    curve = np.array([to_blocks(flat(), q) for q in _smooth_points(flat(), pts)])
    for q in pts:  # passes through every control point
        assert np.hypot(*(curve - to_blocks(flat(), q)).T).min() < 1e-6
    d = np.diff(curve, axis=0)
    turn = np.degrees(np.abs(np.angle(np.exp(1j * np.diff(np.arctan2(d[:, 1], d[:, 0]))))))
    assert turn.max() < 30  # no corner: the 90° turns are spread over several ~2-block segments


def test_path_wobble_is_deterministic_per_seed():
    region = PathRegion(path={"points": [[0.1, 0.5], [0.9, 0.5]], "width": 0.1})
    a, b = flat(), flat()
    c = flat()
    c.seed = 5
    assert np.array_equal(region_shape(a, region)[0], region_shape(b, region)[0])
    assert not np.array_equal(region_shape(a, region)[0], region_shape(c, region)[0])


def stack_with_mesa():
    stack = OpStack(BaseSpec(N, N, base_y=BASE))
    stack.add("raise_region", {"region": {"polygon": SQUARE}, "height": 10, "edge": 0}, id="mesa_1")
    return stack


def test_feature_selects_the_columns_its_op_changed():
    s = stack_with_mesa().state
    poly, signed = region_shape(s, PolygonRegion(polygon=SQUARE))
    feat, _ = region_shape(s, FeatureRegion(feature="mesa_1"))
    # polygon_inside is half-open; the raise (edge_weight) also covers columns exactly on the outline
    assert np.all(feat[poly]) and np.all(signed[feat & ~poly] == 0)


@pytest.mark.parametrize("region", [
    {"circle": {"center": [0.5, 0.5], "radius": 0.2}},
    {"polygon": SQUARE},
    {"path": {"points": [[0.2, 0.5], [0.8, 0.5]], "width": 0.2}},
])
def test_edge_fades_from_one_inside_to_zero_beyond(region):
    from terrain.ops.region import AreaParams

    r = AreaParams.model_validate({"region": region}).region
    s = flat()
    inside, signed = region_shape(s, r)
    w = region_weight(s, r, edge_blocks=6.0)
    assert np.all(w[inside] == 1.0)
    assert np.all(w[signed >= 6.0] == 0.0)
    fade = (signed > 0) & (signed < 6.0)
    assert fade.any() and np.all((w[fade] > 0) & (w[fade] < 1))


def test_no_region_is_the_whole_area():
    inside, _ = region_shape(flat(), None)
    assert inside.all()
    assert np.all(region_weight(flat(), None, edge_blocks=5.0) == 1.0)


def test_unknown_feature_lists_known_ones():
    with pytest.raises(OpError, match="unknown feature 'nope'.*mesa_1"):
        stack_with_mesa().add("smooth", {"strength": 0.5, "region": {"feature": "nope"}})


def test_region_outside_the_area_fails_loudly():
    with pytest.raises(OpError, match="covers no column"):
        raise_({"circle": {"center": [0.995, 0.995], "radius": 0.001}})  # between columns, smaller than one
    with pytest.raises(OpError, match="covers no column"):
        apply_op(flat(), "paint_region", {"region": {"polygon": [[0.9, 0.9], [0.905, 0.9], [0.905, 0.905]]}, "block": "sand"})


@pytest.mark.parametrize("region", [{}, {"circle": {"center": [0.5, 0.5], "radius": 0.1}, "polygon": SQUARE}, {"square": SQUARE}])
def test_region_needs_exactly_one_key(region):
    with pytest.raises(OpError, match="region: region needs exactly one of the keys circle, polygon, path, feature"):
        raise_(region)


def test_bad_member_error_names_the_field_only():
    with pytest.raises(OpError) as e:
        raise_({"polygon": [[0.1, 0.1], [0.2, 0.2]]})
    assert "region.polygon.polygon" in str(e.value) and "circle" not in str(e.value)


def test_raise_region_and_paint_region_require_a_region():
    with pytest.raises(OpError, match="region: Field required"):
        apply_op(flat(), "raise_region", {"height": 10})
    with pytest.raises(OpError, match="region: Field required"):
        apply_op(flat(), "paint_region", {"block": "sand"})


def test_paint_region_paints_a_circle():
    s = apply_op(flat(), "paint_region", {"region": {"circle": {"center": [0.5, 0.5], "radius": 0.25}}, "block": "gravel"})
    painted = s.material == s.material_names.index("gravel")
    assert painted[32, 32] and painted[32, 48] and not painted[32, 49] and not painted[0, 0]


def test_basin_in_a_circle_digs_below_the_rim():
    s = apply_op(flat(), "raise_region", {"region": {"circle": {"center": [0.5, 0.5], "radius": 0.2}}, "height": -6, "mode": "basin", "edge": 0.02})
    assert s.height[32, 32] == pytest.approx(BASE - 6)
    assert s.height.max() == BASE  # never raises anything


@pytest.mark.parametrize("op", ["raise_region", "paint_region", "smooth", "warp", "roughen"])
def test_region_schema_is_a_union_of_closed_objects(op):
    schema = get_op(op).params.model_json_schema()
    prop = schema["properties"]["region"]
    union = prop.get("oneOf") or next(a["oneOf"] for a in prop["anyOf"] if "oneOf" in a)  # optional: anyOf [union, null]
    members = [schema["$defs"][m["$ref"].rsplit("/", 1)[1]] for m in union]
    assert sorted(next(iter(m["properties"])) for m in members) == ["circle", "feature", "path", "polygon"]
    assert all(m["additionalProperties"] is False and len(m["required"]) == 1 for m in members)
    assert "polygon" not in schema["properties"] and "feature" not in schema["properties"]
