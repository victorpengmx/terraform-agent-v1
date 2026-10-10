import numpy as np
import pytest

from terrain import brushes as b
from terrain.state import WorldState


@pytest.fixture
def state():
    return WorldState.flat(65, 65, base_y=64)


@pytest.mark.parametrize("kind", ["linear", "smoothstep", "gaussian"])
def test_falloff_is_monotonic_and_bounded(kind):
    t = np.linspace(-0.5, 1.5, 201)
    w = b.falloff(t, kind)
    assert np.all(np.diff(w) >= -1e-7)
    assert w.min() == pytest.approx(0) and w.max() == pytest.approx(1)


def test_path_distance_and_parameter(state):
    d, t = b.path_distance(state, [(0.0, 0.5), (1.0, 0.5)])
    assert d[32, 10] == pytest.approx(0) and d[0, 10] == pytest.approx(32)
    assert t[32, 0] == pytest.approx(0) and t[32, 64] == pytest.approx(1) and t[32, 32] == pytest.approx(0.5)


def test_polygon_inside_and_signed_distance(state):
    square = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]
    inside = b.polygon_inside(state, square)
    assert inside[32, 32] and not inside[5, 5]
    sd = b.polygon_distance(state, square)
    assert sd[32, 32] == pytest.approx(-16) and sd[32, 0] == pytest.approx(16)


def test_edge_weight(state):
    sd = np.array([-5.0, 0.0, 2.0, 4.0, 10.0])
    w = b.edge_weight(sd, edge=4.0)
    assert list(w[:2]) == [1.0, 1.0] and 0 < w[2] < 1 and w[3] == 0 and w[4] == 0


def test_blend_contracts():
    h = np.array([10.0, 50.0, 90.0], dtype=np.float32)
    w = np.ones(3, dtype=np.float32)
    assert np.all(b.blend(h, 50.0, w, "max") >= h)
    assert np.all(b.blend(h, 50.0, w, "min") <= h)
    assert np.allclose(b.blend(h, 50.0, w, "lerp_to_target"), 50)
    assert np.allclose(b.blend(h, 50.0, w * 0.5, "lerp_to_target"), [30, 50, 70])
    assert np.allclose(b.blend(h, 5.0, w * 0, "add"), h)


def test_smooth_blend_reduces_variance():
    rng = np.random.default_rng(0)
    h = rng.normal(64, 10, (32, 32)).astype(np.float32)
    out = b.blend(h, 0.0, np.ones_like(h), "smooth", sigma=2)
    assert out.std() < h.std() / 2 and out.mean() == pytest.approx(h.mean(), abs=0.5)


def test_feature_mask_unknown(state):
    with pytest.raises(KeyError, match="unknown feature 'x'"):
        b.feature_mask(state, "x")


WHOLE = [[0, 0], [1, 0], [1, 1], [0, 1]]


@pytest.mark.parametrize("width, depth", [(65, 65), (192, 192), (256, 256), (100, 60)])
def test_whole_area_polygon_covers_every_column(width, depth):
    s = WorldState.flat(width, depth)
    assert b.polygon_inside(s, WHOLE).all()
    assert (b.polygon_distance(s, WHOLE) <= 0).all()


def test_polygon_edges_count_on_every_side(state):
    square = [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]  # edges on block columns 16 and 48
    inside = b.polygon_inside(state, square)
    assert inside.sum() == 33 * 33
    assert inside[16, 16] and inside[48, 48] and inside[16, 48] and inside[48, 16]
    assert not inside[15, 32] and not inside[49, 32] and not inside[32, 15] and not inside[32, 49]
