"""Invariant tests for the general detail passes: warp, roughen, hydraulic erode."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from terrain.ops import OpError, apply_op
from terrain.opstack import BaseSpec, OpStack
from terrain.state import WorldState

N = 129
BASE = 64.0
SQUARE = [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]


def flat():
    return WorldState.flat(N, N, base_y=BASE)


def mesa():
    return apply_op(flat(), "raise_region", {"region": {"polygon": SQUARE}, "height": 30, "mode": "plateau", "edge": 0})


def cone():
    return apply_op(flat(), "raise_peak", {"center": [0.5, 0.5], "radius": 0.35, "height": 80, "steepness": 0.6, "roughness": 0})


def edge_straightness(state, band=range(40, 90)):
    """Mean over the four mesa walls of the std of the wall position (0 = perfectly straight)."""
    top = state.height > BASE + 15
    spreads = []
    for grid in (top, top[:, ::-1], top.T, top.T[:, ::-1]):  # west, east, north, south walls
        spreads.append(np.std([np.argmax(grid[i]) for i in band if grid[i].any()]))
    return float(np.mean(spreads))


# ---------------------------------------------------------------- warp


def test_warp_makes_straight_walls_irregular():
    m = mesa()
    w = apply_op(m, "warp", {"amount": 0.06, "scale": 0.15}, seed=1)
    assert edge_straightness(m) == 0
    assert edge_straightness(w) > 1.5


def test_warp_only_moves_heights_around():
    m = mesa()
    w = apply_op(m, "warp", {"amount": 0.08}, seed=2)
    assert w.height.max() <= m.height.max() + 1e-4 and w.height.min() >= m.height.min() - 1e-4


def test_warp_is_deterministic_and_respects_the_selection():
    m = mesa()
    a = apply_op(m, "warp", {"amount": 0.06, "region": {"polygon": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]}, "edge": 0}, seed=3)
    b = apply_op(m, "warp", {"amount": 0.06, "region": {"polygon": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]}, "edge": 0}, seed=3)
    assert np.array_equal(a.height, b.height)
    assert np.array_equal(a.height[:, 80:], m.height[:, 80:])  # east of the selection + its fade (>= 1.5x shift)
    assert not np.array_equal(a.height[:, :60], m.height[:, :60])


def test_warp_feature_selection_and_unknown_feature():
    stack = OpStack(BaseSpec(N, N, BASE))
    stack.add("raise_region", {"region": {"polygon": SQUARE}, "height": 30, "mode": "plateau", "edge": 0}, id="mesa_1")
    stack.add("warp", {"amount": 0.05, "region": {"feature": "mesa_1"}})
    assert edge_straightness(stack.state) > 1
    with pytest.raises(OpError, match="unknown feature"):
        stack.add("warp", {"amount": 0.05, "region": {"feature": "nope"}})
    with pytest.raises(OpError, match="exactly one of"):
        stack.add("warp", {"amount": 0.05, "region": {"feature": "mesa_1", "polygon": SQUARE}})


# ---------------------------------------------------------------- roughen


@pytest.mark.parametrize("style", ["fbm", "ridged", "billow"])
def test_roughen_is_bounded(style):
    s = apply_op(flat(), "roughen", {"amount": 6, "style": style, "weight": "uniform"}, seed=4)
    d = s.height - BASE
    assert np.abs(d).max() <= 6 + 1e-4 and d.std() > 0.5


def test_roughen_relief_weighting_spares_the_base():
    c = cone()
    s = apply_op(c, "roughen", {"amount": 8, "weight": "relief"}, seed=5)
    d = np.abs(s.height - c.height)
    assert d[c.height <= BASE + 0.5].max() < 0.5  # flat surroundings barely touched
    assert d[c.height > BASE + 60].mean() > 1.0  # upper slopes clearly roughened


def test_ridged_has_sharp_crests_and_round_troughs():
    def crest_to_trough(h):  # how much sharper the crests are than the troughs (1 = symmetric)
        lap = gaussian_filter(h.astype(np.float64), 1.0, order=(2, 0)) + gaussian_filter(h.astype(np.float64), 1.0, order=(0, 2))
        return float(-lap[lap < 0].mean() / lap[lap > 0].mean())

    args = {"amount": 8, "weight": "uniform", "scale": 0.06}  # 7.7 blocks: a single octave, so the style shows alone
    ridged = apply_op(flat(), "roughen", args | {"style": "ridged"}, seed=6).height
    fbm = apply_op(flat(), "roughen", args | {"style": "fbm"}, seed=6).height
    assert crest_to_trough(ridged) > 1.3 * crest_to_trough(fbm)


def test_roughen_respects_polygon():
    s = apply_op(flat(), "roughen", {"amount": 6, "weight": "uniform", "region": {"polygon": SQUARE}, "edge": 0}, seed=7)
    assert np.all(s.height[:30, :] == BASE) and (s.height[40:90, 40:90] != BASE).any()


# ---------------------------------------------------------------- hydraulic erosion


def test_hydraulic_conserves_mass_and_is_deterministic():
    c = cone()
    a = apply_op(c, "erode", {"kind": "hydraulic", "strength": 0.125}, seed=8)  # 15 rain events
    b = apply_op(c, "erode", {"kind": "hydraulic", "strength": 0.125}, seed=8)
    assert np.array_equal(a.height, b.height)
    assert a.height.astype(np.float64).sum() == pytest.approx(c.height.astype(np.float64).sum(), rel=1e-6)


def test_hydraulic_cuts_gullies_and_builds_an_apron():
    c = cone()
    e = apply_op(c, "erode", {"kind": "hydraulic", "strength": 0.75}, seed=9)  # 40 rain events
    yy, xx = np.mgrid[0:N, 0:N]
    r = np.hypot(xx - 64, yy - 64)
    mid = (r > 15) & (r < 25)
    detail = lambda h: float((h - gaussian_filter(h.astype(np.float64), 4))[mid].std())
    assert detail(e.height) > 1.2 * detail(c.height)  # gullies and spurs on the flanks
    foot = (r > 42) & (r < 52)  # just outside the cone's radius (0.35 * 128 = 45)
    assert (e.height - c.height)[foot].mean() > 0.2  # sediment apron
    assert e.height.max() < c.height.max()  # the summit wears down


def test_thermal_default_is_unchanged():
    c = cone()
    assert np.array_equal(apply_op(c, "erode", {"strength": 0}).height, apply_op(c, "erode", {"kind": "thermal", "strength": 0}).height)


def test_hydraulic_leaves_no_pillars():
    """Regression: unspread pit deposits piled sediment onto single columns (~300 pillars per demo in game)."""
    from terrain.compiler import surface_y

    s = apply_op(cone(), "noise_base", {"amplitude": 3, "feature_size": 0.1}, seed=10)
    e = apply_op(s, "erode", {"kind": "hydraulic", "strength": 0.75}, seed=11)
    h = surface_y(e).astype(int)
    p = np.pad(h, 1, mode="edge")
    highest_neighbour = np.max([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]], axis=0)
    assert int(((h - highest_neighbour) >= 2).sum()) == 0
