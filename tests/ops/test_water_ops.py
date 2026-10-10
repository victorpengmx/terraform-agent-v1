"""Invariant tests for carve_river and fill_lake: beds descend, water never floats, lakes fill to spill."""

import numpy as np
import pytest

from terrain.ops import OpError, apply_op
from terrain.ops.water import spill_levels
from terrain.state import WorldState

N = 65


def sloped(rise_toward_east: float = 0.0, base=80.0) -> WorldState:
    """Terrain that rises (or falls) linearly from west to east."""
    s = WorldState.flat(N, N, base_y=base)
    s.height += np.linspace(0, rise_toward_east, N, dtype=np.float32)[None, :]
    return s


def assert_water_never_floats(state: WorldState) -> None:
    """Every dry 4-neighbour of a water column is at least as high as that water surface, in blocks: the
    compiler rounds both, and that is what stands in the world."""
    wet = ~np.isnan(state.water)
    assert wet.any()
    for dz, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        z, x = np.nonzero(wet)
        nz, nx = z + dz, x + dx
        ok = (nz >= 0) & (nz < N) & (nx >= 0) & (nx < N)
        z, x, nz, nx = z[ok], x[ok], nz[ok], nx[ok]
        dry = ~wet[nz, nx]
        assert np.all(np.round(state.height[nz, nx][dry]) >= np.round(state.water[z, x][dry]))
    assert np.all(state.water[wet] > state.height[wet])  # water above its bed


# ---------------------------------------------------------------- carve_river

RIVER = {"path": [[0.0, 0.5], [1.0, 0.5]], "width": 0.1, "depth": 4}


@pytest.mark.parametrize("rise", [-30.0, 0.0, 30.0])  # downhill, flat, and uphill toward the mouth
def test_river_bed_descends_monotonically(rise):
    s = apply_op(sloped(rise), "carve_river", RIVER, seed=1)
    bed = s.height[32, :]  # along the river centre line, source (west) to mouth (east)
    assert np.all(np.diff(bed) <= 1e-3)
    surface = s.water[32, :]
    assert not np.isnan(surface).any() and np.all(np.diff(surface) <= 1e-3)


def test_river_cuts_through_a_rise_instead_of_climbing_it():
    s = sloped()
    s = apply_op(s, "raise_region", {"region": {"polygon": [[0.4, 0], [0.6, 0], [0.6, 1], [0.4, 1]]}, "height": 20, "edge": 0})
    out = apply_op(s, "carve_river", RIVER, seed=1)
    assert out.height[32, 32] <= out.height[32, 5] + 1e-3  # the ridge got a gorge


@pytest.mark.parametrize("meander", [0.0, 0.6])
def test_river_water_never_floats(meander):
    s = apply_op(WorldState.flat(N, N, base_y=80), "noise_base", {"amplitude": 6}, seed=4)
    s = apply_op(s, "carve_river", RIVER | {"meander": meander, "path": [[0.1, 0.2], [0.5, 0.6], [1.0, 0.8]]}, seed=2)
    assert_water_never_floats(s)


def test_meander_bends_the_river():
    straight = apply_op(sloped(), "carve_river", RIVER, seed=3)
    winding = apply_op(sloped(), "carve_river", RIVER | {"meander": 1.0}, seed=3)
    wet_rows = lambda s: set(np.nonzero(~np.isnan(s.water))[0])
    assert len(wet_rows(winding)) > len(wet_rows(straight)) + 4


def test_river_must_reach_edge_or_water():
    with pytest.raises(OpError, match="neither at the area edge nor in water"):
        apply_op(sloped(), "carve_river", RIVER | {"path": [[0.0, 0.5], [0.5, 0.5]]})


def test_river_may_end_in_a_lake():
    s = apply_op(sloped(), "raise_region", {"region": {"polygon": [[0.6, 0.4], [0.8, 0.4], [0.8, 0.6], [0.6, 0.6]]}, "height": -10, "edge": 0})
    s = apply_op(s, "fill_lake", {"at": [0.7, 0.5]})
    s = apply_op(s, "carve_river", RIVER | {"path": [[0.0, 0.5], [0.7, 0.5]]}, seed=1)
    assert not np.isnan(s.water[32, 10])


# ---------------------------------------------------------------- fill_lake


def crater(notch_y: float = 70.0) -> WorldState:
    """Bowl at y=60 inside a ring wall at y=80 with one notch at notch_y; outside ground y=64."""
    s = WorldState.flat(N, N, base_y=64)
    zz, xx = np.mgrid[0:N, 0:N]
    r = np.hypot(xx - 32, zz - 32)
    s.height[r < 20] = 80
    s.height[r < 16] = 60
    s.height[(r >= 16) & (r < 20) & (np.abs(zz - 32) <= 1) & (xx > 32)] = notch_y
    return s


def test_lake_fills_to_spill_height():
    s = apply_op(crater(70), "fill_lake", {"at": [0.5, 0.5]})
    assert s.water[32, 32] == pytest.approx(70)
    assert np.isnan(s.water[0, 0])  # outside the crater stays dry
    assert_water_never_floats(s)


def test_lake_depth_cap():
    s = apply_op(crater(70), "fill_lake", {"at": [0.5, 0.5], "max_depth": 4})
    assert s.water[32, 32] == pytest.approx(64)
    assert_water_never_floats(s)


def test_no_basin_is_an_error():
    with pytest.raises(OpError, match="no basin"):
        apply_op(WorldState.flat(N, N, base_y=64), "fill_lake", {"at": [0.5, 0.5]})


def test_spill_levels_basic():
    h = np.array([[5, 5, 5, 5], [5, 1, 1, 5], [5, 1, 3, 2], [5, 5, 5, 5]], dtype=np.float32)
    lv = spill_levels(h)
    assert lv[1, 1] == 3  # the only way out passes over the 3 (the 2 on the border is behind it)
    assert lv[0, 0] == 5


# ---------------------------------------------------------------- set_sea_level


def test_sea_floods_from_the_edge_but_not_enclosed_basins():
    s = WorldState.flat(N, N, base_y=50)
    s.height[16:48, 16:48] = 80  # island plateau
    s.height[28:36, 28:36] = 40  # enclosed crater below sea level
    out = apply_op(s, "set_sea_level", {"level": 62})
    assert out.water[0, 0] == 62 and np.isnan(out.water[32, 32]) and np.isnan(out.water[20, 20])
    assert_water_never_floats(out)


def test_sea_needs_low_ground_at_the_edge():
    with pytest.raises(OpError, match="touches the area edge"):
        apply_op(WorldState.flat(N, N, base_y=70), "set_sea_level", {"level": 62})


@pytest.mark.parametrize("depth", [2, 3, 6])
def test_shallow_river_water_survives_compile_along_its_whole_length(depth):
    """Regression: depth-2 rivers used to hold < 1 block of water; rounding at compile left 3 water blocks."""
    from terrain.brushes import path_distance
    from terrain.compiler.compile import surface_y, water_top
    from terrain.opstack import BaseSpec, OpStack

    creek = [[0.53, 0.45], [0.42, 0.6], [0.3, 0.75], [0.2, 0.9], [0.14, 1.0]]
    stack = OpStack(BaseSpec(N, N, base_y=64, seed=1))
    stack.add("raise_peak", {"center": [0.85, 0.12], "radius": 0.95, "height": 30, "steepness": 0.143, "roughness": 0})  # ~10° slope
    stack.add("noise_base", {"amplitude": 3, "feature_size": 0.25})
    stack.add("carve_river", {"path": creek, "width": 0.05, "depth": depth})
    s = stack.state
    canvas = ~np.isnan(s.water)
    compiled = water_top(s) > surface_y(s)
    assert np.array_equal(canvas, compiled)  # every canvas water column becomes at least one water block
    dist, t = path_distance(s, creek)
    for a in np.linspace(0, 0.8, 5):  # water all the way from source to mouth
        assert compiled[(dist < 1.0) & (t >= a) & (t < a + 0.2)].mean() > 0.8  # along the centre line


# ---------------------------------------------------------------- meander calibration (DESIGN §4.2)


def _sinuosity(meander, width=0.03, size=256, seed=0):
    from terrain.features import river_sinuosity
    from terrain.opstack import BaseSpec, OpEntry, OpStack

    entry = OpEntry("river", "carve_river", {"path": [[0.5, 0.0], [0.5, 1.0]], "width": width, "depth": 4.0, "meander": meander})
    return river_sinuosity(OpStack(BaseSpec(size, size, seed=seed)), entry)


@pytest.mark.parametrize("meander, category", [(0.0, "almost straight"), (0.3, "winding"), (0.6, "twisty"), (1.0, "meandering")])
def test_meander_reaches_each_winding_category(meander, category):
    from terrain.features import WINDING_CLASSES, category_distance, classify

    got = [classify(_sinuosity(meander, seed=seed), WINDING_CLASSES) for seed in range(4)]
    # Most seeds hit the stated category; none is off by more than the one category of slack checks allow.
    assert got.count(category) >= 3 and all(category_distance(category, c, WINDING_CLASSES) <= 1 for c in got), got


def test_meander_keeps_the_endpoints_and_scales_bends_with_width():
    from terrain.ops.water import CarveRiverParams, river_centerline

    s = WorldState.flat(256, 256, base_y=80, seed=1)

    def line(width):
        return river_centerline(s, CarveRiverParams(path=[[0.5, 0.0], [0.5, 1.0]], width=width, depth=4, meander=0.6))

    narrow, wide = line(0.02), line(0.08)
    assert np.allclose(narrow[0], [127.5, 0]) and np.allclose(narrow[-1], [127.5, 255])
    turns = lambda pts: int((np.diff(np.sign(np.diff(pts[:, 0]))) != 0).sum())  # noqa: E731  (east-west reversals)
    assert turns(narrow) > turns(wide)  # bends repeat every ~12 channel widths
