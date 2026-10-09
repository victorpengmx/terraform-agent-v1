"""Invariant tests for noise_base, raise_region, carve_valley, smooth, erode."""

import math

import numpy as np
import pytest

from terrain.ops import OpError, apply_op
from terrain.noise import MIN_DETAIL_BLOCKS, detail_octaves
from terrain.ops.erode import hydraulic_iterations, thermal_settings, thermal_step
from terrain.state import WorldState

BASE = 64.0
SQUARE = [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]


def flat(n=65):
    return WorldState.flat(n, n, base_y=BASE)


def max_drop(h):
    """Largest height difference between 4-neighbours."""
    return max(np.abs(np.diff(h, axis=0)).max(), np.abs(np.diff(h, axis=1)).max())


# ---------------------------------------------------------------- noise_base


def test_noise_is_deterministic_per_seed_and_bounded():
    p = {"amplitude": 10}
    a = apply_op(flat(), "noise_base", p, seed=1)
    assert np.array_equal(a.height, apply_op(flat(), "noise_base", p, seed=1).height)
    assert not np.array_equal(a.height, apply_op(flat(), "noise_base", p, seed=2).height)
    assert np.all(np.abs(a.height - BASE) <= 10 + 1e-4)
    assert (a.height - BASE).std() > 1  # actually bumpy


def test_noise_feature_size_controls_smoothness():
    big = apply_op(flat(129), "noise_base", {"amplitude": 10, "feature_size": 0.5, "roughness": 0}, seed=3)
    small = apply_op(flat(129), "noise_base", {"amplitude": 10, "feature_size": 0.05, "roughness": 0}, seed=3)
    assert max_drop(big.height) < max_drop(small.height)


def test_noise_roughness_adds_finer_detail():
    smooth = apply_op(flat(129), "noise_base", {"amplitude": 10, "character": "gentle"}, seed=3)
    rugged = apply_op(flat(129), "noise_base", {"amplitude": 10, "character": "jagged"}, seed=3)
    assert max_drop(smooth.height) < max_drop(rugged.height)


@pytest.mark.parametrize("n", [33, 129, 513])
@pytest.mark.parametrize("feature_size", [0.02, 0.1, 0.3, 1.0])
def test_noise_octaves_stop_at_four_block_features(n, feature_size):
    octaves = detail_octaves(flat(n), feature_size)
    finest = feature_size * (n - 1) / 2 ** (octaves - 1)
    assert octaves == 1 or finest >= MIN_DETAIL_BLOCKS  # no single-column pillars
    assert octaves == 8 or finest / 2 < MIN_DETAIL_BLOCKS  # but every landform-sized octave is there


def test_noise_base_resolves_roughness_only():
    from terrain.opstack import BaseSpec, OpStack

    p = OpStack(BaseSpec(33, 33)).add("noise_base", {"amplitude": 3, "character": "gentle"}).params
    assert p == {"amplitude": 3.0, "feature_size": 0.25, "character": "gentle", "roughness": 0.2}


# ---------------------------------------------------------------- raise_region


def test_region_only_changes_inside_polygon_plus_edge():
    s = apply_op(flat(), "raise_region", {"region": {"polygon": SQUARE}, "height": 20, "edge": 0.05})  # edge 3.2 blocks
    assert s.height[32, 32] == pytest.approx(BASE + 20)
    assert np.all(s.height[:12, :] == BASE) and np.all(s.height[:, 53:] == BASE)


def test_region_negative_height_lowers():
    s = apply_op(flat(), "raise_region", {"region": {"polygon": SQUARE}, "height": -10})
    assert s.height.min() == pytest.approx(BASE - 10) and s.height.max() == pytest.approx(BASE)


def test_plateau_is_flat_and_never_lowers():
    bumpy = apply_op(flat(), "noise_base", {"amplitude": 8}, seed=5)
    s = apply_op(bumpy, "raise_region", {"region": {"polygon": SQUARE}, "height": 15, "mode": "plateau", "edge": 0})
    assert np.all(s.height >= bumpy.height - 1e-4)
    top = s.height[20:44, 20:44]
    assert top.max() - top.min() < 1e-3


def test_plateau_rejects_negative_height():
    with pytest.raises(OpError, match="plateau mode needs a positive height"):
        apply_op(flat(), "raise_region", {"region": {"polygon": SQUARE}, "height": -5, "mode": "plateau"})


# ---------------------------------------------------------------- carve_valley


@pytest.mark.parametrize("profile", ["v", "u"])
def test_valley_only_lowers_and_reaches_depth_on_path(profile):
    hilly = apply_op(flat(), "raise_region", {"region": {"polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}, "height": 40, "edge": 0})
    s = apply_op(hilly, "carve_valley", {"path": [[0, 0.5], [1, 0.5]], "width": 0.3, "depth": 20, "profile": profile})
    assert np.all(s.height <= hilly.height + 1e-4)
    assert np.allclose(s.height[32, :], hilly.height[32, :] - 20)
    assert np.all(s.height[:20, :] == hilly.height[:20, :])  # beyond the rim (half width 9.6 blocks)


def test_valley_floor_curves_through_its_points():
    hilly = apply_op(flat(), "raise_region", {"region": {"polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}, "height": 40, "edge": 0})
    points = [[0.125, 0.125], [0.5, 0.875], [0.875, 0.125]]  # on block columns (x, z = 8, 32, 56)
    s = apply_op(hilly, "carve_valley", {"path": points, "width": 0.1, "depth": 20})
    rim = float(hilly.height[0, 0])
    for u, v in points:
        assert s.height[round(v * 64), round(u * 64)] == pytest.approx(rim - 20, abs=1e-3)
    # The tip is rounded: 2 blocks inside the V the raw polyline's arms are < 1 block away (~17 deep);
    # the smoothed floor passes the tip horizontally, so that column is cut much less.
    assert s.height[54, 32] > rim - 10


def test_u_valley_has_flatter_floor_than_v():
    args = {"path": [[0, 0.5], [1, 0.5]], "width": 0.4, "depth": 20}
    v = apply_op(flat(), "carve_valley", args | {"profile": "v"}).height[:, 32]
    u = apply_op(flat(), "carve_valley", args | {"profile": "u"}).height[:, 32]
    assert u[28] < v[28]  # 4 blocks off-centre, the U is still near full depth


# ---------------------------------------------------------------- smooth


def test_smooth_reduces_roughness_and_keeps_mean():
    rough = apply_op(flat(), "noise_base", {"amplitude": 10, "feature_size": 0.05}, seed=9)
    s = apply_op(rough, "smooth", {"strength": 1.0, "radius": 3})
    assert max_drop(s.height) < max_drop(rough.height) / 2
    assert s.height.mean() == pytest.approx(rough.height.mean(), abs=0.2)


def test_local_smooth_leaves_far_terrain_alone():
    rough = apply_op(flat(), "noise_base", {"amplitude": 10, "feature_size": 0.05}, seed=9)
    s = apply_op(rough, "smooth", {"strength": 1.0, "region": {"circle": {"center": [0.25, 0.25], "radius": 0.15}}})
    assert np.array_equal(s.height[45:, 45:], rough.height[45:, 45:])
    assert not np.array_equal(s.height[16, 16], rough.height[16, 16])


# ---------------------------------------------------------------- erode


def test_thermal_step_conserves_mass():
    h = np.zeros((20, 20))
    h[10, 10] = 50
    out = thermal_step(h, 1.0)
    assert out.sum() == pytest.approx(h.sum()) and out[10, 10] < 50


def test_thermal_steps_relax_slopes_to_talus():
    spire = apply_op(flat(), "raise_peak", {"center": [0.5, 0.5], "radius": 0.15, "height": 60, "steepness": 1, "roughness": 0})
    h = spire.height.astype(np.float64)
    for _ in range(300):
        h = thermal_step(h, 1.0)
    assert max_drop(h) <= 1.0 + 0.25


def test_stronger_thermal_erosion_wears_slopes_further():
    spire = apply_op(flat(), "raise_peak", {"center": [0.5, 0.5], "radius": 0.15, "height": 60, "steepness": 1, "roughness": 0})
    worn = [apply_op(spire, "erode", {"strength": s}).height for s in (0.1, 0.5, 0.9)]
    assert max_drop(spire.height) > max_drop(worn[0]) > max_drop(worn[1]) > max_drop(worn[2])
    assert spire.height.max() > worn[0].max() > worn[1].max() > worn[2].max()
    for h in worn:
        assert h.sum() == pytest.approx(spire.height.sum(), rel=1e-5)  # material moves, never vanishes


def test_erode_strength_one_half_keeps_the_old_defaults():
    assert thermal_settings(0.5) == (30, 40.0)  # old iterations=30, talus_deg=40
    assert hydraulic_iterations(0.5) == 30


def test_erode_leaves_gentle_terrain_alone():
    gentle = apply_op(flat(), "raise_peak", {"center": [0.5, 0.5], "radius": 0.5, "height": 10, "steepness": 0, "roughness": 0})
    s = apply_op(gentle, "erode", {"strength": 0.5})
    assert np.allclose(s.height, gentle.height)


@pytest.mark.parametrize("kind", ["thermal", "hydraulic"])
def test_erode_region_limits_the_change(kind):
    spire = apply_op(flat(), "raise_peak", {"center": [0.3, 0.5], "radius": 0.15, "height": 60, "steepness": 1, "roughness": 0})
    spire = apply_op(spire, "raise_peak", {"center": [0.7, 0.5], "radius": 0.15, "height": 60, "steepness": 1, "roughness": 0})
    region = {"circle": {"center": [0.3, 0.5], "radius": 0.2}}
    s = apply_op(spire, "erode", {"kind": kind, "strength": 0.8, "region": region, "edge": 0}, seed=1)
    assert np.array_equal(s.height[:, 40:], spire.height[:, 40:])  # the east peak (outside) is untouched
    assert s.height[:, :32].max() < spire.height[:, :32].max()  # the west one is worn


@pytest.mark.parametrize(
    "op, params, base_y",
    [
        ("noise_base", {"amplitude": 128}, 300),
        ("raise_region", {"region": {"polygon": SQUARE}, "height": 256}, 200),
        ("carve_valley", {"path": [[0, 0.5], [1, 0.5]], "width": 0.2, "depth": 256}, 0),
    ],
)
def test_ops_respect_world_limits(op, params, base_y):
    with pytest.raises(OpError, match="build limit|world floor"):
        apply_op(WorldState.flat(65, 65, base_y=base_y), op, params, seed=1)


def test_basin_digs_a_flat_floor_below_the_rim_on_a_peak():
    peak = apply_op(flat(), "raise_peak", {"center": [0.5, 0.5], "radius": 0.4, "height": 60})
    s = apply_op(peak, "raise_region", {"region": {"polygon": [[0.42, 0.42], [0.58, 0.42], [0.58, 0.58], [0.42, 0.58]]}, "height": -6, "mode": "basin", "edge": 0.02})
    assert np.all(s.height <= peak.height + 1e-4)  # never raises
    floor = s.height[30:35, 30:35]
    assert floor.max() - floor.min() < 1e-3  # flat floor
    lake = apply_op(s, "fill_lake", {"at": [0.5, 0.5]})  # holds water even on a summit
    assert not np.isnan(lake.water[32, 32])


def test_basin_rejects_positive_height():
    with pytest.raises(OpError, match="basin mode needs a negative height"):
        apply_op(flat(), "raise_region", {"region": {"polygon": SQUARE}, "height": 5, "mode": "basin"})
