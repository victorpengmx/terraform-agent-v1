"""Invariant tests for terrace: steps follow the slope, flats and the outside stay put, heights move ≤ one step."""

import numpy as np
import pytest

from terrain.ops import OpError, apply_op
from terrain.opstack import BaseSpec, OpStack
from terrain.state import WorldState

N = 129
BASE = 64.0


def ramp(rise=60.0):
    """Slope rising west -> east by `rise` blocks (~25° for 60 over 128)."""
    s = WorldState.flat(N, N, base_y=BASE)
    s.height = (BASE + np.linspace(0, rise, N)[None, :] * np.ones((N, 1))).astype(np.float32)
    return s


def test_steps_are_monotonic_with_flat_ledges_and_steeper_risers():
    before = ramp()
    after = apply_op(before, "terrace", {"step": 8, "roughness": 0, "steepness": 0.8, "edge": 0})
    row = after.height[N // 2, 10:-10]
    d = np.diff(row)
    assert np.all(d >= -1e-4)  # never steps back down along the slope: no pits
    original = np.diff(before.height[N // 2]).max()
    assert (d < original / 4).mean() > 0.5  # mostly ledges, much flatter than the slope they replace
    assert d.max() > 2 * original  # risers steeper than the original slope


def test_ledges_flatten_as_steepness_rises():
    def ledge_slope(steepness):
        row = apply_op(ramp(), "terrace", {"step": 8, "roughness": 0, "steepness": steepness, "edge": 0}).height[N // 2, 10:-10]
        return float(np.percentile(np.diff(row), 25))  # typical ledge gradient

    assert ledge_slope(1.0) < 0.01 < ledge_slope(0.5) < ledge_slope(0.1)  # gentle steps stay sloped, sheer ones flat


def test_heights_move_at_most_one_step():
    before = ramp()
    after = apply_op(before, "terrace", {"step": 6, "character": "jagged"}, seed=3)
    assert np.abs(after.height - before.height).max() <= 6 + 1e-3


def test_flat_ground_is_left_alone():
    flat = WorldState.flat(N, N, base_y=65.4)  # not on a step boundary: a naive quantizer would move it
    after = apply_op(flat, "terrace", {"step": 8})
    assert np.array_equal(after.height, flat.height)


def test_outside_the_region_is_unchanged():
    before = ramp()
    region = {"polygon": [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]}
    after = apply_op(before, "terrace", {"step": 8, "region": region, "edge": 0})
    outside = np.ones(before.height.shape, dtype=bool)
    outside[30:100, 30:100] = False
    assert np.array_equal(after.height[outside], before.height[outside])
    assert not np.array_equal(after.height, before.height)


def test_steepness_makes_sheerer_risers():
    gentle = apply_op(ramp(), "terrace", {"step": 8, "roughness": 0, "steepness": 0.1, "edge": 0})
    sheer = apply_op(ramp(), "terrace", {"step": 8, "roughness": 0, "steepness": 0.9, "edge": 0})
    assert np.diff(sheer.height[N // 2]).max() > 1.5 * np.diff(gentle.height[N // 2]).max()


def test_roughness_varies_with_the_seed_and_zero_roughness_does_not():
    a = apply_op(ramp(), "terrace", {"step": 8, "roughness": 0.8}, seed=1)
    b = apply_op(ramp(), "terrace", {"step": 8, "roughness": 0.8}, seed=2)
    assert not np.array_equal(a.height, b.height)
    c = apply_op(ramp(), "terrace", {"step": 8, "roughness": 0}, seed=1)
    d = apply_op(ramp(), "terrace", {"step": 8, "roughness": 0}, seed=2)
    assert np.array_equal(c.height, d.height)


def test_character_resolves_into_the_op_stack():
    stack = OpStack(BaseSpec(33, 33, base_y=BASE))
    stack.add("raise_peak", {"center": [0.5, 0.5], "radius": 0.4, "height": 40}, id="hill")
    stack.add("terrace", {"step": 5, "character": "jagged", "roughness": 0.2}, id="steps")
    params = stack.entries[-1].params
    assert params["character"] == "jagged" and params["steepness"] == 0.85 and params["roughness"] == 0.2
    default = stack.add("terrace", {"step": 5}).params
    assert default["character"] == "rugged" and default["roughness"] == 0.55


@pytest.mark.parametrize("bad, msg", [({"step": 1}, "step"), ({"step": 8, "character": "spiky"}, "character")])
def test_invalid_params_fail_loudly(bad, msg):
    with pytest.raises(OpError, match=msg):
        apply_op(ramp(), "terrace", bad)


def test_terraced_stream_slope_gives_a_stepped_river():
    stack = OpStack(BaseSpec(N, N, base_y=BASE, seed=2))
    stack.add("raise_peak", {"center": [1.0, 0.5], "radius": 1.0, "height": 40, "steepness": 0.304, "roughness": 0}, id="slope")
    creek = [[0.8, 0.5], [0.0, 0.5]]
    stack.add("terrace", {"step": 5, "steepness": 0.9, "roughness": 0.1, "region": {"path": {"points": creek, "width": 0.15}}}, id="steps")
    stack.add("carve_river", {"path": creek, "width": 0.03, "depth": 2})
    surface = stack.state.water[N // 2]
    levels = surface[~np.isnan(surface)]
    assert len(np.unique(np.round(levels))) >= 4  # the water drops in several distinct steps (cascades)
