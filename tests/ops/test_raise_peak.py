import numpy as np
import pytest

from terrain.ops import OpError, apply_op
from terrain.ops.peaks import sharpness
from terrain.opstack import BaseSpec, OpStack
from terrain.state import WorldState

BASE = 64.0


def peak(width=65, depth=65, **params):
    p = {"center": [0.5, 0.5], "radius": 0.25, "height": 80, "roughness": 0} | params  # round, smooth flanks
    return apply_op(WorldState.flat(width, depth, base_y=BASE), "raise_peak", p)


def test_peak_reaches_requested_height_at_centre():
    s = peak()
    assert s.height[32, 32] == pytest.approx(BASE + 80)
    assert s.height.max() == pytest.approx(BASE + 80)


def test_profile_descends_monotonically_from_centre():
    row = peak().height[32, 32:]  # centre outwards along +x
    assert np.all(np.diff(row) <= 1e-4)


def test_no_change_outside_radius():
    s = peak()  # radius 0.25 * 64 = 16 blocks
    x, z = np.meshgrid(np.arange(65), np.arange(65))
    outside = np.hypot(x - 32, z - 32) >= 16
    assert np.all(s.height[outside] == BASE)


def test_footprint_is_circular_on_non_square_area():
    s = peak(width=129, depth=65)  # radius uses the shorter side: 16 blocks
    raised = s.height > BASE
    assert raised[32].sum() == raised[:, 64].sum()


def test_steeper_peak_is_narrower():
    broad, sharp = peak(steepness=0.0), peak(steepness=1.0)
    assert (sharp.height > BASE + 10).sum() < (broad.height > BASE + 10).sum()


def test_steepness_maps_onto_the_old_sharpness_range():
    assert sharpness(0.0) == pytest.approx(0.5) and sharpness(1.0) == pytest.approx(4.0)
    assert sharpness(0.6) == pytest.approx(1.5, abs=0.01)  # default (rugged) = the old default mountain
    steps = [sharpness(s) for s in np.linspace(0, 1, 11)]
    assert all(a < b for a, b in zip(steps, steps[1:]))


def test_peaks_stack_additively():
    s = WorldState.flat(65, 65, base_y=BASE)
    p = {"center": [0.5, 0.5], "radius": 0.25, "height": 40}
    s = apply_op(apply_op(s, "raise_peak", p), "raise_peak", p)
    assert s.height.max() == pytest.approx(BASE + 80)


def test_deterministic():
    assert np.array_equal(peak().height, peak().height)


def test_smooth_peak_reproduces_the_round_peak_exactly():
    # The original formula: smoothstep falloff over the radius, raised to sharpness.
    x, z = np.meshgrid(np.arange(65, dtype=np.float32), np.arange(65, dtype=np.float32))
    t = np.clip(1 - np.hypot(x - 32, z - 32) / 16, 0, 1)
    expected = BASE + 80 * (t * t * (3 - 2 * t)) ** sharpness(0.6)
    s = peak(irregularity=0, ridges=0, steepness=0.6)
    assert np.allclose(s.height, expected, atol=1e-3)


# ---------------------------------------------------------------- character presets


def resolved(params, stack=None):
    stack = stack or OpStack(BaseSpec(65, 65, base_y=BASE))
    return stack.add("raise_peak", {"center": [0.5, 0.5], "radius": 0.25, "height": 40} | params).params


def test_character_resolves_into_the_op_stack():
    p = resolved({"character": "jagged"})
    assert (p["roughness"], p["steepness"], p["irregularity"], p["ridges"]) == (0.8, 0.85, 0.8, 1.0)
    default = resolved({})
    assert default["character"] == "rugged" and default["steepness"] == 0.6 and default["irregularity"] == 0.55


def test_explicit_params_override_the_preset():
    p = resolved({"character": "gentle", "steepness": 0.9, "ridges": 0.0})
    assert (p["roughness"], p["steepness"], p["irregularity"], p["ridges"]) == (0.2, 0.9, 0.2, 0.0)


def test_rougher_preset_gives_a_more_irregular_peak():
    def outline(character):
        s = apply_op(WorldState.flat(129, 129, base_y=BASE), "raise_peak",
                     {"center": [0.5, 0.5], "radius": 0.25, "height": 80, "character": character}, seed=5)
        return half_height_outline_spread(s)
    assert outline("gentle") < outline("jagged")


def test_update_rederives_values_from_a_changed_preset_or_scale():
    stack = OpStack(BaseSpec(65, 65, base_y=BASE))
    stack.add("raise_peak", {"center": [0.5, 0.5], "radius": 0.25, "height": 40, "ridges": 0.1}, id="m")
    p = stack.update("m", {"character": "jagged"}).params
    assert (p["roughness"], p["steepness"], p["irregularity"], p["ridges"]) == (0.8, 0.85, 0.8, 1.0)
    p = stack.update("m", {"roughness": 0.4, "ridges": 0.0}).params
    assert (p["character"], p["steepness"], p["irregularity"], p["ridges"]) == ("jagged", 0.85, 0.4, 0.0)
    p = stack.update("m", {"steepness": 0.3}).params
    assert (p["irregularity"], p["ridges"]) == (0.4, 0.0)  # unrelated values keep their (explicit) settings


def test_old_sharpness_param_is_rejected_with_a_clear_error():
    with pytest.raises(OpError, match="sharpness"):
        peak(sharpness=0.2)


# ---------------------------------------------------------------- irregularity and ridges

ROUGH = {"radius": 0.25, "irregularity": 0.8, "ridges": 0.8}


def rough_peak(seed=5, **params):
    p = {"center": [0.5, 0.5], "height": 80} | ROUGH | params
    return apply_op(WorldState.flat(129, 129, base_y=BASE), "raise_peak", p, seed=seed)


def half_height_outline_spread(state, height=80):
    """Relative spread of the distance from the centre to the half-height contour (0 = a circle)."""
    above = state.height > BASE + height / 2
    edge = above & ~(np.roll(above, 1, 0) & np.roll(above, -1, 0) & np.roll(above, 1, 1) & np.roll(above, -1, 1))
    z, x = np.nonzero(edge)
    r = np.hypot(x - 64, z - 64)
    return r.std() / r.mean()


def test_irregular_peak_keeps_its_summit():
    s = rough_peak()
    assert s.height[64, 64] == pytest.approx(BASE + 80)
    assert s.height.max() == pytest.approx(BASE + 80)


def test_irregularity_makes_the_outline_non_circular():
    round_ = rough_peak(irregularity=0, ridges=0)
    lobed = rough_peak(ridges=0)
    assert half_height_outline_spread(round_) < 0.03
    assert half_height_outline_spread(lobed) > 2 * half_height_outline_spread(round_) + 0.03


def test_irregular_footprint_stays_within_bound():
    s = rough_peak(irregularity=1.0)  # radius 32 blocks -> footprint at most 64
    x, z = np.meshgrid(np.arange(129), np.arange(129))
    assert np.all(s.height[np.hypot(x - 64, z - 64) >= 64 * 1.0 + 0.5] == BASE)


def test_ridges_cut_gullies_into_mid_slopes():
    smooth = rough_peak(irregularity=0, ridges=0)
    ridged = rough_peak(irregularity=0, ridges=1.0)
    assert np.all(ridged.height <= smooth.height + 1e-4)  # gullies only remove
    ring = smooth.height.copy()
    ring_mask = (ring > BASE + 30) & (ring < BASE + 50)  # mid-slope band
    diff = (smooth.height - ridged.height)[ring_mask]
    assert diff.max() > 8 and diff.min() < 1  # deep gullies next to intact ridges


def test_irregular_peak_is_deterministic_per_seed():
    assert np.array_equal(rough_peak(seed=3).height, rough_peak(seed=3).height)
    assert not np.array_equal(rough_peak(seed=3).height, rough_peak(seed=4).height)


def test_peak_through_build_limit_is_rejected():
    with pytest.raises(OpError, match="above the build limit"):
        peak(height=300)  # 64 + 300 > 319
