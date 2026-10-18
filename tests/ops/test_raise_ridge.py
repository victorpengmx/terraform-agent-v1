"""Invariant tests for raise_ridge: crest along the path, height = highest summit, bounded width, rounded ends, cols."""

import numpy as np
import pytest

from terrain.ops import apply_op
from terrain.opstack import BaseSpec, OpStack
from terrain.state import WorldState

BASE = 64.0
N = 129
RIDGE = {"path": [[0.1, 0.5], [0.9, 0.5]], "width": 0.2, "height": 40}


def ridge(**kw) -> WorldState:
    return apply_op(WorldState.flat(N, N, base_y=BASE, seed=1), "raise_ridge", RIDGE | kw, seed=1)


def test_crest_follows_the_path_and_reaches_the_height_without_roughness():
    s = ridge(roughness=0.0)
    dh = s.height - BASE
    assert dh.max() == pytest.approx(40, abs=0.05)
    crest_rows = dh[:, 30:100].argmax(axis=0)  # along the middle of the ridge, the highest row is the path's
    assert np.all(np.abs(crest_rows - 64) <= 1)


def test_nothing_changes_beyond_the_widest_flank():
    s = ridge(roughness=1.0)
    half = 0.2 * (N - 1) / 2  # the footprint is exactly the base width
    far = np.abs(np.arange(N) - 64) > half + 1
    assert np.all(s.height[far, :] == BASE)


def test_ends_round_off_instead_of_stopping_in_a_wall():
    crest = ridge(roughness=0.0).height[64] - BASE  # path ends at x = 12.8 and 115.2; half width 12.8 blocks
    assert crest[0] == 0 and crest[-1] == 0 and crest[13] == pytest.approx(40, abs=0.5)
    assert 0 < crest[5] < 30  # the end cap slopes down over the half width beyond the endpoint
    assert np.abs(np.diff(crest)).max() < 6  # no cliff at the ends


def test_roughness_makes_summits_and_saddles_along_the_crest():
    smooth, rough = ridge(roughness=0.0).height[64, 35:95] - BASE, ridge(roughness=0.9).height[64, 35:95] - BASE
    assert np.ptp(smooth) < 0.5 and np.ptp(rough) > 8
    assert rough.max() <= 40 + 1e-3  # saddles only lower the crest


@pytest.mark.parametrize("roughness", [0.0, 0.5, 1.0])
def test_height_is_the_highest_summit(roughness):
    assert (ridge(roughness=roughness).height - BASE).max() == pytest.approx(40, abs=1e-3)


def test_steepness_narrows_the_flanks():
    broad, sharp = ridge(steepness=0.1, roughness=0.0), ridge(steepness=0.9, roughness=0.0)
    assert (sharp.height[:, 64] > BASE + 20).sum() < (broad.height[:, 64] > BASE + 20).sum()


def test_character_resolves_into_the_stack_and_replay_is_deterministic():
    stack = OpStack(BaseSpec(N, N, base_y=BASE, seed=2))
    stack.add("raise_ridge", RIDGE | {"character": "jagged"}, id="range")
    assert stack.get("range").params["steepness"] == 0.85 and stack.get("range").params["roughness"] == 0.8
    assert np.array_equal(stack.replay().height, stack.state.height)
    assert stack.state.features["range"].kind == "raise_ridge"
