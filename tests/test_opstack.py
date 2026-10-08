import numpy as np
import pytest

from terrain.ops import OpError, OpParams, register_op
from terrain.ops import registry
from terrain.opstack import BaseSpec, OpStack

PEAK = {"center": [0.5, 0.5], "radius": 0.25, "height": 40}


@pytest.fixture
def noisy_op():
    """A seeded water-stage op, to test seeding and stage ordering without depending on real ops."""

    class P(OpParams):
        amount: float = 1.0

    @register_op("zz_test_jitter", params=P, stage="water")
    def jitter(state, p):
        """Add seeded noise everywhere."""
        state.height += np.random.default_rng(state.seed).random(state.height.shape).astype(np.float32) * p.amount
        return state

    yield "zz_test_jitter"
    registry._REGISTRY.pop("zz_test_jitter")


def stack():
    return OpStack(BaseSpec(33, 33, base_y=64, seed=7))


def test_replay_matches_cached_state_and_is_deterministic(noisy_op):
    s = stack()
    s.add("raise_peak", PEAK)
    s.add(noisy_op, {})
    assert np.array_equal(s.replay().height, s.state.height)
    assert np.array_equal(stack().replay().height, stack().replay().height)


def test_ids_default_and_custom():
    s = stack()
    assert s.add("raise_peak", PEAK).id == "raise_peak_1"
    assert s.add("raise_peak", PEAK, id="mountain_1").id == "mountain_1"
    assert s.add("raise_peak", PEAK).id == "raise_peak_3"
    with pytest.raises(OpError, match="already used"):
        s.add("raise_peak", PEAK, id="mountain_1")
    with pytest.raises(OpError, match="invalid id"):
        s.add("raise_peak", PEAK, id="Mountain One")


def test_update_equals_fresh_stack_with_new_params():
    s = stack()
    s.add("raise_peak", PEAK, id="m")
    s.update("m", {"height": 80})
    fresh = stack()
    fresh.add("raise_peak", PEAK | {"height": 80}, id="m")
    assert np.array_equal(s.state.height, fresh.state.height)
    assert s.get("m").params["radius"] == 0.25  # unspecified params kept


def test_remove_equals_stack_without_it():
    s = stack()
    s.add("raise_peak", PEAK, id="a")
    s.add("raise_peak", PEAK | {"center": [0.2, 0.2]}, id="b")
    s.remove("a")
    fresh = stack()
    fresh.add("raise_peak", PEAK | {"center": [0.2, 0.2]}, id="b")
    assert np.array_equal(s.state.height, fresh.state.height) and [e.id for e in s.entries] == ["b"]


def test_stage_order_independent_of_insertion_order(noisy_op):
    a = stack()
    a.add("raise_peak", PEAK, id="p")
    a.add(noisy_op, {}, id="j")
    b = stack()
    b.add(noisy_op, {}, id="j")
    b.add("raise_peak", PEAK, id="p")  # terrain-stage op added after a water op -> full replay
    assert np.array_equal(a.state.height, b.state.height)
    assert [e.id for e in b.entries] == ["p", "j"]


def test_op_noise_depends_only_on_its_own_id(noisy_op):
    a = stack()
    a.add(noisy_op, {}, id="j")
    b = stack()
    b.add(noisy_op, {}, id="other")
    b.add(noisy_op, {}, id="j")
    a_noise = a.state.height - 64
    b_noise = b.state.height - 64
    other = stack()
    other.add(noisy_op, {}, id="other")
    assert np.allclose(b_noise - (other.state.height - 64), a_noise)


def test_failed_edit_leaves_stack_unchanged():
    s = stack()
    s.add("raise_peak", PEAK, id="m")
    before = s.state.height.copy()
    with pytest.raises(OpError, match="m: .*above the build limit"):
        s.update("m", {"height": 300})
    with pytest.raises(OpError, match="invalid params"):
        s.add("raise_peak", {"center": [2, 2], "radius": 0.1, "height": 1})
    assert np.array_equal(s.state.height, before) and s.get("m").params["height"] == 40 and len(s) == 1


def test_unknown_id_lists_existing():
    s = stack()
    s.add("raise_peak", PEAK, id="m")
    with pytest.raises(OpError, match="unknown op id 'x'.*m"):
        s.remove("x")


def test_features_recorded_with_bounds_and_mask():
    s = stack()
    s.add("raise_peak", PEAK | {"center": [0.25, 0.75], "radius": 0.2}, id="mountain_1")
    f = s.state.features["mountain_1"]
    assert f.kind == "raise_peak" and f.meta["top_y"] == pytest.approx(104, abs=0.1)
    u0, v0, u1, v1 = f.bounds
    assert u0 < 0.25 < u1 and v0 < 0.75 < v1 and u1 < 0.5 and v0 > 0.5
    mask = s.state.masks["feature:mountain_1"]
    assert mask.max() == pytest.approx(1) and mask[0, 32] == 0
    s.remove("mountain_1")
    assert "mountain_1" not in s.state.features and "feature:mountain_1" not in s.state.masks


def test_checkpoint_restore():
    s = stack()
    s.add("raise_peak", PEAK, id="a")
    cp = s.checkpoint()
    s.add("raise_peak", PEAK, id="b")
    s.restore(cp)
    assert [e.id for e in s.entries] == ["a"] and s.state.height.max() == pytest.approx(104)
    s.add("raise_peak", PEAK, id="b")  # the checkpoint itself is not mutated by later edits
    s.restore(cp)
    assert len(s) == 1


def test_json_round_trip(noisy_op):
    s = stack()
    s.add("raise_peak", PEAK, id="m")
    s.add(noisy_op, {"amount": 2})
    back = OpStack.from_json(s.to_json())
    assert back.to_json() == s.to_json() and np.array_equal(back.state.height, s.state.height)
