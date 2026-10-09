import numpy as np
import pytest

from agent.session import Session
from terrain.config import MAX_MEAN_DH
from terrain.ops import OpError

WHOLE = {"polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}


def test_over_limit_apply_is_refused_and_undone():
    s = Session.flat(33, 33, base_y=64)
    with pytest.raises(OpError, match="per column on average.*Narrow the region"):
        s.apply("raise_region", {"region": WHOLE, "height": MAX_MEAN_DH + 20, "edge": 0})
    assert s.ops == [] and np.all(s.state.height == 64)


def test_over_limit_update_is_refused_and_keeps_the_old_params():
    s = Session.flat(33, 33, base_y=64)
    s.apply("raise_region", {"region": WHOLE, "height": 10, "edge": 0}, id="lift")
    with pytest.raises(OpError, match="lift would move"):
        s.update("lift", {"height": MAX_MEAN_DH + 20})
    assert s.ops[0].params["height"] == 10 and np.all(s.state.height == 74)


def test_remove_is_never_limited():
    s = Session.flat(33, 33, base_y=64, max_changed_volume=1e9)
    s.apply("raise_region", {"region": WHOLE, "height": 100, "edge": 0}, id="lift")
    s.max_changed_volume = 1.0
    s.remove("lift")
    assert s.ops == []


def test_largest_demo_op_passes():
    # The island demo's whole-area -26 seafloor is the largest legitimate single edit (EVIDENCE.md).
    s = Session.flat(256, 256, base_y=64)
    s.apply("raise_region", {"region": WHOLE, "height": -26, "edge": 0})
    assert len(s.ops) == 1
