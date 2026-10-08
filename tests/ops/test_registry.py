import json

import numpy as np
import pytest

from terrain.ops import OpError, OpParams, apply_op, get_op, list_ops, register_op, tool_schemas
from terrain.ops import registry
from terrain.state import WorldState


@pytest.fixture
def scratch_op():
    """Register a throwaway op and remove it afterwards so the global registry stays clean."""
    name = "_test_scratch_op"

    class P(OpParams):
        amount: float

    @register_op(name, params=P, stage="water")
    def scratch(state: WorldState, p: P) -> WorldState:
        """Add amount everywhere and scribble on the state in place."""
        state.height += p.amount
        return state

    yield name
    registry._REGISTRY.pop(name)


def test_tool_schema_shape():
    schema = get_op("raise_peak").tool_schema()
    assert schema["name"] == "raise_peak"
    assert schema["description"].startswith("Raise a single peak")
    inp = schema["input_schema"]
    assert inp["type"] == "object"
    assert inp["additionalProperties"] is False
    assert set(inp["required"]) == {"center", "radius", "height"}
    assert "description" in inp["properties"]["radius"]
    json.dumps(schema)  # must be serializable for the API


def test_tool_list_is_deterministic_and_stage_ordered(scratch_op):
    names = [s["name"] for s in tool_schemas()]
    assert names == [s.name for s in list_ops()]
    assert names.index("raise_peak") < names.index(scratch_op)  # terrain before water
    water = [s.name for s in list_ops("water")]
    assert scratch_op in water and all(s.stage == "water" for s in list_ops("water"))
    assert max(names.index(n) for n in [s.name for s in list_ops("terrain")]) < min(names.index(n) for n in water)


def test_unknown_op_lists_available_ops():
    with pytest.raises(OpError, match="unknown op 'raise_peek'.*raise_peak"):
        get_op("raise_peek")


@pytest.mark.parametrize(
    "params, fragment",
    [
        ({"center": [0.5, 0.5], "radius": 0.2, "height": 50, "colour": "red"}, "colour"),
        ({"center": [1.5, 0.5], "radius": 0.2, "height": 50}, "center.0"),
        ({"center": [0.5, 0.5], "height": 50}, "radius"),
    ],
)
def test_invalid_params_are_reported_not_fixed(params, fragment):
    with pytest.raises(OpError, match=f"invalid params for raise_peak:.*{fragment}"):
        apply_op(WorldState.flat(32, 32), "raise_peak", params)


def test_apply_op_does_not_mutate_input(scratch_op):
    state = WorldState.flat(8, 8, base_y=10)
    out = apply_op(state, scratch_op, {"amount": 3})
    assert np.all(state.height == 10)
    assert np.all(out.height == 13)


def test_result_above_build_limit_is_an_error(scratch_op):
    with pytest.raises(OpError, match="above the build limit y=319"):
        apply_op(WorldState.flat(8, 8, base_y=300), scratch_op, {"amount": 50})


def test_result_below_world_floor_is_an_error(scratch_op):
    with pytest.raises(OpError, match="below the world floor y=-64"):
        apply_op(WorldState.flat(8, 8, base_y=-60), scratch_op, {"amount": -10})


def test_duplicate_registration_rejected(scratch_op):
    with pytest.raises(ValueError, match="already registered"):
        register_op(scratch_op, params=get_op(scratch_op).params, stage="water")(lambda s, p: s)


def test_op_without_docstring_rejected():
    class P(OpParams):
        pass

    with pytest.raises(ValueError, match="needs a docstring"):
        register_op("_test_no_doc", params=P, stage="terrain")(lambda s, p: s)
    assert "_test_no_doc" not in registry._REGISTRY


RAW_KNOBS = {"octaves", "iterations", "talus_deg", "sharpness", "deposition", "persistence"}


def test_no_raw_algorithm_knobs_in_tool_schemas():
    """DESIGN §5.1 principle 3: the model picks calibrated scales; code derives algorithm settings."""
    for spec in list_ops():
        assert not RAW_KNOBS & set(spec.params.model_fields), spec.name
