import json

import pytest

from scripts.migrate_stacks import migrate_op, migrate_stack
from terrain.ops.erode import hydraulic_iterations, thermal_settings
from terrain.ops.peaks import sharpness
from terrain.opstack import OpStack

OLD = {
    "base": {"width": 33, "depth": 33, "base_y": 64.0, "seed": 1},
    "ops": [
        {"id": "n", "op": "noise_base", "params": {"amplitude": 3.0, "feature_size": 0.25, "octaves": 4, "roughness": 0.5}},
        {"id": "m", "op": "raise_peak", "params": {"center": [0.5, 0.5], "radius": 0.3, "height": 40.0, "sharpness": 1.6,
                                                     "irregularity": 0.6, "ridges": 0.7}},
        {"id": "r", "op": "roughen", "params": {"amount": 4.0, "scale": 0.1, "style": "ridged", "weight": "relief", "edge": 0.05,
                                                  "region": None, "octaves": 4}},
        {"id": "t", "op": "erode", "params": {"kind": "thermal", "iterations": 30, "talus_deg": 40.0, "strength": 0.5, "deposition": 0.5}},
        {"id": "h", "op": "erode", "params": {"kind": "hydraulic", "iterations": 40, "talus_deg": 40.0, "strength": 0.5, "deposition": 0.5}},
    ],
}


def test_old_stack_migrates_and_loads():
    data = json.loads(json.dumps(OLD))
    assert len(migrate_stack(data)) == 5
    stack = OpStack.from_json(json.dumps(data))
    p = {e.id: e.params for e in stack.entries}
    assert p["n"]["roughness"] == 0.5 and "octaves" not in p["n"]  # same persistence (0.5)
    assert sharpness(p["m"]["steepness"]) == 1.6 and (p["m"]["irregularity"], p["m"]["ridges"]) == (0.6, 0.7)
    assert thermal_settings(p["t"]["strength"]) == (30, 40.0)
    assert hydraulic_iterations(p["h"]["strength"]) == 40
    assert migrate_stack(data) == []  # idempotent


@pytest.mark.parametrize("old, strength", [
    ({"kind": "hydraulic", "iterations": 40, "strength": 0.4}, 0.55),  # same total cutting as 32 events at the fixed rate
    ({"kind": "thermal", "iterations": 12, "talus_deg": 62.0}, 0.1),  # between 0.06 (angle) and 0.14 (passes)
])
def test_erode_maps_to_the_nearest_strength(old, strength):
    assert migrate_op("erode", old)["strength"] == pytest.approx(strength)
