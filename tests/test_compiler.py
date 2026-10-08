import numpy as np
import pytest

from terrain.backends import MemoryBackend
from terrain.compiler import AIR, WATER, Box, compile_state, diff, surface_y, water_top
from terrain.compiler.pipeline import build_patch
from terrain.materials import Banded, Condition, Rule, Single
from terrain.ops import apply_op
from terrain.state import WorldState

N = 33


def flat(y=64.0):
    return WorldState.flat(N, N, base_y=y)


def rules(state, *rs):
    state.palette_rules = [r.model_dump(mode="json") for r in rs]
    return state


def test_flat_column_layering():
    vol = compile_state(rules(flat(), Rule(use=Single(block="grass_block"))), y_range=(55, 70))
    col = vol.column(5, 5)  # y 55..69
    assert col[64 - 55] == "grass_block"
    assert col[61 - 55 : 64 - 55] == ["dirt"] * 3
    assert set(col[: 61 - 55]) == {"stone"}
    assert set(col[65 - 55 :]) == {AIR}


def test_heights_are_rounded_only_at_compile():
    s = flat(64.4)
    s.height[3, 3] = 64.6
    vol = compile_state(rules(s, Rule(use=Single(block="sand"))), y_range=(60, 70))
    assert vol.block(0, 64, 0) == "sand" and vol.block(0, 65, 0) == AIR
    assert vol.block(3, 65, 3) == "sand"


def test_water_fills_above_ground_only():
    s = flat()
    s.height[10:20, 10:20] = 58
    s.water[10:20, 10:20] = 63.0
    vol = compile_state(s, y_range=(55, 70))
    col = vol.column(15, 15)
    assert col[58 - 55] != WATER and col[59 - 55 : 64 - 55] == [WATER] * 5 and col[64 - 55] == AIR


def test_water_never_floats_after_rounding():
    s = apply_op(flat(80), "noise_base", {"amplitude": 6}, seed=1)
    s = apply_op(s, "carve_river", {"path": [[0, 0.5], [1, 0.5]], "width": 0.15, "depth": 4, "meander": 0.5}, seed=2)
    vol = compile_state(s)
    d = vol.data
    water_id, air_id = vol.palette.index(WATER), vol.palette.index(AIR)
    assert not np.any((d[:, 1:, :] == water_id) & (d[:, :-1, :] == air_id))  # never air below water
    # Beside a water block, a dry column must be at least as high (a river may step down onto lower water).
    h = surface_y(s)
    wt = water_top(s)
    wet = wt > h
    for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        src = (slice(max(-dz, 0), N + min(-dz, 0)), slice(max(-dx, 0), N + min(-dx, 0)))
        nbr = (slice(max(dz, 0), N + min(dz, 0)), slice(max(dx, 0), N + min(dx, 0)))
        bad = wet[src] & ~wet[nbr] & (h[nbr] < wt[src])
        assert not bad.any(), (dz, dx)


def test_cliff_face_shows_the_rule_block():
    s = flat()
    s.height[:, 16:] = 84  # a 20-block wall facing west at x=16
    s = rules(s, Rule(when=Condition(slope_min=50), use=Banded(blocks=["tuff"], noise=0)), Rule(use=Single(block="grass_block")))
    vol = compile_state(s)
    wall = vol.column(16, 10)
    face = wall[64 - vol.box.y0 + 1 : 84 - vol.box.y0 + 1]  # y 65..84 is exposed
    assert set(face) == {"tuff"}
    assert vol.block(30, 84, 10) == "grass_block"  # plateau top stays grass


def test_y_range_within_world_limits():
    vol = compile_state(flat(-60))
    assert vol.box.y0 >= -63  # bedrock floor untouched


def test_diff_contains_only_changes_and_inverts():
    a = compile_state(flat(), y_range=(60, 70))
    s = flat()
    s.height[5, 5] = 66
    b = compile_state(s, y_range=(60, 70))
    patch = diff(b, a)
    # Only column (5, 5) changes: two new blocks on top, and its soil layers shift up by two.
    assert set(map(tuple, patch.xyz[:, [0, 2]].tolist())) == {(5, 5)}
    assert {65, 66} <= set(patch.xyz[:, 1].tolist()) and len(patch) == 4
    inv = patch.inverted()
    assert np.array_equal(inv.new, patch.old)


def test_memory_backend_apply_and_undo():
    backend = MemoryBackend(ground_y=64)
    s = apply_op(flat(), "raise_peak", {"center": [0.5, 0.5], "radius": 0.4, "height": 20})
    snap = backend.snapshot()
    patch, vol = build_patch(s, backend, origin=(100, -40))
    assert len(patch) > 0
    backend.apply_patch(patch)
    after = backend.read_region(vol.box)
    assert np.array_equal(after.in_palette(list(vol.palette)), vol.data)
    top, _ = backend.read_surface(100, -40, 133, -7)
    assert top.max() == 84 and top[0, 0] == 64
    backend.apply_patch(patch.inverted())
    restored, _ = backend.read_surface(100, -40, 133, -7)
    assert np.all(restored == 64)
    backend.restore(snap)


def test_rebuild_against_applied_world_is_empty():
    backend = MemoryBackend(ground_y=64)
    s = apply_op(flat(), "carve_valley", {"path": [[0, 0.5], [1, 0.5]], "width": 0.3, "depth": 10})
    patch, _ = build_patch(s, backend, origin=(0, 0))
    backend.apply_patch(patch)
    again, _ = build_patch(s, backend, origin=(0, 0))
    assert len(again) == 0
