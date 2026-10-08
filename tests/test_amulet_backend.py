"""AmuletBackend integration tests against a temporary copy of a real 26.3 world.

Uses the M0 hello-world output (HelloFlat_hello) as a fixture; skipped if it isn't there.
Never touches the original: everything happens in pytest's tmp_path.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from terrain.backends import BackendError
from terrain.backends.amulet_backend import AmuletBackend
from terrain.compiler.pipeline import build_patch
from terrain.config import WORLD_MIN_Y
from terrain.ops import apply_op
from terrain.state import WorldState

SOURCE = Path(os.environ.get("APPDATA", "")) / ".minecraft" / "saves" / "HelloFlat_hello"
pytestmark = pytest.mark.skipif(not (SOURCE / "level.dat").exists(), reason="fixture world HelloFlat_hello not found")


@pytest.fixture
def backend(tmp_path):
    world = tmp_path / "saves" / "World"
    shutil.copytree(SOURCE, world, ignore=shutil.ignore_patterns("session.lock"))
    return AmuletBackend(world, snapshot_root=tmp_path / "snaps")


def test_read_surface_sees_flat_ground_and_hello_hill(backend):
    top, water = backend.read_surface(-20, -20, 70, 70, y_range=(-64, 0))
    assert top[0, 0] == -61  # flat ground away from the hill (area starts at -20, -20)
    assert top.max() > -50  # the M0 sine hill
    assert np.isnan(water).all()


def test_apply_read_back_and_restore(backend, tmp_path):
    origin = (200, 200)  # away from the hill; generated chunks in the fixture
    s = WorldState.flat(24, 24, base_y=-61)
    s = apply_op(s, "raise_peak", {"center": [0.5, 0.5], "radius": 0.5, "height": 12})
    s.height[2:6, 2:6] = -63  # a small pond
    s.water[2:6, 2:6] = -62

    patch, compiled = build_patch(s, backend, origin)
    assert len(patch) > 0
    backend.apply_patch(patch)
    snap_before = backend.last_snapshot

    after = backend.read_region(compiled.box)
    assert np.array_equal(after.in_palette(list(compiled.palette)), compiled.data)
    assert after.block(204, -62, 204) == "water"

    again, _ = build_patch(s, backend, origin)
    assert len(again) == 0  # idempotent

    backend.restore(snap_before)
    top, _ = backend.read_surface(origin[0], origin[1], origin[0] + 24, origin[1] + 24, y_range=(-64, 0))
    assert np.all(top == -61)


def _raw_sky_light(world: Path, cx: int, cz: int) -> tuple[int, dict[int, np.ndarray]]:
    """(isLightOn, {cy: sky light [y, z, x]}) straight from the chunk NBT."""
    import amulet

    level = amulet.load_level(str(world))
    try:
        nbt = level.level_wrapper.get_raw_chunk_data(cx, cz, "minecraft:overworld").compound
    finally:
        level.close()
    light = {}
    for sec in nbt["sections"]:
        if "SkyLight" in sec:
            b = sec["SkyLight"].np_array.astype(np.uint8)
            light[sec["Y"].py_int] = np.stack([b & 0xF, b >> 4], axis=1).reshape(16, 16, 16)
    return nbt["isLightOn"].py_int, light


def test_writes_exact_sky_light(backend):
    origin = (200, 200)
    s = WorldState.flat(24, 24, base_y=-61)
    s = apply_op(s, "raise_peak", {"center": [0.5, 0.5], "radius": 0.5, "height": 12})
    s.height[2:6, 2:6] = -63  # a two-deep pond
    s.water[2:6, 2:6] = -61
    patch, compiled = build_patch(s, backend, origin)
    backend.apply_patch(patch)
    vol = backend.read_region(compiled.box)

    for x, z in [(204, 204), (212, 212), (201, 220)]:  # pond, peak, flank
        is_light_on, light = _raw_sky_light(backend.world, x >> 4, z >> 4)
        assert is_light_on == 1
        y_lo, y_hi = compiled.box.y0, compiled.box.y1
        assert all(cy in light for cy in range(y_lo >> 4, ((y_hi - 1) >> 4) + 1))

        def sky(y):
            return light[y >> 4][y & 15, z & 15, x & 15]

        column = [vol.block(x, y, z) for y in range(y_lo, y_hi)]
        top = max(y for y, b in zip(range(y_lo, y_hi), column) if b != "air")
        assert sky(top + 1) == 15 and sky(y_hi + 5) == 15
        depth = 0
        for y in range(top, y_lo - 1, -1):
            if vol.block(x, y, z) != "water":
                break
            depth += 1
            assert sky(y) == 15 - depth
        assert sky(y) == 0 and sky(WORLD_MIN_Y) == 0  # first solid block and everything below
        if (x, z) == (204, 204):
            assert depth == 2


def test_refuses_open_world_and_missing_chunks(backend):
    with pytest.raises(BackendError, match="not generated"):
        backend.read_surface(100_000, 100_000, 100_010, 100_010)
    with pytest.raises(BackendError, match="not a Minecraft world"):
        AmuletBackend(backend.world.parent)
