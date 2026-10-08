"""AmuletBackend (DESIGN §8): offline edits of a Java 26.3 world with amulet-core.

Safety rules (CLAUDE.md): never write to a world that may be open; write to a copy and swap.
- `apply_patch`: copy world -> <world>.tf-tmp, write there, save; then move the current world into the
  snapshot store and move the copy into place. The previous version is therefore always a snapshot.
- Snapshots live outside `saves/` (in `<.minecraft>/terraform_snapshots/<world>/`) so they don't clutter
  the in-game world list, but on the same drive so swaps are renames, not copies.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from terrain.backends.base import BackendError, WorldBackend
from terrain.compiler.volume import AIR, BlockPatch, Box, Volume
from terrain.config import WORLD_MAX_Y, WORLD_MIN_Y

DIMENSION = "minecraft:overworld"


def assert_world_closed(world: Path) -> None:
    """A running game holds an OS lock on session.lock; reading it then fails on Windows."""
    lock = world / "session.lock"
    if lock.exists():
        try:
            lock.read_bytes()
        except OSError as e:
            raise BackendError(f"{world.name} appears to be open in Minecraft ({e}). Quit to the title screen first.") from None


class AmuletBackend(WorldBackend):
    def __init__(self, world_dir: Path | str, snapshot_root: Path | str | None = None) -> None:
        self.world = Path(world_dir).resolve()
        if not (self.world / "level.dat").exists():
            raise BackendError(f"{self.world} is not a Minecraft world (no level.dat)")
        root = Path(snapshot_root) if snapshot_root else self.world.parent.parent / "terraform_snapshots"
        self.snapshots = (root / self.world.name).resolve()

    # ------------------------------------------------------------ reading

    def read_region(self, box: Box) -> Volume:
        assert_world_closed(self.world)
        level = self._load(self.world)
        try:
            names = _PaletteNames(level)
            data = np.zeros(box.shape, dtype=np.uint16)
            palette: list[str] = [AIR]
            lut: dict[str, int] = {AIR: 0}
            for cx, cz in box.chunks():
                chunk = self._get_chunk(level, cx, cz)
                xa, xb = max(box.x0, cx * 16), min(box.x1, cx * 16 + 16)
                za, zb = max(box.z0, cz * 16), min(box.z1, cz * 16 + 16)
                for cy in range(box.y0 >> 4, ((box.y1 - 1) >> 4) + 1):
                    ya, yb = max(box.y0, cy * 16), min(box.y1, cy * 16 + 16)
                    if not chunk.blocks.has_section(cy):
                        continue  # missing section = air
                    sec = chunk.blocks.get_sub_chunk(cy)[xa - cx * 16 : xb - cx * 16, ya - cy * 16 : yb - cy * 16, za - cz * 16 : zb - cz * 16]
                    uniq, inv = np.unique(sec, return_inverse=True)
                    ids = np.empty(len(uniq), dtype=np.uint16)
                    for k, u in enumerate(uniq):
                        n = names(int(u))
                        if n not in lut:
                            lut[n] = len(palette)
                            palette.append(n)
                        ids[k] = lut[n]
                    block = ids[inv].reshape(sec.shape)  # [x, y, z]
                    data[za - box.z0 : zb - box.z0, ya - box.y0 : yb - box.y0, xa - box.x0 : xb - box.x0] = block.transpose(2, 1, 0)
            return Volume(box, palette, data)
        finally:
            level.close()

    # ------------------------------------------------------------ writing

    def apply_patch(self, patch: BlockPatch) -> None:
        if len(patch) == 0:
            return
        if not (WORLD_MIN_Y <= patch.xyz[:, 1].min() and patch.xyz[:, 1].max() < WORLD_MAX_Y):
            raise BackendError("patch has blocks outside the world height limits")
        assert_world_closed(self.world)
        tmp = self.world.with_name(self.world.name + ".tf-tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(self.world, tmp, ignore=shutil.ignore_patterns("session.lock"))
        try:
            level = self._load(tmp)
            try:
                ids = self._universal_ids(level, patch.palette)
                sky = _SkyAttenuation(_PaletteNames(level))
                for (cx, cz), idx in patch.by_chunk().items():
                    chunk = self._get_chunk(level, cx, cz)
                    x, y, z = patch.xyz[idx].T
                    new = ids[patch.new[idx]]
                    cys = y >> 4
                    for cy in np.unique(cys):
                        m = cys == cy
                        sec = chunk.blocks.get_sub_chunk(int(cy))  # created (as air) if missing
                        sec[x[m] - cx * 16, y[m] - cy * 16, z[m] - cz * 16] = new[m]
                    _write_sky_light(chunk, sky)
                    chunk.changed = True
                # amulet-core 1.9.49 (pinned): AnvilFormat._calculate_light, run from pre_save_operation,
                # drops our light and writes isLightOn=0; 26.3 then relights only some sections, leaving
                # mountains dark. Disable it on this instance only. Revisit if amulet-core is upgraded.
                level.level_wrapper._calculate_light = _keep_light
                level.save()
            finally:
                level.close()
            self._swap_in(tmp)
        except Exception:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
            raise

    # ------------------------------------------------------------ snapshots

    def snapshot(self) -> str:
        assert_world_closed(self.world)
        sid = self._next_snapshot_id()
        shutil.copytree(self.world, self.snapshots / sid, ignore=shutil.ignore_patterns("session.lock"))
        return sid

    def restore(self, snapshot_id: str) -> None:
        src = self.snapshots / snapshot_id
        if not (src / "level.dat").exists():
            raise BackendError(f"unknown snapshot {snapshot_id!r}; have: {', '.join(self.list_snapshots()) or 'none'}")
        assert_world_closed(self.world)
        tmp = self.world.with_name(self.world.name + ".tf-tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(src, tmp)  # keep the snapshot itself reusable
        self._swap_in(tmp)

    def list_snapshots(self) -> list[str]:
        return sorted(p.name for p in self.snapshots.glob("*") if (p / "level.dat").exists()) if self.snapshots.exists() else []

    # ------------------------------------------------------------ internals

    def _swap_in(self, tmp: Path) -> None:
        """Current world -> new snapshot; tmp -> world. Both are renames on the same drive."""
        self.snapshots.mkdir(parents=True, exist_ok=True)
        backup = self.snapshots / self._next_snapshot_id()
        shutil.move(str(self.world), str(backup))
        try:
            shutil.move(str(tmp), str(self.world))
        except Exception:
            shutil.move(str(backup), str(self.world))  # put the original back
            raise
        self.last_snapshot = backup.name

    def _next_snapshot_id(self) -> str:
        n = len(self.list_snapshots())
        while (self.snapshots / f"{n:04d}").exists():
            n += 1
        return f"{n:04d}"

    @staticmethod
    def _load(path: Path):
        import amulet

        return amulet.load_level(str(path))

    @staticmethod
    def _get_chunk(level, cx: int, cz: int):
        from amulet.api.errors import ChunkDoesNotExist, ChunkLoadError

        try:
            return level.get_chunk(cx, cz, DIMENSION)
        except (ChunkDoesNotExist, ChunkLoadError):
            raise BackendError(
                f"chunk ({cx}, {cz}) (blocks x={cx * 16}..{cx * 16 + 15}, z={cz * 16}..{cz * 16 + 15}) is not generated. "
                "Fly over the area in game first, then quit to the title screen."
            ) from None

    @staticmethod
    def _universal_ids(level, names: list[str]) -> np.ndarray:
        from amulet.api.block import Block
        from amulet_nbt import StringTag

        platform, version = level.level_wrapper.max_world_version
        translator = level.translation_manager.get_version(platform, version).block
        out = np.empty(len(names), dtype=np.uint32)
        for i, name in enumerate(names):
            props = {"level": StringTag("0")} if name == "water" else {}
            namespace, _, base = name.rpartition(":")
            universal, _, _ = translator.to_universal(Block(namespace or "minecraft", base, props))
            out[i] = level.block_palette.get_add_block(universal)
        return out


class _PaletteNames:
    """Cached universal palette id -> Java block name (without the minecraft: namespace)."""

    def __init__(self, level) -> None:
        platform, version = level.level_wrapper.max_world_version
        self.translator = level.translation_manager.get_version(platform, version).block
        self.palette = level.block_palette
        self.cache: dict[int, str] = {}

    def __call__(self, uid: int) -> str:
        if uid not in self.cache:
            block = self.translator.from_universal(self.palette[uid])[0]
            ns = getattr(block, "namespace", "minecraft")
            base = getattr(block, "base_name", "unknown")
            self.cache[uid] = base if ns == "minecraft" else f"{ns}:{base}"
        return self.cache[uid]


# ---------------------------------------------------------------- sky light
# Terrain is 2.5D (no overhangs), so straight-down sky light is exact: per column, start at 15 at the
# top of the world, clear blocks keep it, water/leaves/ice dim it by 1 per block, anything else is
# opaque (0, and everything below stays 0). Unknown blocks count as opaque.

_SKY_CLEAR = frozenset(
    {
        "air", "cave_air", "void_air", "short_grass", "tall_grass", "fern", "large_fern", "snow", "dead_bush",
        "dandelion", "poppy", "blue_orchid", "allium", "azure_bluet", "red_tulip", "orange_tulip", "white_tulip",
        "pink_tulip", "oxeye_daisy", "cornflower", "lily_of_the_valley", "torchflower", "wither_rose",
        "sunflower", "lilac", "rose_bush", "peony", "pink_petals", "sugar_cane",
    }
)  # fmt: skip
_SKY_DIM = frozenset({"water", "ice", "seagrass", "tall_seagrass", "kelp", "kelp_plant", "bubble_column"})
_OPAQUE = 15


class _SkyAttenuation:
    """Cached universal palette id -> sky light lost per block (0, 1 or 15)."""

    def __init__(self, names: _PaletteNames) -> None:
        self.names = names
        self.cache: dict[int, int] = {}

    def __call__(self, uid: int) -> int:
        if uid not in self.cache:
            n = self.names(uid)
            self.cache[uid] = 0 if n in _SKY_CLEAR else 1 if (n in _SKY_DIM or n.endswith("_leaves")) else _OPAQUE
        return self.cache[uid]


def _write_sky_light(chunk, att_of: _SkyAttenuation) -> None:
    """Exact vertical sky light for the whole chunk height into chunk.misc (Amulet's light layout)."""
    cy0, cy1 = WORLD_MIN_Y >> 4, WORLD_MAX_Y >> 4
    att = np.zeros((16, (cy1 - cy0) * 16, 16), dtype=np.uint8)  # [x, y, z], y from WORLD_MIN_Y
    for cy in range(cy0, cy1):
        if not chunk.blocks.has_section(cy):
            continue  # missing section = air
        uniq, inv = np.unique(chunk.blocks.get_sub_chunk(cy), return_inverse=True)
        lut = np.array([att_of(int(u)) for u in uniq], dtype=np.uint8)
        att[:, (cy - cy0) * 16 : (cy - cy0 + 1) * 16, :] = lut[inv].reshape(16, 16, 16)
    lost = np.cumsum(att[:, ::-1, :], axis=1, dtype=np.int32)[:, ::-1, :]  # light lost from the top down to here
    light = np.clip(15 - lost, 0, 15).astype(np.uint8)
    sky, block = {}, {}
    for cy in range(cy0, cy1):
        sec = light[:, (cy - cy0) * 16 : (cy - cy0 + 1) * 16, :]
        sky[cy] = np.ascontiguousarray(sec.transpose(1, 2, 0))  # [y, z, x]: vanilla nibble order
        block[cy] = np.zeros((16, 16, 16), dtype=np.uint8)  # terrain has no light sources
    chunk.misc["sky_light"] = sky
    chunk.misc["block_light"] = block
    chunk.misc["isLightOn"] = 1


def _keep_light(level, chunks):
    """No-op replacement for AnvilFormat._calculate_light (a generator returning 'changed')."""
    return False
    yield
