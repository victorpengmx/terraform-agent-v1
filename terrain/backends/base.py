"""World backend interface (DESIGN §8) and an in-memory implementation for tests."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod

import numpy as np

from terrain.compiler.volume import AIR, WATER, BlockPatch, Box, Volume
from terrain.config import WORLD_MAX_Y, WORLD_MIN_Y

NON_GROUND = {AIR, WATER, "cave_air", "void_air", "short_grass", "tall_grass", "fern", "large_fern", "snow",
              "dandelion", "poppy", "seagrass", "tall_seagrass", "kelp", "kelp_plant"}  # fmt: skip


class BackendError(Exception):
    """A backend problem the user must fix (world open, chunks missing, ...)."""


class WorldBackend(ABC):
    @abstractmethod
    def read_region(self, box: Box) -> Volume: ...

    @abstractmethod
    def apply_patch(self, patch: BlockPatch) -> None: ...

    @abstractmethod
    def snapshot(self) -> str: ...

    @abstractmethod
    def restore(self, snapshot_id: str) -> None: ...

    def read_surface(self, x0: int, z0: int, x1: int, z1: int, y_range: tuple[int, int] = (WORLD_MIN_Y, WORLD_MAX_Y)) -> tuple[np.ndarray, np.ndarray]:
        """(top ground block y, top water block y or NaN) per column, arrays indexed [z, x]."""
        vol = self.read_region(Box(x0, z0, x1, z1, *y_range))
        ys = np.arange(vol.box.y0, vol.box.y1)
        ground_ids = np.array([n not in NON_GROUND for n in vol.palette])
        water_ids = np.array([n == WATER for n in vol.palette])
        ground = ground_ids[vol.data]
        wet = water_ids[vol.data]
        top = np.where(ground.any(axis=1), ys[ground.shape[1] - 1 - np.argmax(ground[:, ::-1, :], axis=1)], vol.box.y0 - 1)
        wtop = np.where(wet.any(axis=1), ys[wet.shape[1] - 1 - np.argmax(wet[:, ::-1, :], axis=1)], np.nan)
        return top.astype(np.float32), wtop.astype(np.float32)


class MemoryBackend(WorldBackend):
    """A superflat world in memory: bedrock at the floor, stone, 3 dirt, grass at `ground_y`."""

    def __init__(self, ground_y: int = 64) -> None:
        self.ground_y = ground_y
        self.palette = [AIR, "bedrock", "stone", "dirt", "grass_block", WATER]
        self.chunks: dict[tuple[int, int], np.ndarray] = {}  # (cx, cz) -> [z, y, x] (16, H, 16) ids
        self._snapshots: dict[str, dict[tuple[int, int], np.ndarray]] = {}

    def _chunk(self, cx: int, cz: int) -> np.ndarray:
        if (cx, cz) not in self.chunks:
            col = np.zeros(WORLD_MAX_Y - WORLD_MIN_Y, dtype=np.uint16)
            g = self.ground_y - WORLD_MIN_Y
            col[0] = 1
            col[1 : g - 3] = 2
            col[g - 3 : g] = 3
            col[g] = 4
            self.chunks[(cx, cz)] = np.broadcast_to(col[None, :, None], (16, len(col), 16)).copy()
        return self.chunks[(cx, cz)]

    def _id(self, name: str) -> int:
        if name not in self.palette:
            self.palette.append(name)
        return self.palette.index(name)

    def read_region(self, box: Box) -> Volume:
        data = np.empty(box.shape, dtype=np.uint16)
        ya, yb = box.y0 - WORLD_MIN_Y, box.y1 - WORLD_MIN_Y
        for cx, cz in box.chunks():
            ch = self._chunk(cx, cz)
            xa, xb = max(box.x0, cx * 16), min(box.x1, cx * 16 + 16)
            za, zb = max(box.z0, cz * 16), min(box.z1, cz * 16 + 16)
            data[za - box.z0 : zb - box.z0, :, xa - box.x0 : xb - box.x0] = ch[za - cz * 16 : zb - cz * 16, ya:yb, xa - cx * 16 : xb - cx * 16]
        return Volume(box, list(self.palette), data)

    def apply_patch(self, patch: BlockPatch) -> None:
        lut = np.array([self._id(n) for n in patch.palette], dtype=np.uint16)
        for (cx, cz), idx in patch.by_chunk().items():
            ch = self._chunk(cx, cz)
            x, y, z = patch.xyz[idx].T
            ch[z - cz * 16, y - WORLD_MIN_Y, x - cx * 16] = lut[patch.new[idx]]

    def snapshot(self) -> str:
        sid = f"mem-{len(self._snapshots)}"
        self._snapshots[sid] = copy.deepcopy(self.chunks)
        return sid

    def restore(self, snapshot_id: str) -> None:
        self.chunks = copy.deepcopy(self._snapshots[snapshot_id])
