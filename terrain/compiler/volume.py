"""Dense block volumes and the changed-blocks patch between two of them (DESIGN §7)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AIR = "air"
WATER = "water"


@dataclass(frozen=True)
class Box:
    """World-coordinate box; max bounds exclusive."""

    x0: int
    z0: int
    x1: int
    z1: int
    y0: int
    y1: int

    @property
    def shape(self) -> tuple[int, int, int]:  # [z, y, x], like Volume.data
        return self.z1 - self.z0, self.y1 - self.y0, self.x1 - self.x0

    def chunks(self) -> list[tuple[int, int]]:
        return [(cx, cz) for cx in range(self.x0 >> 4, ((self.x1 - 1) >> 4) + 1) for cz in range(self.z0 >> 4, ((self.z1 - 1) >> 4) + 1)]


@dataclass
class Volume:
    """Blocks in a box: `data[z, y, x]` holds indices into `palette` (block names without namespace)."""

    box: Box
    palette: list[str]
    data: np.ndarray  # uint16

    def __post_init__(self) -> None:
        if self.data.shape != self.box.shape:
            raise ValueError(f"volume data shape {self.data.shape} != box shape {self.box.shape}")

    def id_of(self, name: str) -> int:
        try:
            return self.palette.index(name)
        except ValueError:
            self.palette.append(name)
            return len(self.palette) - 1

    def block(self, x: int, y: int, z: int) -> str:
        b = self.box
        return self.palette[self.data[z - b.z0, y - b.y0, x - b.x0]]

    def column(self, x: int, z: int) -> list[str]:
        """Block names bottom to top at world column (x, z)."""
        b = self.box
        return [self.palette[i] for i in self.data[z - b.z0, :, x - b.x0]]

    def in_palette(self, palette: list[str]) -> np.ndarray:
        """This volume's data re-indexed into `palette` (names missing there are appended to it)."""
        for name in self.palette:
            if name not in palette:
                palette.append(name)
        lut = np.array([palette.index(n) for n in self.palette], dtype=np.uint16)
        return lut[self.data]


@dataclass
class BlockPatch:
    """Changed blocks only. `old` makes every patch its own undo record."""

    palette: list[str]
    xyz: np.ndarray  # (n, 3) int32 world coordinates
    new: np.ndarray  # (n,) uint16 palette ids
    old: np.ndarray  # (n,) uint16 palette ids

    def __len__(self) -> int:
        return len(self.new)

    def inverted(self) -> BlockPatch:
        return BlockPatch(self.palette, self.xyz, self.old, self.new)

    def by_chunk(self) -> dict[tuple[int, int], np.ndarray]:
        """Chunk (cx, cz) -> indices into this patch."""
        keys = (self.xyz[:, 0] >> 4).astype(np.int64) * 1_000_003 + (self.xyz[:, 2] >> 4)
        order = np.argsort(keys, kind="stable")
        uniq, starts = np.unique(keys[order], return_index=True)
        groups = np.split(order, starts[1:])
        return {(int(self.xyz[g[0], 0] >> 4), int(self.xyz[g[0], 2] >> 4)): g for g in groups}

    def summary(self) -> dict[str, int]:
        """Count of placed blocks by name."""
        names, counts = np.unique(self.new, return_counts=True)
        return {self.palette[n]: int(c) for n, c in zip(names, counts)}


def diff(new: Volume, old: Volume) -> BlockPatch:
    if new.box != old.box:
        raise ValueError(f"cannot diff volumes over different boxes: {new.box} vs {old.box}")
    palette = list(new.palette)
    old_data = old.in_palette(palette)
    changed = new.data != old_data
    z, y, x = np.nonzero(changed)
    b = new.box
    xyz = np.stack([x + b.x0, y + b.y0, z + b.z0], axis=1).astype(np.int32)
    return BlockPatch(palette, xyz, new.data[changed].astype(np.uint16), old_data[changed])
