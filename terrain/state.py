"""The terrain canvas.

Layout convention: every 2D layer has shape (depth, width) and is indexed [z, x]. A normalized
coordinate (u, v) in [0, 1]² maps to block column x = u * (width - 1), z = v * (depth - 1).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from terrain.config import WORLD_MAX_Y, WORLD_MIN_Y


@dataclass
class Feature:
    """A named feature the agent can refer to (e.g. mountain_1). Bounds are normalized (u0, v0, u1, v1)."""

    id: str
    kind: str
    bounds: tuple[float, float, float, float]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorldState:
    height: np.ndarray  # float32 surface height (block y) per column
    water: np.ndarray  # float32 water surface level per column; NaN = no water
    # int16 surface-material override per column, indexing material_names. 0 = "auto": palette rules
    # pick the block at compile time (DESIGN §5.7); materials-stage ops write other indices.
    material: np.ndarray = None  # type: ignore[assignment]  # filled in __post_init__
    material_names: list[str] = field(default_factory=lambda: ["auto"])
    masks: dict[str, np.ndarray] = field(default_factory=dict)  # named float32 masks in [0, 1]
    features: dict[str, Feature] = field(default_factory=dict)
    # Surface palette rules (terrain/materials.py Rule dicts) set by set_palette_rules; None = defaults.
    palette_rules: list[dict[str, Any]] | None = None
    seed: int = 0  # per-op seed during replay (set by the op stack); ops seed their rng from it

    def __post_init__(self) -> None:
        if self.material is None:
            self.material = np.zeros(self.height.shape, dtype=np.int16)
        for name, layer in (("water", self.water), ("material", self.material)):
            if layer.shape != self.height.shape:
                raise ValueError(f"{name} layer shape {layer.shape} != height shape {self.height.shape}")

    @classmethod
    def flat(cls, width: int, depth: int, base_y: float = 64.0, seed: int = 0) -> WorldState:
        if width < 2 or depth < 2:
            raise ValueError(f"area must be at least 2x2 columns, got {width}x{depth}")
        if not WORLD_MIN_Y <= base_y < WORLD_MAX_Y:
            raise ValueError(f"base_y={base_y} outside world limits [{WORLD_MIN_Y}, {WORLD_MAX_Y})")
        return cls(
            height=np.full((depth, width), base_y, dtype=np.float32),
            water=np.full((depth, width), np.nan, dtype=np.float32),
            seed=seed,
        )

    @property
    def width(self) -> int:
        return self.height.shape[1]

    @property
    def depth(self) -> int:
        return self.height.shape[0]

    def copy(self) -> WorldState:
        return WorldState(
            height=self.height.copy(),
            water=self.water.copy(),
            material=self.material.copy(),
            material_names=list(self.material_names),
            masks={k: v.copy() for k, v in self.masks.items()},
            features=copy.deepcopy(self.features),
            palette_rules=copy.deepcopy(self.palette_rules),
            seed=self.seed,
        )

    def grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Normalized coordinate grids (u, v), each shaped like the layers."""
        u = np.linspace(0.0, 1.0, self.width, dtype=np.float32)
        v = np.linspace(0.0, 1.0, self.depth, dtype=np.float32)
        return np.meshgrid(u, v)  # default 'xy' indexing -> shape (depth, width)


def to_xz(state: WorldState, u, v):
    """Normalized (u, v) -> block (x, z), unrounded (the convention in the module docstring)."""
    return u * (state.width - 1), v * (state.depth - 1)


def to_uv(state: WorldState, x, z):
    """Block (x, z) -> normalized (u, v); also turns block extents into shares of the area."""
    return x / (state.width - 1), z / (state.depth - 1)
