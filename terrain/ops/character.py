"""`character` presets (DESIGN §5.3): named bundles of calibrated 0-1 scales.

The model says "rugged"; code maps `roughness` / `steepness` onto each op's internals. Explicit values
override the preset, and the op stack stores the resolved values, so replay never depends on this table.

Ops whose shape has no steepness of its own (noise_base: amplitude and size already set the slope) use
`RoughnessParams`; the others use `CharacterParams`. An op can derive further optional scales from
these in `derive` (raise_peak: roughness -> irregularity, ridges); `DERIVED` lists them so `update_op`
re-derives a value when its source changes instead of keeping the stale resolved one.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import Field, model_validator

from terrain.ops.registry import Norm, OpParams

Character = Literal["gentle", "rolling", "rugged", "jagged"]
CHARACTER: dict[str, dict[str, float]] = {
    "gentle": {"roughness": 0.20, "steepness": 0.25},
    "rolling": {"roughness": 0.35, "steepness": 0.40},
    "rugged": {"roughness": 0.55, "steepness": 0.60},
    "jagged": {"roughness": 0.80, "steepness": 0.85},
}
DEFAULT_CHARACTER: Character = "rugged"
SCALES = ("roughness", "steepness")


class RoughnessParams(OpParams):
    character: Character | None = Field(
        default=None, description="Preset: gentle, rolling, rugged (default) or jagged. Explicit scales below override it.")
    roughness: Norm | None = Field(default=None, description="0 = regular and smooth, 1 = broken and irregular. Overrides character.")

    # Which stored values are derived from which: changing the key re-derives the listed values.
    DERIVED: ClassVar[dict[str, tuple[str, ...]]] = {"character": SCALES}

    @model_validator(mode="before")
    @classmethod
    def resolve_character(cls, data: Any) -> Any:
        """Fill the scales from the preset (and derived values from the scales), so the stored params are
        fully resolved."""
        if not isinstance(data, dict):
            return data
        scales = [k for k in SCALES if k in cls.model_fields]
        name = data.get("character")
        if name is None and any(data.get(k) is None for k in scales):
            name = DEFAULT_CHARACTER
        if name is not None:
            if name not in CHARACTER:
                return data  # field validation reports the bad name
            data = {**data, "character": name, **{k: CHARACTER[name][k] for k in scales if data.get(k) is None}}
        return cls.derive(data)

    @classmethod
    def derive(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Fill op-specific values derived from the resolved scales (only where not given)."""
        return data

    @classmethod
    def stale_on_update(cls, changed: set[str]) -> set[str]:
        """Stored values that must be re-derived when `changed` keys are updated (unless also given)."""
        stale, todo = set(), list(changed)
        while todo:
            for k in cls.DERIVED.get(todo.pop(), ()):
                if k not in stale and k not in changed:
                    stale.add(k)
                    todo.append(k)
        return stale


class CharacterParams(RoughnessParams):
    steepness: Norm | None = Field(default=None, description="0 = soft, 1 = sheer. Overrides character.")
