"""Op registry: one pydantic params model per op is both the validator and the LLM tool schema (DESIGN §5).

    @register_op("raise_peak", params=RaisePeakParams, stage="terrain")
    def raise_peak(state: WorldState, p: RaisePeakParams) -> WorldState:
        ...  # the function's docstring becomes the tool description

Callers go through `apply_op`, which validates raw params, runs the op on a copy (so the caller's
state is never mutated) and checks the result against world limits. Every failure the agent can fix
is raised as `OpError` with a message written for the agent.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Literal, get_args

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from terrain.config import WORLD_MAX_Y, WORLD_MIN_Y
from terrain.state import WorldState

Stage = Literal["terrain", "water", "voxel3d", "materials", "foliage"]
STAGES: tuple[Stage, ...] = get_args(Stage)  # replay order (DESIGN §3.2)

# Shared param types. Normalized coordinates are fractions of the target area.
Norm = Annotated[float, Field(ge=0.0, le=1.0)]
Point = tuple[Norm, Norm]


class OpParams(BaseModel):
    """Base for all op params: unknown fields are rejected, not ignored (fail loudly)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class OpError(Exception):
    """An error the agent can act on. The message is returned to the model verbatim."""


OpFn = Callable[[WorldState, Any], WorldState]


@dataclass(frozen=True)
class OpSpec:
    name: str
    fn: OpFn
    params: type[OpParams]
    stage: Stage
    description: str

    def tool_schema(self) -> dict[str, Any]:
        """Anthropic-style tool definition: {name, description, input_schema}."""
        return {"name": self.name, "description": self.description, "input_schema": self.params.model_json_schema()}


_REGISTRY: dict[str, OpSpec] = {}


def register_op(name: str, *, params: type[OpParams], stage: Stage) -> Callable[[OpFn], OpFn]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    if not (isinstance(params, type) and issubclass(params, OpParams)):
        raise TypeError(f"params for op {name!r} must subclass OpParams")

    def decorator(fn: OpFn) -> OpFn:
        if name in _REGISTRY:
            raise ValueError(f"op {name!r} is already registered")
        doc = inspect.getdoc(fn)
        if not doc:
            raise ValueError(f"op {name!r} needs a docstring; it becomes the tool description")
        _REGISTRY[name] = OpSpec(name=name, fn=fn, params=params, stage=stage, description=doc)
        return fn

    return decorator


def get_op(name: str) -> OpSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise OpError(f"unknown op {name!r}. Available ops: {', '.join(sorted(_REGISTRY))}") from None


def list_ops(stage: Stage | None = None) -> list[OpSpec]:
    """Ops sorted by stage order then name, so generated tool lists are deterministic (prompt caching)."""
    ops = [s for s in _REGISTRY.values() if stage is None or s.stage == stage]
    return sorted(ops, key=lambda s: (STAGES.index(s.stage), s.name))


def tool_schemas(stage: Stage | None = None) -> list[dict[str, Any]]:
    return [s.tool_schema() for s in list_ops(stage)]


def validate_params(name: str, raw: dict[str, Any]) -> OpParams:
    spec = get_op(name)
    try:
        return spec.params.model_validate(raw)
    except ValidationError as e:
        problems = "; ".join(
            f"{'.'.join(map(str, err['loc'])) or '(params)'}: {err['msg']}" for err in e.errors(include_url=False)
        )
        raise OpError(f"invalid params for {name}: {problems}") from None


def apply_op(state: WorldState, name: str, raw_params: dict[str, Any], seed: int | None = None) -> WorldState:
    """Validate and run one op on a copy of `state`. `seed` (set by the op stack) seeds the op's rng."""
    spec = get_op(name)
    p = validate_params(name, raw_params)
    work = state.copy()
    if seed is not None:
        work.seed = seed
    result = spec.fn(work, p)
    _check_limits(name, result)
    return result


def _check_limits(name: str, state: WorldState) -> None:
    h = state.height
    if not np.isfinite(h).all():
        raise OpError(f"{name} produced non-finite heights")  # an op bug, but still surfaced loudly
    lo, hi = float(h.min()), float(h.max())
    top = WORLD_MAX_Y - 1
    if hi > top:
        raise OpError(f"{name} would raise terrain to y={hi:.0f}, above the build limit y={top}; reduce it by at least {hi - top:.0f} blocks")
    if lo < WORLD_MIN_Y:
        raise OpError(f"{name} would lower terrain to y={lo:.0f}, below the world floor y={WORLD_MIN_Y}; reduce it by at least {WORLD_MIN_Y - lo:.0f} blocks")
