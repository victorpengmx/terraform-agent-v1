"""Op stack: the terrain is an ordered list of ops, replayed deterministically.

- Entries run grouped by stage (`STAGES` order), in insertion order within a stage.
- Each op's seed is derived from the base seed and the op's own id, so editing or adding other ops
  never changes an op's noise.
- After each op, the changed area is recorded as a named feature (id = op id) with a footprint mask
  `masks["feature:<id>"]`, so later ops and the agent can refer to "mountain_1".
- Appending to the latest stage applies one op to the cached state; any other edit replays the stack.
  Every mutation is all-or-nothing: if an op fails, the stack and state are unchanged.
"""

from __future__ import annotations

import json
import re
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from terrain.ops import STAGES, OpError, apply_op, get_op
from terrain.ops.registry import validate_params
from terrain.state import Feature, WorldState, to_uv

ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
CHANGE_EPS = 0.01  # blocks; smaller height/water changes don't count toward a feature footprint


@dataclass(frozen=True)
class BaseSpec:
    width: int
    depth: int
    base_y: float = 64.0
    seed: int = 0

    def make_state(self) -> WorldState:
        return WorldState.flat(self.width, self.depth, base_y=self.base_y, seed=self.seed)


@dataclass(frozen=True)
class OpEntry:
    id: str
    op: str
    params: dict[str, Any]  # validated, JSON-safe (pydantic mode="json")
    feature: str | None = None  # the spec feature this op builds (DESIGN §4.3); not an op parameter

    @property
    def stage_index(self) -> int:
        return STAGES.index(get_op(self.op).stage)


@dataclass(frozen=True)
class Checkpoint:
    entries: tuple[OpEntry, ...]
    state: WorldState


def op_seed(base_seed: int, op_id: str) -> int:
    return zlib.crc32(f"{base_seed}:{op_id}".encode())


class OpStack:
    def __init__(self, base: BaseSpec) -> None:
        self.base = base
        self._entries: list[OpEntry] = []
        self._state = base.make_state()

    # ------------------------------------------------------------ read

    @property
    def state(self) -> WorldState:
        """Current state. Treat as read-only; edit the terrain through the stack."""
        return self._state

    @property
    def entries(self) -> tuple[OpEntry, ...]:
        """Entries in replay order (stage, then insertion)."""
        return tuple(sorted(self._entries, key=lambda e: e.stage_index))  # sort is stable

    def get(self, op_id: str) -> OpEntry:
        for e in self._entries:
            if e.id == op_id:
                return e
        raise OpError(f"unknown op id {op_id!r}. Existing ids: {', '.join(e.id for e in self.entries) or 'none'}")

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------ edit

    def add(self, op: str, params: dict[str, Any], id: str | None = None, feature: str | None = None) -> OpEntry:
        entry = self._make_entry(op, params, id, feature=feature)
        if all(e.stage_index <= entry.stage_index for e in self._entries):
            state = self._apply(self._state, entry)  # incremental: it runs last anyway
        else:
            state = self._replay([*self._entries, entry])
        self._entries.append(entry)
        self._state = state
        return entry

    def update(self, op_id: str, params: dict[str, Any], feature: str | None | object = ...) -> OpEntry:
        """Change some params of an existing op (unspecified params keep their values) and replay.

        Stored values derived from a changed param (e.g. roughness from a new `character`) are re-derived.
        `feature` retags the op (None clears the tag; omitted keeps it); a retag replays too, so the spec
        features' footprints stay right."""
        old = self.get(op_id)
        new_feature = old.feature if feature is ... else feature
        stale_on_update = getattr(get_op(old.op).params, "stale_on_update", None)
        stale = stale_on_update(set(params)) if stale_on_update else set()
        kept = {k: v for k, v in old.params.items() if k not in stale}
        merged = validate_params(old.op, {**kept, **params}).model_dump(mode="json")
        new = OpEntry(id=old.id, op=old.op, params=merged, feature=new_feature)  # type: ignore[arg-type]
        entries = [new if e.id == op_id else e for e in self._entries]
        self._state = self._replay(entries)
        self._entries = entries
        return new

    def remove(self, op_id: str) -> OpEntry:
        old = self.get(op_id)
        entries = [e for e in self._entries if e.id != op_id]
        self._state = self._replay(entries)
        self._entries = entries
        return old

    def replay(self) -> WorldState:
        """Fresh full replay of the current stack (does not change the cache)."""
        return self._replay(self._entries)

    # ------------------------------------------------------------ checkpoints and serialization

    def checkpoint(self) -> Checkpoint:
        return Checkpoint(entries=tuple(self._entries), state=self._state.copy())

    def restore(self, cp: Checkpoint) -> None:
        self._entries = list(cp.entries)
        self._state = cp.state.copy()

    def to_json(self) -> str:
        base = {"width": self.base.width, "depth": self.base.depth, "base_y": self.base.base_y, "seed": self.base.seed}
        ops = [{"id": e.id, "op": e.op, "params": e.params} | ({"feature": e.feature} if e.feature else {}) for e in self._entries]
        return json.dumps({"base": base, "ops": ops}, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> OpStack:
        data = json.loads(text)
        stack = cls(BaseSpec(**data["base"]))
        entries: list[OpEntry] = []
        for o in data["ops"]:
            entries.append(stack._make_entry(o["op"], o["params"], o["id"], existing=entries, feature=o.get("feature")))
        stack._state = stack._replay(entries)
        stack._entries = entries
        return stack

    # ------------------------------------------------------------ internals

    def _make_entry(self, op: str, params: dict[str, Any], op_id: str | None, existing: list[OpEntry] | None = None,
                    feature: str | None = None) -> OpEntry:
        existing = self._entries if existing is None else existing
        get_op(op)  # unknown op -> OpError listing the available ones
        validated = validate_params(op, params).model_dump(mode="json")
        if op_id is None:
            n = 1 + sum(e.op == op for e in existing)
            while any(e.id == f"{op}_{n}" for e in existing):
                n += 1
            op_id = f"{op}_{n}"
        elif not ID_PATTERN.match(op_id):
            raise OpError(f"invalid id {op_id!r}: use lowercase letters, digits and _ (e.g. 'mountain_1'), max 40 chars")
        if any(e.id == op_id for e in existing):
            raise OpError(f"id {op_id!r} is already used; pick another or update_op it")
        if feature is not None and not ID_PATTERN.match(feature):
            raise OpError(f"invalid feature {feature!r}: use a spec feature id such as 'river_1'")
        return OpEntry(id=op_id, op=op, params=validated, feature=feature)

    def _replay(self, entries: list[OpEntry]) -> WorldState:
        state = self.base.make_state()
        for e in sorted(entries, key=lambda e: e.stage_index):
            state = self._apply(state, e)
        return state

    def _apply(self, state: WorldState, entry: OpEntry) -> WorldState:
        try:
            after = apply_op(state, entry.op, entry.params, seed=op_seed(self.base.seed, entry.id))
        except OpError as err:
            raise OpError(f"{entry.id}: {err}") from None
        after.seed = self.base.seed
        _record_feature(state, after, entry)
        if entry.feature:  # the spec feature's footprint: the union of its tagged ops (regions can name it)
            key, mask = f"spec:{entry.feature}", after.masks.get(f"feature:{entry.id}")
            if mask is not None:
                after.masks[key] = np.maximum(after.masks[key], mask) if key in after.masks else mask.copy()
        return after


def _record_feature(before: WorldState, after: WorldState, entry: OpEntry) -> None:
    dh = np.abs(after.height - before.height)
    water_changed = (np.isnan(after.water) != np.isnan(before.water)) | (
        np.abs(np.nan_to_num(after.water) - np.nan_to_num(before.water)) > CHANGE_EPS
    )
    changed = (dh > CHANGE_EPS) | water_changed
    after.masks.pop(f"feature:{entry.id}", None)
    after.features.pop(entry.id, None)
    if not changed.any():
        return

    weight = np.where(dh > CHANGE_EPS, dh / max(float(dh.max()), CHANGE_EPS), 0.0)
    weight = np.maximum(weight, water_changed.astype(np.float32)).astype(np.float32)
    zs, xs = np.nonzero(changed)
    (u0, v0), (u1, v1) = to_uv(after, xs.min(), zs.min()), to_uv(after, xs.max(), zs.max())
    after.masks[f"feature:{entry.id}"] = weight
    after.features[entry.id] = Feature(
        id=entry.id,
        kind=entry.op,
        bounds=(round(u0, 3), round(v0, 3), round(u1, 3), round(v1, 3)),
        meta={
            "columns": int(changed.sum()),
            "top_y": round(float(after.height[changed].max()), 1),
            "bottom_y": round(float(after.height[changed].min()), 1),
            "water": bool(water_changed.any()),
        },
    )
