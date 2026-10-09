"""Session: the agent's working world (DESIGN §3, §14 row 1).

A thin agent-facing wrapper around the `OpStack` (the source of truth). Tools close over a session, so
no tool touches globals. Snapshots are op-stack checkpoints, used for score-gated rollback.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from terrain.config import MAX_MEAN_DH
from terrain.ops import OpError, get_op
from terrain.ops.region import region_shape
from terrain.ops.registry import validate_params
from terrain.opstack import BaseSpec, Checkpoint, OpEntry, OpStack
from terrain.state import WorldState

Snapshot = Checkpoint  # name used by the harness
OpRecord = OpEntry


class Session:
    def __init__(self, stack: OpStack, max_changed_volume: float | None = None) -> None:
        """`max_changed_volume`: guardrail on sum |Δh| per agent call (DESIGN §5.8); default from config."""
        self.stack = stack
        base = stack.base
        self.max_changed_volume = max_changed_volume if max_changed_volume is not None else MAX_MEAN_DH * base.width * base.depth

    @classmethod
    def flat(cls, width: int, depth: int, base_y: float = 64.0, seed: int = 0, max_changed_volume: float | None = None) -> Session:
        return cls(OpStack(BaseSpec(width, depth, base_y=base_y, seed=seed)), max_changed_volume=max_changed_volume)

    @property
    def state(self) -> WorldState:
        return self.stack.state

    @property
    def ops(self) -> list[OpEntry]:
        return list(self.stack.entries)

    def apply(self, op: str, params: dict[str, Any], id: str | None = None) -> OpEntry:
        """Add one op. Raises OpError (agent-facing) and leaves the session unchanged on failure."""
        return self._guarded(lambda: self.stack.add(op, params, id=id))

    def update(self, op_id: str, params: dict[str, Any]) -> OpEntry:
        return self._guarded(lambda: self.stack.update(op_id, params))

    def remove(self, op_id: str) -> OpEntry:
        return self.stack.remove(op_id)  # undoing is never limited

    def _guarded(self, edit: Callable[[], OpEntry]) -> OpEntry:
        """Run one edit; undo it and raise OpError if it moves more terrain than the guardrail allows."""
        before = self.stack.checkpoint()
        entry = edit()
        moved = float(np.abs(self.stack.state.height - before.state.height).sum())
        if moved > self.max_changed_volume:
            self.stack.restore(before)
            columns = self.state.width * self.state.depth
            raise OpError(
                f"{entry.id} would move {moved:,.0f} blocks of terrain ({moved / columns:.1f} per column on average); "
                f"the limit per call is {self.max_changed_volume:,.0f} ({self.max_changed_volume / columns:.0f} per column). "
                "Narrow the region, or lower height, amount or strength."
            )
        return entry

    def snapshot(self) -> Snapshot:
        return self.stack.checkpoint()

    def restore(self, snap: Snapshot) -> None:
        self.stack.restore(snap)

    def summary(self) -> str:
        """Compact text view of the op stack with each op's footprint, for observations and context."""
        if not len(self.stack):
            return "(no ops yet)"
        return "\n".join(f"{e.id}: {e.op}({e.params}) -> {self.effect(e)}" for e in self.stack.entries)

    def effect(self, e: OpEntry, detail: bool = False) -> str:
        """What an op changed: its footprint, or for materials-stage ops which surface blocks it sets.

        `detail` adds the feature's stats (highest point, slopes, water) for tool results; the compact
        form goes into summaries, which every episode and critique repeat.
        """
        if get_op(e.op).stage == "materials":
            return f"surface blocks only, no height or water change ({self._material_effect(e)})"
        f = self.state.features.get(e.id)
        if f is None:
            return "no height or water change"
        text = (f"footprint u={f.bounds[0]:.2f}..{f.bounds[2]:.2f}, v={f.bounds[1]:.2f}..{f.bounds[3]:.2f}, "
                f"y={f.meta['bottom_y']:.0f}..{f.meta['top_y']:.0f}" + (", water" if f.meta["water"] else ""))
        return f"{text}; {self._feature_stats(e.id)}" if detail else text

    def _feature_stats(self, op_id: str) -> str:
        s = self.state
        sel = s.masks[f"feature:{op_id}"] > 0.02
        h = np.where(sel, s.height, -np.inf)
        z, x = np.unravel_index(int(h.argmax()), h.shape)
        gz, gx = np.gradient(s.height)
        slope = np.degrees(np.arctan(np.hypot(gx, gz)))[sel]
        parts = [f"top y={s.height[z, x]:.0f} at (u={x / (s.width - 1):.2f}, v={z / (s.depth - 1):.2f})",
                 f"slope max {slope.max():.0f}°, mean {slope.mean():.0f}°", f"{int(sel.sum()):,} columns"]
        wet = sel & ~np.isnan(s.water)
        if wet.any():
            parts.append(f"water on {int(wet.sum()):,} columns, surface y={np.nanmin(s.water[wet]):.0f}..{np.nanmax(s.water[wet]):.0f}")
        return "; ".join(parts)

    def _material_effect(self, e: OpEntry) -> str:
        if e.op == "paint_region":
            inside, _ = region_shape(self.state, validate_params(e.op, e.params).region)
            return f"{int(inside.sum())} columns set to {e.params['block']}; block colours show only in L2 renders"
        if e.op == "set_palette_rules":
            n = len(e.params["rules"])
            return f"{n} palette rule{'' if n == 1 else 's'}; block colours show only in L2 renders"
        return "block colours show only in L2 renders"

    def stats(self) -> str:
        """Terrain statistics as text: the model never reads raw arrays (DESIGN §10)."""
        s = self.state
        h = s.height
        gz, gx = np.gradient(h)
        slope = np.degrees(np.arctan(np.hypot(gx, gz)))
        peak_z, peak_x = np.unravel_index(int(h.argmax()), h.shape)
        u, v = peak_x / (s.width - 1), peak_z / (s.depth - 1)
        wet = ~np.isnan(s.water)
        water = f"{wet.mean() * 100:.0f}% of columns under water" if wet.any() else "no water"
        return (
            f"area {s.width}x{s.depth} blocks; height y={h.min():.0f}..{h.max():.0f} (mean {h.mean():.0f}); "
            f"highest point at (u={u:.2f}, v={v:.2f}); slope mean {slope.mean():.0f}°, max {slope.max():.0f}°; {water}"
        )
