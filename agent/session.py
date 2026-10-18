"""Session: the agent's working world (DESIGN §3, §14 row 1).

A thin agent-facing wrapper around the `OpStack` (the source of truth). Tools close over a session, so
no tool touches globals. Snapshots are op-stack checkpoints, used by the acceptance gate's rollback.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from terrain.config import MAX_MEAN_DH
from terrain.materials import active_rules, rule_coverage, rule_text
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
        self.spec_feature_ids: set[str] | None = None  # set by the harness when the run has a feature graph
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

    def apply(self, op: str, params: dict[str, Any], id: str | None = None, feature: str | None = None) -> OpEntry:
        """Add one op. Raises OpError (agent-facing) and leaves the session unchanged on failure."""
        self._check_feature(feature)
        return self._guarded(lambda: self.stack.add(op, params, id=id, feature=feature))

    def update(self, op_id: str, params: dict[str, Any]) -> OpEntry:
        """`params` may include `feature` to retag the op (null clears the tag)."""
        params = dict(params)
        feature = params.pop("feature", ...)
        if feature is not ...:
            self._check_feature(feature)
        return self._guarded(lambda: self.stack.update(op_id, params, feature=feature))

    def _check_feature(self, feature: str | None) -> None:
        if feature is not None and self.spec_feature_ids is not None and feature not in self.spec_feature_ids:
            raise OpError(f"unknown spec feature {feature!r}; the spec's features are: {', '.join(sorted(self.spec_feature_ids))}")

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
        return "\n".join(f"{e.id}{f' [feature={e.feature}]' if e.feature else ''}: {e.op}({e.params}) -> {self.effect(e)}"
                         for e in self.stack.entries)

    def effect(self, e: OpEntry, detail: bool = False) -> str:
        """What an op changed: its footprint, or for materials-stage ops which surface blocks it sets.

        `detail` adds the feature's stats (highest point, slopes, water) for tool results; the compact
        form goes into summaries, which every episode and critique repeat.
        """
        if get_op(e.op).stage == "materials":
            return f"surface blocks only, no height or water change ({self._material_effect(e, detail)})"
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

    def _material_effect(self, e: OpEntry, detail: bool = False) -> str:
        if e.op == "paint_region":
            inside, _ = region_shape(self.state, validate_params(e.op, e.params).region)
            return f"{int(inside.sum())} columns set to {e.params['block']}; block colours show only in L2 renders"
        if e.op == "set_palette_rules":
            n = len(e.params["rules"])
            text = f"{n} palette rule{'' if n == 1 else 's'}"
            latest = [x for x in self.stack.entries if x.op == "set_palette_rules"][-1]
            if latest.id != e.id:
                return f"{text}; replaced by {latest.id} (only the latest set_palette_rules applies)"
            return "; ".join([text, *self._palette_report(detail), "block colours show only in L2 renders"])
        return "block colours show only in L2 renders"

    def _palette_report(self, detail: bool) -> list[str]:
        """Warnings for rules that decide little of the ground they match (rule order, paint), plus in `detail`
        (tool results) every rule's share of the area. Rules are numbered from 1, as the builder lists them."""
        rules, cover = active_rules(self.state), rule_coverage(self.state)
        total = self.state.width * self.state.depth
        out = []
        if detail:
            shares = [f"rule {i + 1} ({rule_text(r)}) {c.won / total:.0%}" for i, (r, c) in enumerate(zip(rules[:-1], cover))]
            if cover[-1].won:
                shares.append(f"no rule matches {cover[-1].won / total:.0%} (stone)")
            if painted := int((self.state.material > 0).sum()):
                shares.append(f"paint_region {painted / total:.0%}")
            out.append("area shares: " + ", ".join(shares))
        for i, (r, c) in enumerate(zip(rules[:-1], cover)):
            name = f"rule {i + 1} ({rule_text(r)})"
            if c.matched == 0:
                out.append(f"WARNING: {name} matches no column")
            elif c.won == 0 or (r.when.feature is not None and c.won < c.matched / 2):
                k, n = max(c.taken_by.items(), key=lambda kv: kv[1])
                lead = f"WARNING: {name} decides {c.won / c.matched:.0%} of the ground it matches; "
                if k == "paint":
                    out.append(lead + f"paint_region covers {n / c.matched:.0%} of it")
                else:
                    out.append(lead + f"earlier rule {k + 1} ({rule_text(rules[k])}) takes {n / c.matched:.0%}: move rule "
                               f"{i + 1} above rule {k + 1} (with its own slope limits) to cover that ground")
        return out

    def stats(self) -> str:
        return terrain_stats(self.state)


def terrain_stats(s: WorldState) -> str:
    """Terrain statistics as text: the model never reads raw arrays (DESIGN §10). Mentions no op ids, so the
    blind comparison can use it too."""
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


def op_diff(before: Sequence[OpEntry], after: Sequence[OpEntry]) -> list[str]:
    """What changed between two op lists, one line per op: added, removed, or changed params (old -> new)."""
    old, new = {e.id: e for e in before}, {e.id: e for e in after}
    lines = [f"added {e.id}: {e.op}({e.params})" for e in after if e.id not in old]
    lines += [f"removed {e.id} ({e.op})" for e in before if e.id not in new]
    for e in after:
        prev = old.get(e.id)
        if prev is not None and prev.params != e.params:
            keys = sorted(prev.params.keys() | e.params.keys())
            changes = [f"{k} {prev.params.get(k)!r} -> {e.params.get(k)!r}" for k in keys if prev.params.get(k) != e.params.get(k)]
            lines.append(f"changed {e.id}: " + ", ".join(changes))
    return lines
