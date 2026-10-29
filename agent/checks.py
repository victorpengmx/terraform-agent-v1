"""Programmatic checks (DESIGN §11.2): cheap, objective checks on the canvas, run before every critique.

Blocking failures skip the critic model call; major ones force "revise" and feed the acceptance gate's rule 1.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel
from scipy.ndimage import grey_opening

from agent.session import Session
from terrain.compiler.compile import surface_y, water_top
from terrain.materials import surface_metrics
from terrain.state import WorldState, to_uv

# Programmatic check thresholds (DESIGN §11.2). Demos / baseline run vs. broken terrains, 2026-10-22:
MIN_CHANGED_SHARE = 0.01  # demos change 57-98% of columns; a 3-block bump changes 0%
SPIKE_BLOCKS = 3.0  # a column this much higher than all 4 neighbours reads as a 1-block spike in game
SHEER_DROP, MAX_SHEER_SHARE = 3.0, 0.25  # 3 blocks down to a neighbour ≈ 72°. Demos 0.2-10.4% (mesa cliffs); a wreck 81%
GENTLE_DEG = 15.0  # "gentle" slope class (terrain/features.py)
MAX_WATER_LOST = 0.25  # demos lose 0.7-8.6% (shallow lake fringes, a known hard case)
THIN_BLOCKS, MAX_THIN_SHARE = 4.0, 0.002  # demos 0.02-0.05% of columns; a 1.5-block wall 0.47%, the wreck 34%


class CheckResult(BaseModel):
    """Output of one programmatic check."""

    name: str
    passed: bool
    severity: Literal["info", "major", "blocking"]  # blocking = skip the model call, must revise
    detail: str = ""


def run_checks(session: Session) -> list[CheckResult]:
    """Cheap, objective checks on the canvas (DESIGN §11.2); no model call. Thresholds are calibrated so the
    three demos pass and constructed broken terrains fail (EVIDENCE.md, 2026-10-22)."""
    s = session.state
    h = s.height.astype(np.float64)
    r = np.round(h)  # what stands in the world
    checks = [CheckResult(name="built_something", passed=bool(session.ops), severity="blocking",
                          detail=f"{len(session.ops)} ops" if session.ops else "no ops were applied")]

    changed = (np.abs(h - session.stack.base.base_y) >= 1.0) | ~np.isnan(s.water)
    checks.append(CheckResult(
        name="meaningful_change", passed=bool(changed.mean() >= MIN_CHANGED_SHARE), severity="blocking",
        detail=f"{changed.mean():.1%} of columns differ from the flat base by at least a block (need {MIN_CHANGED_SHARE:.0%})"))

    spikes = np.argwhere(h - _neighbours(h).max(axis=0) > SPIKE_BLOCKS)
    checks.append(CheckResult(
        name="no_spikes", passed=len(spikes) == 0, severity="major",
        detail=f"{len(spikes)} columns stand >{SPIKE_BLOCKS:.0f} blocks above all 4 neighbours" + _example(s, spikes)))

    # Per block, not a central gradient: that compares a column's two neighbours and misses 1-column walls.
    sheer = r - _neighbours(r).min(axis=0) >= SHEER_DROP
    checks.append(CheckResult(
        name="steep_share", passed=bool(sheer.mean() <= MAX_SHEER_SHARE), severity="major",
        detail=f"{sheer.mean():.1%} of columns drop {SHEER_DROP:.0f}+ blocks to a neighbour (steeper than ~70°; "
        f"limit {MAX_SHEER_SHARE:.0%})" + _example(s, np.argwhere(sheer))))

    gentle = surface_metrics(s)[0] < GENTLE_DEG
    nb = _neighbours(r)
    speck = gentle & (np.all(nb == r - 1, axis=0) | np.all(nb == r + 1, axis=0))
    # TODO: gentle_steps is info-only and always passes, so it doesn't belong in the checks list; moving it out
    # changes the critic's input text ("# Programmatic checks"), so it waits for a deliberate prompt change.
    checks.append(CheckResult(
        name="gentle_steps", passed=True, severity="info",
        detail=f"{speck.sum() / max(int(gentle.sum()), 1):.2%} of gentle columns are isolated 1-block bumps or pits"
        + _example(s, np.argwhere(speck))))

    wet = ~np.isnan(s.water) & (s.water > s.height)
    lost = wet & ~(water_top(s) > surface_y(s))
    lost_share = lost.sum() / max(int(wet.sum()), 1)
    checks.append(CheckResult(
        name="water_containment", passed=bool(lost_share <= MAX_WATER_LOST), severity="major",
        detail=f"{int(lost.sum())} of {int(wet.sum())} water columns ({lost_share:.0%}) are too shallow to survive "
        f"rounding to blocks (limit {MAX_WATER_LOST:.0%})" + _example(s, np.argwhere(lost))))

    thin = r - grey_opening(r, size=(3, 3)) >= THIN_BLOCKS
    checks.append(CheckResult(
        name="thin_features", passed=bool(thin.mean() <= MAX_THIN_SHARE), severity="major",
        detail=f"{int(thin.sum())} columns ({thin.mean():.2%}) belong to pillars or walls 1-2 blocks wide and at least "
        f"{THIN_BLOCKS:.0f} high (limit {MAX_THIN_SHARE:.1%})" + _example(s, np.argwhere(thin))))
    return checks


def _neighbours(a: np.ndarray) -> np.ndarray:
    p = np.pad(a, 1, mode="edge")
    return np.stack([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])


def _example(state: WorldState, zx: np.ndarray) -> str:
    if not len(zx):
        return ""
    z, x = zx[len(zx) // 2]
    u, v = to_uv(state, x, z)
    return f"; e.g. at u={u:.2f}, v={v:.2f}"
