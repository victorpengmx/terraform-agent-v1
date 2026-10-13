"""Materials-stage ops: choose which blocks the surface is made of (DESIGN §5.7)."""

from __future__ import annotations

import numpy as np
from pydantic import Field

from terrain.brushes import known_features
from terrain.materials import BLOCK_LOOKS_TEXT, BlockName, Rule
from terrain.ops.region import REGION_HELP, Region, region_shape
from terrain.ops.registry import OpError, OpParams, register_op
from terrain.state import WorldState


class SetPaletteRulesParams(OpParams):
    rules: list[Rule] = Field(
        min_length=1, max_length=16,
        description="Evaluated per surface column in order; the first rule whose conditions all hold wins. "
        "Put specific rules (cliffs, shores, a feature) before general ones and end with a rule without "
        "conditions. Columns no rule matches get stone. How the blocks look (approximate colour): "
        + BLOCK_LOOKS_TEXT + ".",
    )


@register_op("set_palette_rules", params=SetPaletteRulesParams, stage="materials")
def set_palette_rules(state: WorldState, p: SetPaletteRulesParams) -> WorldState:
    """Replace the surface palette: which blocks cover slopes, cliffs, shores, peaks or named features.

    Use banded for layered rock walls, mix for natural patchy ground, single for uniform surfaces.
    Replaces the default palette entirely; only the latest set_palette_rules applies.
    """
    known = set(known_features(state))
    for i, rule in enumerate(p.rules):
        if rule.when.feature is not None and rule.when.feature not in known:
            raise OpError(f"rule {i} refers to unknown feature {rule.when.feature!r}; known: {', '.join(sorted(known)) or 'none'}")
    state.palette_rules = [r.model_dump(mode="json") for r in p.rules]
    return state


class PaintRegionParams(OpParams):
    region: Region = Field(description=f"Area to paint: {REGION_HELP}.")
    block: BlockName = Field(description="Surface block for every column inside; overrides the palette rules.")


@register_op("paint_region", params=PaintRegionParams, stage="materials")
def paint_region(state: WorldState, p: PaintRegionParams) -> WorldState:
    """Force the surface block inside a region (paths, beaches, scree fields), overriding the palette rules."""
    inside, _ = region_shape(state, p.region)
    if p.block not in state.material_names:
        state.material_names.append(p.block)
    state.material = np.where(inside, state.material_names.index(p.block), state.material).astype(np.int16)
    return state
