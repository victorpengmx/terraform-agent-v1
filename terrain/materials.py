"""Palette rule engine (DESIGN §5.7): which block each surface column gets.

Rules are structured data, evaluated per column, first match wins:

    Rule(when=Condition(slope_min=50), use=Banded(blocks=["stone", "andesite", "tuff"]))

`resolve_surface(state)` returns a `SurfacePlan` that the compiler queries for the block at any
(column, y): mixes are chosen per column (spatially coherent patches), bands per y (strata on cliffs).
The LLM picks blocks and conditions; this code applies them consistently.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from scipy.ndimage import distance_transform_edt, gaussian_filter

from terrain.noise import fractal_noise
from terrain.state import WorldState

# Natural blocks the palette may use. A closed list keeps tool schemas explicit (enum) and guarantees
# the colour table and backend translation cover every block the agent can pick.
SURFACE_BLOCKS: tuple[str, ...] = (
    "grass_block", "podzol", "mycelium", "moss_block", "dirt", "coarse_dirt", "rooted_dirt", "mud", "packed_mud",
    "clay", "sand", "red_sand", "gravel", "sandstone", "red_sandstone", "stone", "cobblestone", "mossy_cobblestone",
    "andesite", "diorite", "granite", "tuff", "calcite", "deepslate", "dripstone_block", "basalt", "smooth_basalt",
    "blackstone", "obsidian", "magma_block", "terracotta", "white_terracotta", "orange_terracotta",
    "yellow_terracotta", "brown_terracotta", "red_terracotta", "light_gray_terracotta", "snow_block", "packed_ice",
    "blue_ice", "ice",
)  # fmt: skip
BlockName = Literal[SURFACE_BLOCKS]  # type: ignore[valid-type]

# How each block looks, for the agent: it picks block names from memory and only sees colours in L2
# renders. Checked against the averaged texture colours in terrain/render/block_colors.json.
BLOCK_LOOKS: dict[str, str] = {
    "grass_block": "green top, dirt sides", "podzol": "dark brown", "mycelium": "grey-purple",
    "moss_block": "olive green", "dirt": "brown", "coarse_dirt": "darker brown", "rooted_dirt": "light brown",
    "mud": "near-black grey", "packed_mud": "tan", "clay": "pale blue-grey", "sand": "pale beige",
    "red_sand": "bright orange", "gravel": "grey, speckled", "sandstone": "pale beige", "red_sandstone": "bright orange",
    "stone": "mid grey", "cobblestone": "mid grey, rough", "mossy_cobblestone": "grey-green", "andesite": "mid grey",
    "diorite": "light grey", "granite": "dull pinkish brown", "tuff": "dark grey-green", "calcite": "near white",
    "deepslate": "dark grey", "dripstone_block": "greyish brown", "basalt": "dark grey", "smooth_basalt": "dark grey",
    "blackstone": "near black", "obsidian": "black-purple", "magma_block": "dark orange-red, glowing",
    "terracotta": "muted orange-brown", "white_terracotta": "pale pinkish beige", "orange_terracotta": "rust orange",
    "yellow_terracotta": "ochre", "brown_terracotta": "dark chocolate brown", "red_terracotta": "brick red",
    "light_gray_terracotta": "greyish mauve-brown", "snow_block": "white", "packed_ice": "pale blue",
    "blue_ice": "blue", "ice": "pale blue, translucent",
}  # fmt: skip
if set(BLOCK_LOOKS) != set(SURFACE_BLOCKS):
    raise RuntimeError(f"BLOCK_LOOKS must cover SURFACE_BLOCKS exactly; differs in {set(BLOCK_LOOKS) ^ set(SURFACE_BLOCKS)}")
BLOCK_LOOKS_TEXT = "; ".join(f"{b}: {BLOCK_LOOKS[b]}" for b in SURFACE_BLOCKS)
SLOPE_SMOOTHING = 1.5  # gaussian sigma (blocks) applied before measuring slope for palette rules
FUZZ_PATCH = 0.06  # size of the patches over which slope_fuzz shifts a rule's slope limits (fraction of the shorter side)
MAX_BAND_DIP = 0.12  # band tilt (blocks of rise per block) at noise=1, ~7°; strata dip, so cones don't show rings

# Soil under organic surfaces; everything else continues as itself (then stone below).
SOIL_UNDER = {"grass_block": "dirt", "podzol": "dirt", "mycelium": "dirt", "moss_block": "dirt", "coarse_dirt": "dirt",
              "rooted_dirt": "dirt", "sand": "sandstone", "red_sand": "red_sandstone", "snow_block": "stone"}  # fmt: skip


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Condition(_Strict):
    """All given limits must hold. Omit a field to ignore it."""

    slope_min: float | None = Field(default=None, ge=0, le=90, description="Degrees.")
    slope_max: float | None = Field(default=None, ge=0, le=90, description="Degrees.")
    y_min: float | None = Field(default=None, description="Surface block y.")
    y_max: float | None = Field(default=None, description="Surface block y.")
    water_dist_max: float | None = Field(default=None, ge=0, description="Blocks to the nearest water column (0 = under water).")
    feature: str | None = Field(default=None, description="Only inside this feature's footprint (op id, e.g. 'mountain_1').")
    slope_fuzz: float | None = Field(
        default=None, ge=0, le=30,
        description="Degrees: the slope limits wander by up to this much in large patches, so ground near a limit is "
        "partly this rule and partly the next (a 30° slope can be grass in one place and rock in another).")


class Single(_Strict):
    kind: Literal["single"] = "single"
    block: BlockName


class Mix(_Strict):
    kind: Literal["mix"] = "mix"
    blocks: dict[BlockName, float] = Field(description="Block -> relative weight, e.g. {\"grass_block\": 0.8, \"podzol\": 0.2}.")
    patch_size: float = Field(default=0.03, gt=0, le=0.5, description="Patch size as a fraction of the shorter side.")

    @field_validator("blocks")
    @classmethod
    def positive(cls, v: dict[str, float]) -> dict[str, float]:
        if not v or any(w <= 0 for w in v.values()):
            raise ValueError("mix needs at least one block and all weights > 0")
        return v


class Banded(_Strict):
    kind: Literal["banded"] = "banded"
    blocks: list[BlockName] = Field(min_length=1, max_length=12, description="Blocks cycled as horizontal strata.")
    band_min: int = Field(default=2, ge=1, le=16, description="Thinnest band, blocks.")
    band_max: int = Field(default=5, ge=1, le=32, description="Thickest band, blocks.")
    noise: float = Field(default=0.3, ge=0, le=1, description="How much bands wobble and tilt across the area (0 = perfectly level).")

    @model_validator(mode="after")
    def ordered(self) -> Banded:
        if self.band_max < self.band_min:
            raise ValueError("band_max must be >= band_min")
        return self


Use = Annotated[Single | Mix | Banded, Field(discriminator="kind")]


class Rule(_Strict):
    when: Condition = Field(default_factory=Condition)
    use: Use
    subsurface: BlockName | None = Field(default=None, description="Block under the surface; default depends on the surface block.")
    subsurface_depth: int = Field(default=3, ge=0, le=8)


DEFAULT_RULES: tuple[Rule, ...] = (
    # Cliffs: patches of rock types, not bands (default bands drew rings on cones, EVIDENCE 2026-10-22). Soft
    # limits: rock takes over gradually between ~50° and ~70°, and moderate slopes are a patchwork of grass,
    # gravel and stone from ~23° up, so no slope is all one material at a hard threshold.
    Rule(when=Condition(slope_min=60, slope_fuzz=10),
         use=Mix(blocks={"stone": 0.55, "andesite": 0.22, "tuff": 0.13, "gravel": 0.10}, patch_size=0.04)),
    Rule(when=Condition(y_min=170), use=Mix(blocks={"snow_block": 0.75, "stone": 0.25})),
    Rule(when=Condition(water_dist_max=2), use=Mix(blocks={"sand": 0.5, "gravel": 0.3, "mud": 0.2})),
    Rule(when=Condition(slope_min=35, slope_fuzz=12),
         use=Mix(blocks={"grass_block": 0.40, "coarse_dirt": 0.12, "stone": 0.25, "andesite": 0.10, "gravel": 0.13},
                 patch_size=0.05)),
    Rule(use=Mix(blocks={"grass_block": 0.9, "podzol": 0.06, "coarse_dirt": 0.04}, patch_size=0.015)),
)
FALLBACK = Rule(use=Single(block="stone"))


def surface_metrics(state: WorldState) -> tuple[np.ndarray, np.ndarray]:
    """(slope in degrees, distance in blocks to the nearest water column; inf if there is no water).

    Slope is taken over a lightly smoothed surface: material should follow the landform, not single-block
    steps, or rock rules speckle grassland wherever detail noise left a 1-block jump."""
    gz, gx = np.gradient(gaussian_filter(state.height.astype(np.float64), SLOPE_SMOOTHING))
    slope = np.degrees(np.arctan(np.hypot(gx, gz)))
    wet = ~np.isnan(state.water)
    dist = distance_transform_edt(~wet) if wet.any() else np.full(state.height.shape, np.inf)
    return slope, dist


@dataclass
class SurfacePlan:
    """Per-column block choice. `blocks_at(y)` gives the block name per column for a horizontal slice."""

    rules: list[Rule]
    rule_index: np.ndarray  # int, per column
    column_block: np.ndarray  # object array: block name per column for single/mix rules ('' for banded)
    band_tables: dict[int, tuple[np.ndarray, int]]  # rule index -> (block per y offset, y start)
    band_offset: np.ndarray  # per-column y wobble for banded rules
    subsurface: np.ndarray  # object array: subsurface block per column
    subsurface_depth: np.ndarray  # int per column

    def surface_block(self, y: np.ndarray) -> np.ndarray:
        """Block name per column at height y (array shaped like the columns)."""
        out = self.column_block.copy()
        for r, (table, y_start) in self.band_tables.items():
            cols = self.rule_index == r
            idx = np.clip(np.round(y[cols] + self.band_offset[cols]).astype(int) - y_start, 0, len(table) - 1)
            out[cols] = table[idx]
        return out


def active_rules(state: WorldState) -> list[Rule]:
    """The palette in force (the latest set_palette_rules, else the defaults), ending with the stone fallback."""
    rules = [Rule.model_validate(r) for r in state.palette_rules] if state.palette_rules is not None else list(DEFAULT_RULES)
    return [*rules, FALLBACK]


def _rule_matches(state: WorldState, rules: list[Rule]) -> list[np.ndarray]:
    """Per rule, the columns its own conditions accept (ignoring earlier rules)."""
    slope, water_dist = surface_metrics(state)
    y = np.round(state.height)
    return [_matches(rule.when, state, slope, y, water_dist, zlib.crc32(f"{state.seed}:fuzz:{i}".encode()))
            for i, rule in enumerate(rules)]


def _first_match(matches: list[np.ndarray]) -> np.ndarray:
    rule_index = np.full(matches[0].shape, -1, dtype=int)
    for i, match in enumerate(matches):
        rule_index[match & (rule_index < 0)] = i
    return rule_index


@dataclass(frozen=True)
class RuleCoverage:
    """How much of the surface one palette rule decides, and who decides the rest of the ground it matches."""

    matched: int  # columns its conditions accept
    won: int  # columns whose block it decides (first match, not painted over)
    taken_by: dict[int | str, int]  # matched columns decided elsewhere: an earlier rule's index, or "paint"


def rule_coverage(state: WorldState) -> list[RuleCoverage]:
    """Coverage of every rule in `active_rules` (the fallback last). Rule order is the usual trap: a feature
    rule after a slope rule only gets the feature's ground the slope rule leaves (EVIDENCE 2026-10-28)."""
    matches = _rule_matches(state, active_rules(state))
    owner = _first_match(matches)
    painted = state.material > 0
    out = []
    for i, match in enumerate(matches):
        taken: dict[int | str, int] = {}
        if n := int((match & painted).sum()):
            taken["paint"] = n
        lost = owner[match & ~painted]
        for k, n in zip(*np.unique(lost[lost != i], return_counts=True)):
            taken[int(k)] = int(n)
        out.append(RuleCoverage(int(match.sum()), int((match & ~painted & (owner == i)).sum()), taken))
    return out


def rule_text(rule: Rule) -> str:
    """A rule in a few words, e.g. 'slope >= 42 -> stone', for tool results and critiques."""
    c, parts = rule.when, []
    for key, word in (("slope_min", "slope >= {}"), ("slope_max", "slope <= {}"), ("y_min", "y >= {}"),
                      ("y_max", "y <= {}"), ("water_dist_max", "within {} of water"), ("feature", "feature {}")):
        if (v := getattr(c, key)) is not None:
            parts.append(word.format(f"{v:g}" if isinstance(v, float) else v))
    use = rule.use
    blocks = use.block if isinstance(use, Single) else "/".join(use.blocks)
    return f"{', '.join(parts) or 'everywhere'} -> {blocks}"


def resolve_surface(state: WorldState) -> SurfacePlan:
    rules = active_rules(state)
    y = np.round(state.height)
    shape = state.height.shape
    rule_index = _first_match(_rule_matches(state, rules))

    column_block = np.full(shape, "", dtype=object)
    band_tables: dict[int, tuple[np.ndarray, int]] = {}
    band_offset = np.zeros(shape)
    for i, rule in enumerate(rules):
        cols = rule_index == i
        if not cols.any():
            continue
        seed = zlib.crc32(f"{state.seed}:palette:{i}".encode())
        use = rule.use
        if isinstance(use, Single):
            column_block[cols] = use.block
        elif isinstance(use, Mix):
            column_block[cols] = _mix(state, use, seed)[cols]
        else:
            rng = np.random.default_rng(seed)
            y_start, y_end = int(y.min()) - 64, int(y.max()) + 64
            table, ys = [], y_start
            while ys < y_end:
                thick = int(rng.integers(use.band_min, use.band_max + 1))
                table += [rng.choice(use.blocks)] * thick
                ys += thick
            band_tables[i] = (np.array(table, dtype=object), y_start)
            wobble = fractal_noise(state, feature_size=0.2, octaves=2, roughness=0.5, seed=seed)
            band_offset[cols] = (wobble * use.noise * use.band_max * 2 + _dip(state, use.noise, seed))[cols]

    subsurface = np.empty(shape, dtype=object)
    depth = np.zeros(shape, dtype=int)
    for i, rule in enumerate(rules):
        cols = rule_index == i
        depth[cols] = rule.subsurface_depth
        if rule.subsurface is not None:
            subsurface[cols] = rule.subsurface
    unset = subsurface == None  # noqa: E711  (elementwise on an object array)
    if unset.any():
        surf = np.where(column_block == "", "stone", column_block)
        subsurface[unset] = [SOIL_UNDER.get(b, b) for b in surf[unset]]

    # Explicit paint (material layer) beats the rules.
    painted = state.material > 0
    if painted.any():
        names = np.array(state.material_names, dtype=object)
        column_block[painted] = names[state.material[painted]]
        rule_index[painted] = -1  # no banding
        subsurface[painted] = [SOIL_UNDER.get(b, b) for b in column_block[painted]]

    return SurfacePlan(rules, rule_index, column_block, band_tables, band_offset, subsurface, depth)


def _dip(state: WorldState, noise: float, seed: int) -> np.ndarray:
    """Tilted strata: a plane rising MAX_BAND_DIP * noise blocks per block in a seeded direction.

    Bands that follow pure height draw concentric rings on any cone; a dip, as real strata have, breaks
    the symmetry. Its own random stream, so the band sequence is unchanged.
    """
    angle = np.random.default_rng(seed + 1).uniform(0.0, 2 * np.pi)
    zz, xx = np.mgrid[0 : state.depth, 0 : state.width].astype(np.float64)
    plane = (xx - state.width / 2) * np.cos(angle) + (zz - state.depth / 2) * np.sin(angle)
    return plane * MAX_BAND_DIP * noise


def _matches(c: Condition, state: WorldState, slope: np.ndarray, y: np.ndarray, water_dist: np.ndarray, fuzz_seed: int = 0) -> np.ndarray:
    m = np.ones(state.height.shape, dtype=bool)
    if c.slope_fuzz:  # shift this rule's view of the slope by up to ±slope_fuzz, in coherent patches
        slope = slope + c.slope_fuzz * (2.0 * _patch_ranks(state, FUZZ_PATCH, fuzz_seed) - 1.0)
    if c.slope_min is not None:
        m &= slope >= c.slope_min
    if c.slope_max is not None:
        m &= slope <= c.slope_max
    if c.y_min is not None:
        m &= y >= c.y_min
    if c.y_max is not None:
        m &= y <= c.y_max
    if c.water_dist_max is not None:
        m &= water_dist <= c.water_dist_max
    if c.feature is not None:
        mask = state.masks.get(f"feature:{c.feature}", state.masks.get(f"spec:{c.feature}"))
        if mask is None:
            m &= False  # unknown feature matches nothing; set_palette_rules validates names up front
        else:
            m &= mask > 0.05
    return m


def _patch_ranks(state: WorldState, patch_size: float, seed: int) -> np.ndarray:
    """Smooth seeded noise rank-normalized to uniform [0, 1): coherent patches, every value equally likely."""
    noise = fractal_noise(state, feature_size=patch_size, octaves=2, roughness=0.5, seed=seed)
    return np.argsort(np.argsort(noise, axis=None)).reshape(noise.shape) / noise.size


def _mix(state: WorldState, mix: Mix, seed: int) -> np.ndarray:
    """Coherent patches: uniform patch noise, then cut by cumulative weight."""
    ranks = _patch_ranks(state, mix.patch_size, seed)
    names = list(mix.blocks)
    edges = np.cumsum([mix.blocks[n] for n in names]) / sum(mix.blocks.values())
    return np.array(names, dtype=object)[np.minimum(np.searchsorted(edges, ranks, side="right"), len(names) - 1)]


def rules_json_schema() -> dict[str, Any]:
    return Rule.model_json_schema()
