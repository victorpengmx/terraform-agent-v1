import numpy as np
import pytest

from terrain.materials import SURFACE_BLOCKS, Banded, Condition, Mix, Rule, Single, resolve_surface, rule_coverage, rule_text
from terrain.ops import OpError, apply_op
from terrain.opstack import BaseSpec, OpStack
from terrain.state import WorldState

N = 65


def flat(y=64.0):
    return WorldState.flat(N, N, base_y=y)


def with_rules(state, *rules):
    state.palette_rules = [r.model_dump(mode="json") for r in rules]
    return state


def test_defaults_give_grassland_on_flat_ground():
    plan = resolve_surface(flat())
    assert set(plan.column_block.ravel()) <= {"grass_block", "podzol", "coarse_dirt"}
    assert (plan.column_block == "grass_block").mean() > 0.6
    assert set(plan.subsurface.ravel()) == {"dirt"}


def test_first_matching_rule_wins():
    s = with_rules(flat(), Rule(when=Condition(y_min=60), use=Single(block="sand")), Rule(use=Single(block="stone")))
    assert set(resolve_surface(s).column_block.ravel()) == {"sand"}
    s = with_rules(flat(), Rule(when=Condition(y_min=100), use=Single(block="sand")), Rule(use=Single(block="clay")))
    assert set(resolve_surface(s).column_block.ravel()) == {"clay"}


def test_unmatched_columns_fall_back_to_stone():
    s = with_rules(flat(), Rule(when=Condition(y_max=10), use=Single(block="sand")))
    assert set(resolve_surface(s).column_block.ravel()) == {"stone"}


def test_slope_and_water_conditions():
    s = apply_op(flat(), "raise_peak", {"center": [0.5, 0.5], "radius": 0.3, "height": 60, "steepness": 0.888, "roughness": 0})
    s.water[:, :3] = 65.0  # a strip of water along the west edge
    s = with_rules(
        s,
        Rule(when=Condition(water_dist_max=2), use=Single(block="gravel")),
        Rule(when=Condition(slope_min=50), use=Single(block="tuff")),
        Rule(use=Single(block="grass_block")),
    )
    blocks = resolve_surface(s).column_block
    assert np.all(blocks[:, :5] == "gravel") and not np.any(blocks[:, 5] == "gravel")  # 3 wet + 2 within distance
    assert (blocks == "tuff").any() and blocks[32, 32] == "grass_block"  # summit is flat-ish, flanks steep


def test_mix_proportions_and_patches():
    s = with_rules(flat(), Rule(use=Mix(blocks={"grass_block": 3, "podzol": 1}, patch_size=0.05)))
    blocks = resolve_surface(s).column_block
    share = (blocks == "podzol").mean()
    assert 0.2 < share < 0.3
    # Patches, not salt-and-pepper: most neighbours share a block.
    same = (blocks[:, 1:] == blocks[:, :-1]).mean()
    assert same > 0.85


def test_banded_strata_vary_with_y_and_are_deterministic():
    s = with_rules(flat(), Rule(use=Banded(blocks=["stone", "tuff", "calcite"], band_min=2, band_max=3, noise=0)))
    plan = resolve_surface(s)
    column = [plan.surface_block(np.full((N, N), float(y)))[10, 10] for y in range(40, 80)]
    assert len(set(column)) > 1
    runs = [len(list(g)) for _, g in __import__("itertools").groupby(column)]
    assert max(runs[1:-1] or [2]) <= 6  # bands are thin (consecutive equal picks may merge)
    plan2 = resolve_surface(s)
    assert np.array_equal(plan.surface_block(np.full((N, N), 50.0)), plan2.surface_block(np.full((N, N), 50.0)))


def test_feature_condition_uses_op_footprint():
    stack = OpStack(BaseSpec(N, N, 64))
    stack.add("raise_peak", {"center": [0.3, 0.3], "radius": 0.2, "height": 30}, id="hill")
    stack.add("set_palette_rules", {"rules": [
        {"when": {"feature": "hill"}, "use": {"kind": "single", "block": "moss_block"}},
        {"use": {"kind": "single", "block": "sand"}},
    ]})
    blocks = resolve_surface(stack.state).column_block
    assert blocks[19, 19] == "moss_block" and blocks[60, 60] == "sand"


def test_set_palette_rules_rejects_unknown_feature_and_block():
    stack = OpStack(BaseSpec(N, N, 64))
    with pytest.raises(OpError, match="unknown feature 'nope'"):
        stack.add("set_palette_rules", {"rules": [{"when": {"feature": "nope"}, "use": {"kind": "single", "block": "sand"}}]})
    with pytest.raises(OpError, match="invalid params"):
        stack.add("set_palette_rules", {"rules": [{"use": {"kind": "single", "block": "diamond_block"}}]})


def test_paint_region_overrides_rules():
    stack = OpStack(BaseSpec(N, N, 64))
    stack.add("paint_region", {"region": {"polygon": [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]}, "block": "gravel"})
    plan = resolve_surface(stack.state)
    assert plan.column_block[5, 5] == "gravel" and plan.column_block[60, 60] != "gravel"
    assert plan.subsurface[5, 5] == "gravel"


def test_default_blocks_are_all_in_the_allowed_list():
    from terrain.materials import DEFAULT_RULES

    used = set()
    for r in DEFAULT_RULES:
        u = r.use
        used |= {u.block} if isinstance(u, Single) else set(u.blocks)
    assert used <= set(SURFACE_BLOCKS)


def test_block_looks_cover_every_block_and_match_the_colour_table():
    import json
    from pathlib import Path

    from terrain.materials import BLOCK_LOOKS, SURFACE_BLOCKS
    from terrain.ops import get_op

    colours = json.loads((Path(__file__).parents[1] / "terrain" / "render" / "block_colors.json").read_text())["colors"]
    for block in SURFACE_BLOCKS:
        r, g, b = colours[block]["top"]
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        look = BLOCK_LOOKS[block]
        if "white" in look and "pinkish" not in look:
            assert lum > 200, (block, look, lum)
        if "pale" in look:
            assert lum > 150, (block, look, lum)
        if "dark" in look or "black" in look:
            assert lum < 115, (block, look, lum)
        if "blue" in look and "grey" not in look:
            assert b > max(r, g), (block, look)
    description = get_op("set_palette_rules").params.model_json_schema()["properties"]["rules"]["description"]
    assert all(f"{block}: {BLOCK_LOOKS[block]}" in description for block in SURFACE_BLOCKS)


def test_banded_strata_tilt_with_noise_and_stay_level_without():
    import numpy as np

    from terrain.materials import MAX_BAND_DIP, _dip
    from terrain.state import WorldState

    s = WorldState.flat(65, 65, base_y=64)
    assert np.all(_dip(s, 0.0, seed=3) == 0)  # noise 0: perfectly level (vanilla badlands)
    plane = _dip(s, 0.5, seed=3)
    gz, gx = np.gradient(plane)
    assert np.allclose(np.hypot(gx, gz), MAX_BAND_DIP * 0.5)  # a plane with the promised dip
    assert not np.allclose(_dip(s, 0.5, seed=4), plane)  # direction varies with the seed


# ---------------------------------------------------------------- soft slope limits (slope_fuzz)


def ramp(deg: float, n: int = 129) -> WorldState:
    """A plane rising east at a constant slope."""
    s = WorldState.flat(n, n, base_y=64, seed=3)
    s.height += (np.arange(n, dtype=np.float32) * np.tan(np.radians(deg)))[None, :]
    return s


ROCKY = Rule(when=Condition(slope_min=35, slope_fuzz=12), use=Single(block="tuff"))
GRASS = Rule(use=Single(block="grass_block"))


def test_slope_fuzz_splits_one_slope_into_large_patches():
    blocks = resolve_surface(with_rules(ramp(30), ROCKY, GRASS)).column_block[8:-8, 8:-8]
    rock = blocks == "tuff"
    # 30° sits 5° below the limit; with ±12° of fuzz ~29% of the ground passes, in coherent patches.
    assert 0.15 < rock.mean() < 0.45
    from scipy.ndimage import label
    assert label(rock)[1] < 40  # a few large patches, not speckle (113 x 113 columns)


def test_without_fuzz_a_slope_below_the_limit_is_all_one_rule():
    hard = Rule(when=Condition(slope_min=35), use=Single(block="tuff"))
    assert np.all(resolve_surface(with_rules(ramp(30), hard, GRASS)).column_block[8:-8, 8:-8] == "grass_block")


def test_slope_fuzz_is_deterministic_and_seeded():
    a = resolve_surface(with_rules(ramp(30), ROCKY, GRASS)).column_block
    b = resolve_surface(with_rules(ramp(30), ROCKY, GRASS)).column_block
    other = ramp(30)
    other.seed = 4
    c = resolve_surface(with_rules(other, ROCKY, GRASS)).column_block
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_default_palette_turns_to_rock_gradually_with_slope():
    rock_blocks = ["stone", "andesite", "tuff", "gravel"]
    shares = []
    for deg in (20, 30, 40, 50, 60, 70, 80):
        top = resolve_surface(ramp(deg)).column_block[8:-8, 8:-8]
        shares.append(np.isin(top.astype(str), rock_blocks).mean())
    assert shares == sorted(shares)  # more rock on steeper ground
    assert all(0.05 < x < 0.95 for x in shares[1:5])  # 30-60°: a patchwork, never all grass or all rock
    assert shares[-1] > 0.9


def test_default_cliffs_are_patches_not_bands():
    from terrain.materials import DEFAULT_RULES

    assert not any(isinstance(r.use, Banded) for r in DEFAULT_RULES)


def _ridge_with_palette(rules):
    stack = OpStack(BaseSpec(N, N, 64))
    stack.add("raise_peak", {"center": [0.5, 0.5], "radius": 0.3, "height": 40, "steepness": 0.9}, id="hill")
    stack.add("set_palette_rules", {"rules": rules})
    return stack.state


STEEP_STONE_RULE = {"when": {"slope_min": 40}, "use": {"kind": "single", "block": "stone"}}
HILL_MOSS_RULE = {"when": {"feature": "hill"}, "use": {"kind": "single", "block": "moss_block"}}
GRASS_RULE = {"use": {"kind": "single", "block": "grass_block"}}


def test_rule_coverage_names_the_earlier_rule_that_shadows_a_feature_rule():
    cover = rule_coverage(_ridge_with_palette([STEEP_STONE_RULE, HILL_MOSS_RULE, GRASS_RULE]))
    assert len(cover) == 4  # three rules plus the stone fallback
    moss = cover[1]
    assert 0 < moss.won < moss.matched and set(moss.taken_by) == {0}
    assert moss.won + moss.taken_by[0] == moss.matched
    first = rule_coverage(_ridge_with_palette([HILL_MOSS_RULE, STEEP_STONE_RULE, GRASS_RULE]))[0]
    assert first.won == first.matched and first.taken_by == {}


def test_rule_coverage_counts_paint_and_rules_that_match_nothing():
    state = _ridge_with_palette([{"when": {"y_min": 300}, "use": {"kind": "single", "block": "snow_block"}}, GRASS_RULE])
    assert rule_coverage(state)[0].matched == 0
    stack = OpStack(BaseSpec(N, N, 64))
    stack.add("set_palette_rules", {"rules": [GRASS_RULE]})
    stack.add("paint_region", {"region": {"polygon": [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]}, "block": "gravel"})
    grass = rule_coverage(stack.state)[0]
    assert grass.taken_by["paint"] > 0 and grass.won + grass.taken_by["paint"] == N * N


def test_resolve_surface_unchanged_by_the_coverage_refactor():
    state = _ridge_with_palette([STEEP_STONE_RULE, HILL_MOSS_RULE, GRASS_RULE])
    owner = resolve_surface(state).rule_index
    cover = rule_coverage(state)
    assert [int((owner == i).sum()) for i in range(4)] == [c.won for c in cover]


def test_rule_text_is_short_and_complete():
    assert rule_text(Rule.model_validate(STEEP_STONE_RULE)) == "slope >= 40 -> stone"
    assert rule_text(Rule(when=Condition(feature="hill", y_max=90.5), use=Mix(blocks={"grass_block": 1, "podzol": 1})))         == "y <= 90.5, feature hill -> grass_block/podzol"
    assert rule_text(Rule.model_validate(GRASS_RULE)) == "everywhere -> grass_block"
