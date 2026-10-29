"""Tag-independent feature matching for evaluation (DESIGN §16).

The in-loop checklist measures a spec feature over the ops the builder tagged with it. That can't grade a
build: an untagged but correct feature fails, a global op tagged with a feature id passes, a hand-written
graph's ids never match the builder's, and SPEC-off runs have no tags. Here code finds the landforms
itself:

1. Candidates: every form op with a footprint (`terrain.features.feature_type`). Modifier and materials
   ops are never candidates; tags are ignored.
2. A candidate fits a spec feature if its kind is compatible with the feature's type (`COMPATIBLE`), it
   is not a whole-area edit (unless the feature is a sea or spans the full area both ways), its centroid
   lies in the feature's band widened by `POSITION_SLACK`, and its extent is within one category of the
   spec's span_x / span_y.
3. Each spec feature gets at most one candidate and each candidate goes to at most one feature: an optimal
   assignment over cost = type preference + distance to the strict band + span category distance.
4. Unions: a matched feature of a chain type (ridge, river, ...) absorbs unassigned compatible candidates
   that touch it, a group type (range, hills) those whose own centre is in its widened band, while the
   union still fits. An unmatched
   feature of either type tries the unions of the unassigned candidates (touching components, or all in
   the band) instead.
5. A part_of parent is measured over its own ops plus its children's.

`match_checklist` runs the usual checklist on these matches: same items, same thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment

from agent.checklist import Item, checklist
from agent.spec import FeatureGraph, SpecFeature, bands, span_classes
from terrain.features import CORE, FOOTPRINT, category_distance, classify, feature_type
from terrain.opstack import OpStack

# Spec type -> groups of compatible candidate kinds, most preferred first (rank = group index).
COMPATIBLE: dict[str, tuple[tuple[str, ...], ...]] = {
    "peak": (("peak",), ("ridge", "raised_area")),
    "mountain_range": (("ridge", "peak", "raised_area"),),
    "hills": (("peak", "raised_area", "ridge"),),
    "ridge": (("ridge",), ("raised_area", "peak")),
    "plateau": (("plateau",), ("raised_area",)),
    "mesa": (("plateau",), ("raised_area",)),
    "valley": (("valley",), ("depression",)),
    "canyon": (("valley",), ("depression",)),
    "river": (("river",),),
    "lake": (("lake",),),
    "basin": (("basin", "depression"), ("lake",)),
    "sea": (("sea",),),
    "island": (("raised_area", "peak", "plateau", "ridge"),),
    "other": (("peak", "ridge", "raised_area", "plateau", "basin", "depression", "valley", "river", "lake", "sea"),),
}
# chain: one landform built in touching pieces; group: a collection that need not touch.
UNION = {"ridge": "chain", "plateau": "chain", "mesa": "chain", "valley": "chain", "canyon": "chain", "river": "chain",
         "mountain_range": "group", "hills": "group"}
GLOBAL_SHARE = 0.6  # a candidate covering this share of the columns is a whole-area edit
POSITION_SLACK = 0.15  # normalized: how far outside its band a feature's centre may be and still match
TOUCH_BLOCKS = 2.0  # chain pieces closer than this touch
RANK_COST, SPAN_COST = 0.5, 0.25
INFEASIBLE = 1e6


@dataclass(frozen=True)
class Shape:
    """Where a candidate (or union of candidates) is: weighted centroid, body extent, columns covered."""

    centroid: tuple[float, float]
    extent: tuple[float, float]
    cover: float
    footprint: np.ndarray  # bool, weight > FOOTPRINT


@dataclass(frozen=True)
class Candidate:
    id: str
    kind: str
    mask: np.ndarray
    shape: Shape


@dataclass(frozen=True)
class Match:
    ops: dict[str, list[str]]  # spec feature id -> op ids, in stack order (empty = unmatched)
    reasons: dict[str, str]  # unmatched feature id -> why


def candidates(stack: OpStack) -> list[Candidate]:
    state = stack.state
    out = []
    for e in stack.entries:
        kind = feature_type(e)
        mask = state.masks.get(f"feature:{e.id}")
        if kind is None or mask is None or not (mask > FOOTPRINT).any():
            continue
        out.append(Candidate(e.id, kind, mask, _shape(mask)))
    return out


def _shape(mask: np.ndarray) -> Shape:
    d, w = mask.shape
    fp = mask > FOOTPRINT
    body = mask > CORE
    if not body.any():
        body = fp
    zz, xx = np.mgrid[0:d, 0:w]
    weights = np.where(fp, mask, 0.0)
    centroid = (float((xx * weights).sum() / weights.sum()) / (w - 1), float((zz * weights).sum() / weights.sum()) / (d - 1))
    zs, xs = np.nonzero(body)
    extent = (float(xs.max() - xs.min()) / (w - 1), float(zs.max() - zs.min()) / (d - 1))
    return Shape(centroid, extent, float(fp.mean()), fp)


def _union(cands: list[Candidate]) -> Shape:
    return _shape(np.maximum.reduce([c.mask for c in cands]))


def _rank(f: SpecFeature, c: Candidate) -> int | None:
    """Type preference (0 = best), or None if the candidate can't be this feature."""
    if c.shape.cover >= GLOBAL_SHARE and not (f.type == "sea" or (f.span_x == "full" and f.span_y == "full")):
        return None
    for rank, kinds in enumerate(COMPATIBLE[f.type]):
        if c.kind in kinds:
            return rank
    return None


def _fit(f: SpecFeature, s: Shape) -> float | None:
    """Position and size cost of a shape for a feature, or None if it fails a gate."""
    u0, u1, v0, v1 = bands(f.position)
    cu, cv = s.centroid
    du, dv = max(u0 - cu, 0.0, cu - u1), max(v0 - cv, 0.0, cv - v1)
    if du > POSITION_SLACK or dv > POSITION_SLACK:
        return None
    cost = float(np.hypot(du, dv))
    for axis, want, share in (("x", f.span_x, s.extent[0]), ("y", f.span_y, s.extent[1])):
        if want:
            classes = span_classes(f, axis)
            dist = category_distance(want, classify(share, classes), classes)
            if dist > 1:
                return None
            cost += SPAN_COST * dist
    return cost


def _near(f: SpecFeature, c: Candidate) -> bool:
    """A group member's own centre lies in the feature's widened band (not just the union's)."""
    u0, u1, v0, v1 = bands(f.position)
    cu, cv = c.shape.centroid
    return u0 - POSITION_SLACK <= cu <= u1 + POSITION_SLACK and v0 - POSITION_SLACK <= cv <= v1 + POSITION_SLACK


def _touches(a: np.ndarray, b: np.ndarray) -> bool:
    return float(distance_transform_edt(~a)[b].min()) <= TOUCH_BLOCKS


def match(graph: FeatureGraph, stack: OpStack) -> Match:
    feats, cands = graph.features, candidates(stack)
    ranks = [[_rank(f, c) for c in cands] for f in feats]
    cost = np.full((len(feats), max(len(cands), 1)), INFEASIBLE)
    for i, f in enumerate(feats):
        for j, c in enumerate(cands):
            if ranks[i][j] is not None and (fit := _fit(f, c.shape)) is not None:
                cost[i, j] = RANK_COST * ranks[i][j] + fit
    chosen: dict[str, list[Candidate]] = {f.id: [] for f in feats}
    taken: set[str] = set()
    if cands:
        for i, j in zip(*linear_sum_assignment(cost)):
            if cost[i, j] < INFEASIBLE:
                chosen[feats[i].id] = [cands[j]]
                taken.add(cands[j].id)

    # Unions, from candidates nobody took: first unmatched features try groups (existence comes first), then
    # matched ones absorb leftovers, each leftover going to the feature whose band is closest to it.
    pools = {f.id: [c for j, c in enumerate(cands) if ranks[i][j] is not None] for i, f in enumerate(feats)}
    for f in feats:
        if f.type in UNION and not chosen[f.id]:
            chosen[f.id] = _best_group(f, [c for c in pools[f.id] if c.id not in taken], UNION[f.type])
            taken |= {c.id for c in chosen[f.id]}
    while (best := _best_absorb([f for f in feats if f.type in UNION and chosen[f.id]], chosen, pools, taken)) is not None:
        fid, c = best
        chosen[fid].append(c)
        taken.add(c.id)

    order = {e.id: k for k, e in enumerate(stack.entries)}
    ops = {f.id: [c.id for c in chosen[f.id]] for f in feats}
    for f in feats:  # a parent contains its parts
        if f.part_of in ops:
            ops[f.part_of] = sorted(set(ops[f.part_of]) | set(ops[f.id]), key=order.__getitem__)
    ops = {k: sorted(v, key=order.__getitem__) for k, v in ops.items()}
    reasons = {f.id: _reason(f, cands, ranks[i], taken) for i, f in enumerate(feats) if not ops[f.id]}
    return Match(ops, reasons)


def _best_absorb(feats: list[SpecFeature], chosen: dict[str, list[Candidate]], pools: dict[str, list[Candidate]],
                 taken: set[str]) -> tuple[str, Candidate] | None:
    """The (feature, leftover candidate) pair to join next: a chain piece must touch the union, a group member
    lie in the widened band, and the union must still fit; among those, the candidate nearest the feature's
    strict band wins (ties: spec order, then stack order)."""
    best: tuple[float, str, Candidate] | None = None
    for f in feats:
        for c in pools[f.id]:
            if c.id in taken:
                continue
            if UNION[f.type] == "chain" and not _touches(_union(chosen[f.id]).footprint, c.shape.footprint):
                continue
            if UNION[f.type] == "group" and not _near(f, c):
                continue
            if _fit(f, _union([*chosen[f.id], c])) is None:
                continue
            u0, u1, v0, v1 = bands(f.position)
            cu, cv = c.shape.centroid
            d = float(np.hypot(max(u0 - cu, 0.0, cu - u1), max(v0 - cv, 0.0, cv - v1)))
            if best is None or d < best[0]:
                best = (d, f.id, c)
    return None if best is None else (best[1], best[2])


def _best_group(f: SpecFeature, pool: list[Candidate], kind: str) -> list[Candidate]:
    if kind == "chain":
        groups, left = [], list(pool)
        while left:
            group = [left.pop(0)]
            grew = True
            while grew:
                grew = False
                for c in left:
                    if _touches(_union(group).footprint, c.shape.footprint):
                        group.append(c)
                        left.remove(c)
                        grew = True
                        break
            groups.append(group)
    else:
        near = [c for c in pool if _near(f, c)]
        groups = [near] if near else []
    scored = [(fit, g) for g in groups if (fit := _fit(f, _union(g))) is not None]
    return min(scored, key=lambda t: t[0])[1] if scored else []


def _reason(f: SpecFeature, cands: list[Candidate], ranks: list[int | None], taken: set[str]) -> str:
    kinds = sorted({k for group in COMPATIBLE[f.type] for k in group})
    typed = [c for c in cands if c.kind in kinds]
    if not typed:
        return f"no op builds a {' or '.join(kinds)}"
    ok = [c for c, r in zip(cands, ranks) if r is not None]
    if not ok:
        return f"the only {' or '.join(kinds)} ops ({', '.join(c.id for c in typed)}) cover most of the area"
    u0, u1, v0, v1 = bands(f.position)
    placed = [c for c in ok if _fit(f.model_copy(update={"span_x": None, "span_y": None}), c.shape) is not None]
    if not placed:
        return (f"no compatible landform has its centre near u {u0:.2f}-{u1:.2f}, v {v0:.2f}-{v1:.2f} "
                f"(candidates: {', '.join(f'{c.id} at u={c.shape.centroid[0]:.2f}, v={c.shape.centroid[1]:.2f}' for c in ok)})")
    if not [c for c in placed if _fit(f, c.shape) is not None]:
        return f"no compatible landform there has the spec's extent ({', '.join(c.id for c in placed)} checked)"
    return f"every compatible landform there was matched to another spec feature ({', '.join(c.id for c in placed if c.id in taken)})"


def match_checklist(graph: FeatureGraph, stack: OpStack) -> list[Item]:
    """The spec checklist in matcher mode: features measured over the matched ops, tags ignored."""
    m = match(graph, stack)
    return checklist(graph, stack, ops=m.ops, unmatched=m.reasons)
