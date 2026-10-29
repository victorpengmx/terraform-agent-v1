"""Spec checklist (DESIGN §11.3): the feature graph turned into atomic items that code checks.

Each spec feature is measured over the union of the footprints of its ops (`terrain/features.py`
`measure_spec_feature`): in tag mode (the in-loop critic) the ops tagged with it, in matcher mode (evaluation)
the ops `eval/match.py` matched to it without looking at tags. Items:
- `<id>.exists`: some tagged op changed the terrain. If it fails, the feature's other items and its relations
  are skipped (dependency skipping, as in the Davidsonian Scene Graph).
- `<id>.position`: the footprint's centre lies in the band the spec's position projects to (agent/spec.py).
- `<id>.slope`, `<id>.winding`: measured category vs. spec category, one category of slack (one off = minor).
- `<id>.tallest`: the highest summit among the spec's raised features.
- relations: left_of / right_of / behind / in_front_of (centres at least 0.05 apart), taller_than /
  lower_than (summits at least 2 blocks apart), flows_from (river source within 3 blocks of the feature),
  reaches (river mouth within 2 blocks of the edge or feature).
Failed items carry a located fix hint for the builder; the pass count is the acceptance gate's rule 2.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel
from scipy.ndimage import distance_transform_edt

from agent.spec import EDGES, FeatureGraph, bands, span_classes
from terrain.features import (FOOTPRINT, classify, RAISED_SPEC_TYPES, SLOPE_CLASSES, WALL_SPEC_TYPES, WINDING_CLASSES, FeatureAttributes,
                              category_distance, measure_spec_feature)
from terrain.opstack import OpStack
from terrain.state import to_xz

SEPARATION = 0.05  # normalized: centres closer than this are neither left nor right of each other
HEIGHT_MARGIN = 2.0  # blocks: summits closer than this are neither taller nor lower
SOURCE_REACH = 3.0  # blocks from the source feature's footprint
MOUTH_REACH = 2.0  # blocks from the edge or the target feature
WINDING_HINT = "meander 0.3 = winding, 0.6 = twisty, 1.0 = meandering"


class Item(BaseModel):
    id: str
    text: str
    status: Literal["pass", "fail", "skipped"]
    severity: Literal["minor", "major"] = "major"
    detail: str = ""


def checklist(graph: FeatureGraph, stack: OpStack, ops: dict[str, list[str]] | None = None,
              unmatched: dict[str, str] | None = None) -> list[Item]:
    """Tag mode (`ops` None, the in-loop critic): each spec feature is measured over the ops tagged with it.
    Matcher mode (evaluation, `eval/match.py`): `ops` maps each spec feature to the ops matched to it, tags
    are ignored, and `unmatched` says why a feature got no ops. Same items and thresholds in both modes."""
    matched = ops is not None
    if ops is None:
        ops = {f.id: [e.id for e in stack.entries if e.feature == f.id] for f in graph.features}
    ops = {f.id: ops.get(f.id, []) for f in graph.features}
    verb = "matched to" if matched else "tagged with"
    attrs: dict[str, FeatureAttributes | None] = {
        f.id: measure_spec_feature(stack, f.id, f.type, ops[f.id]) if ops[f.id] else None for f in graph.features}
    items: list[Item] = []
    for f in graph.features:
        a = attrs[f.id]
        if a is None:
            if ops[f.id]:
                why = f"the ops {verb} '{f.id}' ({', '.join(ops[f.id])}) changed no terrain"
            elif matched:
                why = f"no landform in the terrain matches it: {(unmatched or {}).get(f.id, 'no candidate')}"
            else:
                why = f"no op is tagged with feature '{f.id}'; pass feature='{f.id}' on the ops that build it"
            items.append(Item(id=f"{f.id}.exists", text=f"{f.id} ({f.type}) exists", status="fail", detail=why))
            items += [Item(id=f"{f.id}.{k}", text=f"{f.id} {k}", status="skipped", detail=f"{f.id} does not exist")
                      for k in _attribute_items(f)]
            continue
        items.append(Item(id=f"{f.id}.exists", text=f"{f.id} ({f.type}) exists", status="pass",
                          detail=f"ops {', '.join(ops[f.id])}"))
        items += _attributes(graph, f, a, attrs, verb)
    for r in graph.relations:
        items.append(_relation(r, attrs, ops, stack, verb))
    return items


def _attribute_items(f) -> list[str]:
    keys = []
    if bands(f.position) != (0.0, 1.0, 0.0, 1.0):
        keys.append("position")
    keys += [k for k in ("span_x", "span_y", "slope", "winding") if getattr(f, k)]
    if f.tallest:
        keys.append("tallest")
    return keys


def _attributes(graph: FeatureGraph, f, a: FeatureAttributes, attrs: dict, verb: str = "tagged with") -> list[Item]:
    out = []
    u0, u1, v0, v1 = bands(f.position)
    if "position" in _attribute_items(f):
        cu, cv = a.centroid_uv
        ok = u0 <= cu <= u1 and v0 <= cv <= v1
        out.append(Item(id=f"{f.id}.position", text=f"{f.id} lies in u {u0:.2f}-{u1:.2f}, v {v0:.2f}-{v1:.2f}",
                        status="pass" if ok else "fail",
                        detail=f"centre at u={cu:.2f}, v={cv:.2f}" + ("" if ok else "; move its ops into the band")))
    for axis, key, word in (("x", "span_x", "west-east"), ("y", "span_y", "north-south")):
        want = getattr(f, key)
        if want:
            classes = span_classes(f, axis)
            share = a.extent_uv[0 if axis == "x" else 1]
            out.append(_category(f.id, key, want, classify(share, classes), classes,
                                 f"body spans {share:.0%} of the area {word}",
                                 (f"make it reach further {word}: a raise_ridge (or a longer path or bigger region) along it",
                                  f"make it shorter {word}: a shorter path or smaller region")))
    if f.slope:
        stat = "wall slope (90th percentile)" if f.type in WALL_SPEC_TYPES else "mean slope"
        out.append(_category(f.id, "slope", f.slope, a.slope_class, SLOPE_CLASSES, f"{stat} {a.slope_mean:.0f}°",
                             ("make it steeper: higher steepness or a steeper character, more height or depth for its width",
                              "make it gentler: lower steepness or a gentler character, less height or depth for its width")))
    if f.winding:
        if a.sinuosity is None:
            out.append(Item(id=f"{f.id}.winding", text=f"{f.id} is {f.winding}", status="fail",
                            detail=f"no carve_river or carve_valley op is {verb} it"))
        else:
            hints = ((f"raise meander on its carve_river op ({WINDING_HINT})", f"lower meander on its carve_river op ({WINDING_HINT})")
                     if a.source_uv is not None else
                     ("add bends (path points offset sideways) to its carve_valley path",
                      "straighten its carve_valley path (fewer or smaller sideways offsets)"))
            out.append(_category(f.id, "winding", f.winding, a.winding, WINDING_CLASSES, f"sinuosity {a.sinuosity:.2f}", hints))
    if f.tallest:
        # Not against the feature it is part of, or its own parts: their footprints contain each other, so their
        # tops are the same summit (a tallest peak of a range could otherwise never pass).
        others = {g.id: attrs[g.id].top_y for g in graph.features
                  if g.id != f.id and g.type in RAISED_SPEC_TYPES and attrs.get(g.id) is not None
                  and g.id != f.part_of and g.part_of != f.id}
        rival = max(others.items(), key=lambda kv: kv[1], default=None)
        ok = rival is None or a.top_y > rival[1]
        out.append(Item(id=f"{f.id}.tallest", text=f"{f.id} is the tallest", status="pass" if ok else "fail",
                        detail=f"summit y={a.top_y:.0f}" + ("" if ok else f"; {rival[0]} reaches y={rival[1]:.0f}: raise {f.id} or lower {rival[0]}")))
    return out


def _category(fid: str, key: str, want: str, got: str, classes, measured: str, hints: tuple[str, str]) -> Item:
    """`hints`: (what to do when the measured category is below the spec's, ... when it is above)."""
    d = category_distance(want, got, classes)
    names = [n for n, _ in classes]
    hint = hints[0] if names.index(got) < names.index(want) else hints[1]
    return Item(id=f"{fid}.{key}", text=f"{fid} {key} is {want}", status="pass" if d == 0 else "fail",
                severity="minor" if d <= 1 else "major",
                detail=f"{measured} ({got})" + ("" if d == 0 else f", spec says {want}; {hint}"))


def _relation(r, attrs: dict, ops: dict, stack: OpStack, verb: str = "tagged with") -> Item:
    ident, text = f"{r.subject}.{r.relation}.{r.object}", f"{r.subject} {r.relation} {r.object}"
    s = attrs.get(r.subject)
    o = None if r.object in EDGES else attrs.get(r.object)
    missing = [x for x, v in ((r.subject, s), (r.object, o)) if v is None and not (x == r.object and x in EDGES)]
    if missing:
        return Item(id=ident, text=text, status="skipped", detail=f"{', '.join(missing)} does not exist")
    rel = r.relation
    if rel in ("left_of", "right_of", "behind", "in_front_of"):
        axis = 0 if rel in ("left_of", "right_of") else 1
        sign = -1 if rel in ("left_of", "behind") else 1
        diff = (s.centroid_uv[axis] - o.centroid_uv[axis]) * sign
        word = "u" if axis == 0 else "v"
        return _verdict(ident, text, diff > SEPARATION,
                        f"centres {word}={s.centroid_uv[axis]:.2f} vs {o.centroid_uv[axis]:.2f} (need {SEPARATION} apart)")
    if rel in ("taller_than", "lower_than"):
        diff = (s.top_y - o.top_y) * (1 if rel == "taller_than" else -1)
        return _verdict(ident, text, diff > HEIGHT_MARGIN, f"summits y={s.top_y:.0f} vs y={o.top_y:.0f}; change height")
    # flows_from / reaches: the subject must be a river.
    if s.source_uv is None:
        return Item(id=ident, text=text, status="fail", detail=f"{r.subject} has no carve_river op {verb} it")
    state = stack.state
    if rel == "flows_from":
        dist = _distance_to(stack, ops[r.object], s.source_uv)
        return _verdict(ident, text, dist <= SOURCE_REACH,
                        f"source {dist:.0f} blocks from {r.object}; start its path inside {r.object}")
    if r.object in EDGES:
        x, z = to_xz(state, *s.mouth_uv)
        dist = {"north_edge": z, "south_edge": state.depth - 1 - z, "west_edge": x, "east_edge": state.width - 1 - x}[r.object]
    else:
        dist = _distance_to(stack, ops[r.object], s.mouth_uv)
    return _verdict(ident, text, dist <= MOUTH_REACH, f"mouth {dist:.0f} blocks from {r.object}; end its path there")


def _verdict(ident: str, text: str, ok: bool, detail: str) -> Item:
    return Item(id=ident, text=text, status="pass" if ok else "fail", detail=detail.split("; ")[0] if ok else detail)


def _distance_to(stack: OpStack, op_ids: list[str], uv: tuple[float, float]) -> float:
    """Blocks from a point to the union footprint of some ops (0 inside)."""
    state = stack.state
    masks = [state.masks[f"feature:{i}"] for i in op_ids if f"feature:{i}" in state.masks]
    inside = np.maximum.reduce(masks) > FOOTPRINT
    x, z = (round(c) for c in to_xz(state, *uv))
    return float(distance_transform_edt(~inside)[z, x])


def summary(items: list[Item]) -> str:
    """Pass count plus every failed item with its detail: what the builder and critic read."""
    checked = [i for i in items if i.status != "skipped"]
    lines = [f"Spec checklist: {sum(i.status == 'pass' for i in items)} of {len(checked)} checked items pass"
             + (f" ({len(items) - len(checked)} skipped)" if len(checked) < len(items) else "") + "."]
    lines += [f"- [{i.severity}] {i.text}: {i.detail}" for i in items if i.status == "fail"]
    return "\n".join(lines)
