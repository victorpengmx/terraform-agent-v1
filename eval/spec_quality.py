"""SPEC quality (E5, DESIGN §16): how well the agent's own feature graph matches the answer key.

Features are matched one-to-one (`linear_sum_assignment`): types must be the same or compatible (valley ~
canyon, plateau ~ mesa, ridge or peak ~ mountain_range, hills ~ peak, other ~ anything) and their position
bands (`agent.spec.bands`) must overlap. Cost = 0.5 x type mismatch + (1 - band IoU).

Reported: feature recall (answer-key features found) and precision (agent features that match one); category
accuracy on matched features (slope, winding, span_x, span_y, only where the answer key states one; exact
and within one category; a missing value is wrong); relation recall (an answer-key relation is found when
the agent states it between the matched features, or its inverse: left_of/right_of, behind/in_front_of,
taller_than/lower_than; edges match edges). Runs without an agent graph (SPEC off, or SPEC failed) are skipped.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from agent.spec import EDGES, FeatureGraph, SpecFeature, bands

COMPATIBLE = {frozenset(p) for p in [("valley", "canyon"), ("plateau", "mesa"), ("ridge", "mountain_range"),
                                     ("peak", "mountain_range"), ("hills", "peak")]}
CATEGORIES = {
    "slope": ("gentle", "moderate", "steep", "cliff"),
    "winding": ("almost straight", "winding", "twisty", "meandering"),
    "span_x": ("narrow", "wide", "full"),
    "span_y": ("narrow", "wide", "full"),
}
INVERSE = {"left_of": "right_of", "right_of": "left_of", "behind": "in_front_of", "in_front_of": "behind",
           "taller_than": "lower_than", "lower_than": "taller_than"}
MISMATCH_COST = 0.5
INFEASIBLE = 1e6


def type_cost(a: str, b: str) -> float | None:
    if a == b:
        return 0.0
    if frozenset((a, b)) in COMPATIBLE or "other" in (a, b):
        return MISMATCH_COST
    return None


def band_iou(a: SpecFeature, b: SpecFeature) -> float:
    au0, au1, av0, av1 = bands(a.position)
    bu0, bu1, bv0, bv1 = bands(b.position)
    iw, ih = max(0.0, min(au1, bu1) - max(au0, bu0)), max(0.0, min(av1, bv1) - max(av0, bv0))
    inter = iw * ih
    union = (au1 - au0) * (av1 - av0) + (bu1 - bu0) * (bv1 - bv0) - inter
    return inter / union if union > 0 else 0.0


def match_graphs(key: FeatureGraph, agent: FeatureGraph) -> dict[str, str]:
    """answer-key feature id -> agent feature id."""
    if not key.features or not agent.features:
        return {}
    cost = np.full((len(key.features), len(agent.features)), INFEASIBLE)
    for i, k in enumerate(key.features):
        for j, a in enumerate(agent.features):
            t, iou = type_cost(k.type, a.type), band_iou(k, a)
            if t is not None and iou > 0:
                cost[i, j] = t + (1 - iou)
    return {key.features[i].id: agent.features[j].id for i, j in zip(*linear_sum_assignment(cost)) if cost[i, j] < INFEASIBLE}


def compare(key: FeatureGraph, agent: FeatureGraph) -> dict[str, Any]:
    """One run's SPEC quality counts (sums, so runs aggregate by adding)."""
    matches = match_graphs(key, agent)
    by_id = {f.id: f for f in agent.features}
    cats = {k: {"n": 0, "exact": 0, "within_one": 0} for k in CATEGORIES}
    for f in key.features:
        if f.id not in matches:
            continue
        g = by_id[matches[f.id]]
        for attr, order in CATEGORIES.items():
            want, got = getattr(f, attr), getattr(g, attr)
            if want is None:
                continue
            cats[attr]["n"] += 1
            if got is not None:
                d = abs(order.index(want) - order.index(got))
                cats[attr]["exact"] += d == 0
                cats[attr]["within_one"] += d <= 1
    stated = {(r.subject, r.relation, r.object) for r in agent.relations}
    stated |= {(o, INVERSE[rel], s) for s, rel, o in stated if rel in INVERSE}
    found = 0
    for r in key.relations:
        s = matches.get(r.subject)
        o = r.object if r.object in EDGES else matches.get(r.object)
        found += s is not None and o is not None and (s, r.relation, o) in stated
    return {"key_features": len(key.features), "agent_features": len(agent.features), "matched": len(matches),
            "categories": cats, "relations": len(key.relations), "relations_found": found, "matches": matches}


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Micro-averaged rates over runs: recall, precision, category accuracy, relation recall."""
    def rate(a: int, b: int) -> float | None:
        return a / b if b else None

    tot = {k: sum(r[k] for r in results) for k in ("key_features", "agent_features", "matched", "relations", "relations_found")}
    cats = {attr: {k: sum(r["categories"][attr][k] for r in results) for k in ("n", "exact", "within_one")} for attr in CATEGORIES}
    return {
        "runs": len(results),
        "recall": rate(tot["matched"], tot["key_features"]),
        "precision": rate(tot["matched"], tot["agent_features"]),
        "relation_recall": rate(tot["relations_found"], tot["relations"]),
        "categories": {a: {"n": c["n"], "exact": rate(c["exact"], c["n"]), "within_one": rate(c["within_one"], c["n"])}
                       for a, c in cats.items()},
        **tot,
    }


def for_rows(rows: list[dict[str, Any]], keys: dict[str, FeatureGraph]) -> dict[str, dict[str, Any]]:
    """E5 summaries for all / text / image runs that have an agent graph."""
    per = [(r["kind"], compare(keys[r["prompt_id"]], FeatureGraph.model_validate(r["spec_graph"])))
           for r in rows if r.get("status") == "done" and r.get("spec_graph")]
    return {"all": summarize([c for _, c in per]), "text": summarize([c for k, c in per if k == "text"]),
            "image": summarize([c for k, c in per if k == "image"])}
