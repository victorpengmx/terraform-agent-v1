"""Feature graph (DESIGN §9.3): the spec the builder works to and the checklist checks against.

A SPEC step turns the prompt (and/or reference images) into a small graph of named landforms with
qualitative positions, categories and relations. Code projects positions to (u, v) bands; the builder sees
both. Viewpoint convention for text: the viewer stands at the south edge looking north, so left = west,
right = east, back = north (v = 0), front = south (v = 1). Image positions are image-relative and projected
with a fixed camera at the south edge (the frame widens with distance).
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal


from agent.models import ModelAdapter, ModelReplyError, ModelResponse, structured_call
from agent.tools import image_block, text_block
from terrain.schema import Strict

FeatureType = Literal["peak", "mountain_range", "hills", "ridge", "valley", "canyon", "river", "lake", "sea", "plateau",
                      "mesa", "basin", "island", "other"]
RelationKind = Literal["left_of", "right_of", "behind", "in_front_of", "taller_than", "lower_than", "flows_from", "reaches"]
EDGES = ("north_edge", "south_edge", "west_edge", "east_edge")
ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
MAX_FEATURES = 15


class Position(Strict):
    """Text form (x, y) or image form (image_x, depth); the other pair is null."""

    x: Literal["left", "center", "right", "any"] | None
    y: Literal["back", "middle", "front", "any"] | None
    image_x: Literal["left", "center-left", "center", "center-right", "right"] | None
    depth: Literal["near", "middle", "far"] | None


class SpecFeature(Strict):
    id: str
    type: FeatureType
    position: Position
    span_x: Literal["narrow", "wide", "full"] | None  # how far it reaches west-east (image: across the frame)
    span_y: Literal["narrow", "wide", "full"] | None  # how far it reaches north-south (image: from far to near)
    slope: Literal["gentle", "moderate", "steep", "cliff"] | None
    winding: Literal["almost straight", "winding", "twisty", "meandering"] | None
    tallest: bool
    part_of: str | None
    notes: str


class Relation(Strict):
    subject: str
    relation: RelationKind
    object: str  # a feature id, or north_edge / south_edge / west_edge / east_edge


class FeatureGraph(Strict):
    features: list[SpecFeature]
    relations: list[Relation]
    materials: str
    unsupported: list[str]

    def problems(self) -> list[str]:
        """What makes the graph unusable (empty = fine). Kept out of the JSON schema: structured outputs reject
        numeric and length constraints."""
        out = []
        ids = [f.id for f in self.features]
        if not 1 <= len(ids) <= MAX_FEATURES:
            out.append(f"{len(ids)} features; use 1 to {MAX_FEATURES}")
        out += [f"invalid id {i!r}: lowercase letters, digits and _" for i in ids if not ID_PATTERN.match(i)]
        out += [f"duplicate id {i!r}" for i in sorted({i for i in ids if ids.count(i) > 1})]
        known = set(ids)
        out += [f"{f.id}: part_of unknown feature {f.part_of!r}" for f in self.features if f.part_of and f.part_of not in known]
        if sum(f.tallest for f in self.features) > 1:
            out.append("more than one feature is marked tallest")
        for r in self.relations:
            if r.subject not in known:
                out.append(f"relation subject {r.subject!r} is not a feature")
            if r.object not in known and not (r.relation == "reaches" and r.object in EDGES):
                out.append(f"relation object {r.object!r} is not a feature" + (" or an edge" if r.relation == "reaches" else ""))
        return out

    def ids(self) -> set[str]:
        return {f.id for f in self.features}


# ---------------------------------------------------------------- projection to (u, v) bands

TEXT_U = {"left": (0.0, 0.45), "center": (0.25, 0.75), "right": (0.55, 1.0)}
TEXT_V = {"back": (0.0, 0.45), "middle": (0.25, 0.75), "front": (0.55, 1.0)}
# Images: a fixed camera at the south edge, looking north. Depth gives the v band; the visible ground widens
# with distance, so image_x maps to a u position spread by the frame width at that depth.
IMAGE_V = {"far": (0.0, 0.4), "middle": (0.3, 0.7), "near": (0.6, 1.0)}
IMAGE_X = {"left": 0.1, "center-left": 0.3, "center": 0.5, "center-right": 0.7, "right": 0.9}
FRAME_WIDTH = {"near": 0.5, "middle": 0.75, "far": 1.0}  # share of the area's width visible at that depth
BAND_HALF_WIDTH = {"near": 0.12, "middle": 0.18, "far": 0.25}


def bands(p: Position) -> tuple[float, float, float, float]:
    """(u0, u1, v0, v1) the feature's centre should fall in. Unspecified parts span the whole area."""
    if p.image_x is not None or p.depth is not None:
        depth = p.depth or "middle"
        v0, v1 = IMAGE_V[depth] if p.depth else (0.0, 1.0)
        if p.image_x is None:
            return 0.0, 1.0, v0, v1
        c = 0.5 + (IMAGE_X[p.image_x] - 0.5) * FRAME_WIDTH[depth]
        h = BAND_HALF_WIDTH[depth]
        return round(max(c - h, 0.0), 3), round(min(c + h, 1.0), 3), v0, v1
    u0, u1 = TEXT_U.get(p.x or "any", (0.0, 1.0))
    v0, v1 = TEXT_V.get(p.y or "any", (0.0, 1.0))
    return u0, u1, v0, v1


SPAN_CLASSES = (("narrow", 1 / 3), ("wide", 2 / 3), ("full", float("inf")))


def span_classes(f: SpecFeature, axis: str) -> tuple[tuple[str, float], ...]:
    """Category bounds for a footprint's extent as a share of the area. An image's span_x is relative to the
    frame, which at the feature's depth covers FRAME_WIDTH of the area, so its bounds shrink by that factor."""
    scale = FRAME_WIDTH.get(f.position.depth or "", 1.0) if axis == "x" and f.position.image_x is not None else 1.0
    return tuple((name, bound * scale) for name, bound in SPAN_CLASSES)


def graph_text(graph: FeatureGraph) -> str:
    """The pinned form the builder and critic see: each feature with its (u, v) band, then the relations."""
    lines = ["Features (u = west->east, v = north->south; the band is where the feature's centre should be):"]
    for f in graph.features:
        u0, u1, v0, v1 = bands(f.position)
        attrs = [f"spans {f.span_x} west-east" if f.span_x else "", f"spans {f.span_y} north-south" if f.span_y else "",
                 f"slope {f.slope}" if f.slope else "", f"winding {f.winding}" if f.winding else "",
                 "tallest" if f.tallest else "", f"part of {f.part_of}" if f.part_of else ""]
        attrs = ", ".join(a for a in attrs if a)
        where = "anywhere" if (u0, u1, v0, v1) == (0.0, 1.0, 0.0, 1.0) else f"u {u0:.2f}-{u1:.2f}, v {v0:.2f}-{v1:.2f}"
        lines.append(f"- {f.id} ({f.type}): {where}" + (f"; {attrs}" if attrs else "") + (f". {f.notes}" if f.notes else ""))
    if graph.relations:
        lines.append("Relations: " + "; ".join(f"{r.subject} {r.relation} {r.object}" for r in graph.relations))
    if graph.materials:
        lines.append(f"Materials: {graph.materials}")
    if graph.unsupported:
        lines.append("Not buildable with the tools (ignore): " + ", ".join(graph.unsupported))
    return "\n".join(lines)


# ---------------------------------------------------------------- SPEC step

SPEC_SYSTEM = """\
You turn a request for a Minecraft landscape into a feature graph that a builder and a checker will use. \
You do not build anything.

The area is seen from its south edge, looking north: x left = west, right = east; y back = north (far), \
front = south (near).

- features: the 3 to 15 landforms the request names or clearly implies (peaks, ranges, hills, ridges, \
valleys, canyons, rivers, lakes, sea, plateaus, mesas, basins, islands). One feature per distinct landform; \
for a range with named peaks use the range plus peaks with part_of. ids: lowercase letters, digits and _, \
e.g. river_1.
- position: for a text request set x and y and leave image_x and depth null. Use any unless the request \
says or clearly implies where the feature's centre is: "a river flowing to the south edge" says where it \
ends, not where its centre is, so its y is any. For a reference image set image_x and depth as seen in the \
image and leave x and y null.
- slope: only if the request implies it: gentle (<15 deg), moderate (15-35), steep (35-60), cliff (>60); \
else null. For plateaus, mesas, canyons, valleys and basins it means their walls. winding, for rivers, \
valleys and canyons: almost straight, winding, twisty, meandering; else null.
- span_x, span_y: how far the feature reaches west-east and north-south: narrow (under a third of the area), \
wide (a third to two thirds), full (more than two thirds). Only when the request or image shows it ("a range \
across the whole horizon", "ridges running from the back to the front"); else null. For an image, judge \
span_x against the frame width at the feature's depth and span_y from far to near.
- tallest: true for at most one feature, only if the request implies which is tallest.
- relations: only those the request states or that follow directly from its wording ("a river flowing \
from the mountains" is flows_from; a river and a lake in the same request is not a relation): left_of, right_of, behind, in_front_of, taller_than, \
lower_than between features; flows_from (a river from a feature); reaches (a river to a feature or to \
north_edge, south_edge, west_edge, east_edge).
- materials: the surface look in one sentence. unsupported: things asked for that terrain tools cannot \
make (buildings, trees, mist, animals, weather).
- notes: a few words of character or mood per feature, or empty.
Do not invent detail the request does not ask for: unspecified means null or any."""


class SpecError(ModelReplyError):
    """The SPEC reply was unusable (refusal, truncation, invalid JSON or an invalid graph)."""


def write_spec(model: ModelAdapter, prompt: str, images: list[bytes] | None = None) -> tuple[FeatureGraph, ModelResponse]:
    """One SPEC call in a fresh context. Raises SpecError (with the response, so its cost is charged)."""
    content: list[dict[str, Any]] = [image_block(i) for i in images or []]
    request = f"# Request\n{prompt}" if prompt else "# Request\n(no text)"
    if images:
        request += "\n\nReference image(s) are attached above: describe the landforms in them, with image positions."
    content.append(text_block(request))
    graph, response = structured_call(model, content, SPEC_SYSTEM, FeatureGraph, SpecError)
    if problems := graph.problems():
        raise SpecError("invalid graph: " + "; ".join(problems), response)
    return graph, response


def spec_text(spec: dict[str, Any]) -> str:
    """The spec as the builder, critic and comparison see it: the request, then the graph if there is one."""
    text = f"Request: {spec.get('prompt') or '(no text; see the reference images)'}"
    if spec.get("graph") is not None:
        text += "\n\n" + graph_text(spec["graph"])
    return text


def graph_json(graph: FeatureGraph) -> str:
    return json.dumps(graph.model_dump(mode="json"), sort_keys=True)
