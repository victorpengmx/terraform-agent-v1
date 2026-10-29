"""Tool layer between the harness and the world (DESIGN §5).

`Tool` has the same shape as the loop's `ToolDefinition` (name, description, input_schema, handler),
but handlers may return content blocks so observe tools can send images back. `op_tools(session)`
turns every registered op into a tool automatically; `execute()` runs one tool_use block and turns
`OpError` into an `is_error` tool_result. Any other exception is a bug and propagates to the harness.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agent.session import Session
from terrain.ops import OpError, list_ops

Content = str | list[dict[str, Any]]  # tool_result content: plain text, or text/image blocks


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Content]  # raises OpError for agent-fixable problems

    def to_param(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def image_media_type(data: bytes) -> str:
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    raise ValueError("unsupported image format; use PNG, JPEG, WebP or GIF")


def image_block(data: bytes) -> dict[str, Any]:
    source = {"type": "base64", "media_type": image_media_type(data), "data": base64.b64encode(data).decode()}
    return {"type": "image", "source": source}


def tool_result(tool_use_id: str, content: Content, is_error: bool = False) -> dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content, "is_error": is_error}


ID_PROPERTY = {
    "type": "string",
    "description": "Optional name for this edit, e.g. 'mountain_1' (lowercase, digits, _). "
    "Use it later with update_op/remove_op. Defaults to '<op>_<n>'.",
}


FEATURE_PROPERTY = {
    "type": "string",
    "description": "Optional: the spec feature this op builds or shapes, e.g. 'river_1'. The checklist measures each "
    "spec feature over the ops tagged with it.",
}


def op_tools(session: Session) -> list[Tool]:
    """One tool per registered op, in the registry's deterministic order (keeps the prompt cache stable).

    Each schema gains an optional `id`, which names the op in the op stack, and an optional `feature`, the spec
    feature it builds (DESIGN §4.3).
    """

    def make_handler(op_name: str) -> Callable[[dict[str, Any]], Content]:
        def handler(params: dict[str, Any]) -> Content:
            params = dict(params)
            entry = session.apply(op_name, params, id=params.pop("id", None), feature=params.pop("feature", None))
            return f"applied {entry.id}: {session.effect(entry, detail=True)}. Terrain now: {session.stats()}"

        return handler

    tools = []
    for s in list_ops():
        schema = s.params.model_json_schema()
        schema["properties"] = {"id": ID_PROPERTY, "feature": FEATURE_PROPERTY, **schema.get("properties", {})}
        tools.append(Tool(name=s.name, description=s.description, input_schema=schema, handler=make_handler(s.name)))
    return tools


def control_tools(session: Session) -> list[Tool]:
    """Observe/control tools over the op stack (DESIGN §5.6): list_ops, update_op, remove_op."""

    def update(p: dict[str, Any]) -> Content:
        params = p.get("params") or {}
        if isinstance(params, str):  # models sometimes send the object JSON-encoded (first Haiku run, 2026-10-22)
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                pass
        if not isinstance(params, dict):
            raise OpError(f'update_op params must be an object of parameters to change, e.g. {{"height": 120}}; got {params!r}')
        entry = session.update(str(p.get("id", "")), params)
        return f"updated {entry.id}; replayed the stack. {entry.id} now: {session.effect(entry, detail=True)}. Terrain now: {session.stats()}"

    def remove(p: dict[str, Any]) -> Content:
        entry = session.remove(str(p.get("id", "")))
        return f"removed {entry.id}; replayed the stack. Terrain now: {session.stats()}"

    id_only = {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"], "additionalProperties": False}
    return [
        Tool(
            name="list_ops",
            description="List the ops in the stack in replay order, with ids, parameters and each op's footprint.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            handler=lambda _: session.summary(),
        ),
        Tool(
            name="update_op",
            description="Change parameters of an existing op by id (only the ones you pass; others keep their values), "
            "then replay the stack. Use this to revise a feature instead of stacking corrections on top.",
            input_schema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Op id, e.g. 'mountain_1'"},
                    "params": {"type": "object", "description": "Parameters to change, e.g. {\"height\": 120}"},
                },
                "required": ["id", "params"],
                "additionalProperties": False,
            },
            handler=update,
        ),
        Tool(
            name="remove_op",
            description="Remove an op by id and replay the stack, undoing its effect.",
            input_schema=id_only,
            handler=remove,
        ),
    ]


VIEWS = {"map": None, "3d_sw": "overview_sw", "3d_ne": "overview_ne", "3d_se": "overview_se", "3d_nw": "overview_nw",
         "3d_s": "overview_s"}
VIEW_PX = 512  # ~350 image tokens: cheap enough to look after every big edit


def observe_tools(session: Session, out_dir: Path, max_views: int = 12) -> list[Tool]:
    """`render_view`: the builder looks at its own work mid-round (saved under `out_dir` as view_NNN.png).

    Refuses a repeat of the same view of unchanged terrain and caps views per run, so it can't be spammed.
    """
    seen: set[tuple[str, str | None, str]] = set()
    count = 0

    def render_view(p: dict[str, Any]) -> Content:
        nonlocal count
        view, feature = str(p.get("view", "map")), p.get("feature")
        if view not in VIEWS:
            raise OpError(f"unknown view {view!r}; use one of {', '.join(VIEWS)}")
        if feature is not None and feature not in session.state.features:
            known = ", ".join(sorted(session.state.features)) or "none"
            raise OpError(f"unknown feature {feature!r} (features: {known})")
        key = (view, feature, hashlib.sha1(session.stack.to_json().encode()).hexdigest())
        if key in seen:
            raise OpError(f"terrain unchanged since your last {view} view; edit something first")
        if count >= max_views:
            raise OpError(f"view limit reached ({max_views} per run); rely on the tool results' stats")
        if not session.ops:
            raise OpError("nothing built yet; apply an op first")

        if view == "map" and feature is None:
            from terrain.render.l1 import render_l1

            png = render_l1(session.state, title=f"{len(session.ops)} ops", size_px=VIEW_PX)
        else:
            from terrain.compiler import compile_state
            from terrain.render.l2 import render_volume

            f = session.state.features[feature] if feature else None
            png = render_volume(compile_state(session.state), VIEWS[view] or "overview_sw", feature=f, size_px=VIEW_PX)
        seen.add(key)
        count += 1
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"view_{count:03d}.png"
        path.write_bytes(png)
        what = f"3D close-up of {feature}" if feature else f"{view} view"
        left = max_views - count
        note = f"; only {left} view{'' if left == 1 else 's'} left this run" if left <= 3 else ""
        return [text_block(f"{what} of the terrain after {len(session.ops)} ops ({path.as_posix()}{note})"), image_block(png)]

    schema = {
        "type": "object",
        "properties": {
            "view": {"type": "string", "enum": list(VIEWS), "default": "map",
                     "description": "map: top-down relief with contours. 3d_*: block preview from that corner; "
                     "3d_s: from due south, looking north (the viewpoint of a reference photo)."},
            "feature": {"type": "string", "description": "Optional op id: 3D close-up of its footprint."},
        },
        "additionalProperties": False,
    }
    return [Tool(
        name="render_view",
        description="See the terrain now, mid-build. Views are cheap (about as much as a short message): look after "
        "each major landform and before you finish. map: top-down relief with contours, best for overall shape, "
        "heights and layout. 3d_*: block preview from a corner (3d_s: from due south, as a reference photo is "
        "seen), for silhouette, cliffs and block colours.",
        input_schema=schema,
        handler=render_view,
    )]


def execute(tools: dict[str, Tool], tool_use: Any) -> dict[str, Any]:
    """Run one tool_use block and return its tool_result block."""
    tool = tools.get(tool_use.name)
    if tool is None:
        return tool_result(tool_use.id, f"unknown tool {tool_use.name!r}. Available: {', '.join(sorted(tools))}", is_error=True)
    params = tool_use.input if isinstance(tool_use.input, dict) else json.loads(tool_use.input)
    try:
        return tool_result(tool_use.id, tool.handler(params))
    except OpError as e:
        return tool_result(tool_use.id, str(e), is_error=True)
