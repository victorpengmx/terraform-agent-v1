"""Tool catalogue for the critic (DESIGN §11.5) and the schema check for its structured fixes.

The catalogue is generated from the op registry, one line per tool, so the critic's fixes name real tools,
parameters and values; `hint_problems` checks each fix against the schemas and the judged op stack.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated

from pydantic import ValidationError

from terrain.ops import list_ops

if TYPE_CHECKING:
    from agent.critic import Issue

CHARACTER_FIELDS = ("character", "roughness", "steepness")


def tool_catalogue() -> str:
    """One line per tool, generated from the registry (DESIGN §11.5): name, parameters with their allowed
    values or range, and the first sentence of the description. Deterministic, so the prompt stays cacheable."""
    lines = ["Tools the builder has. A fix names one of them and only its parameters; every parameter takes one "
             "value per call (a width cannot vary along a path, for example).", ""]
    for spec in list_ops():
        schema = spec.params.model_json_schema()
        own = [k for k in spec.params.model_fields if k not in CHARACTER_FIELDS]
        shared = [k for k in spec.params.model_fields if k in CHARACTER_FIELDS]
        params = ", ".join(_param_text(k, schema["properties"][k], schema) for k in own + shared)
        lines.append(f"- {spec.name}({params}): {_first_sentence(spec.description)}")
        if spec.name == "set_palette_rules":
            lines.append(_palette_syntax())
    lines += ["- update_op(id, params): change parameters of an existing op by id; the op keeps its other values.",
              "- remove_op(id): remove an op and its effect.",
              "character presets set roughness and steepness: gentle, rolling, rugged, jagged. A region is "
              '{"circle": {"center", "radius"}}, {"polygon": [[u, v], ...]}, {"path": {"points", "width"}} or '
              '{"feature": "<op id>"}.']
    return "\n".join(lines)


def _palette_syntax() -> str:
    """The rule format, generated from the materials models, so palette fixes can be written as valid rules."""
    from terrain.materials import Condition

    conds = ", ".join(Condition.model_fields)
    return ('  rules is the WHOLE new palette (it replaces the current one), first match wins: a list of '
            '{"when": {<any of ' + conds + '>}, "use": {"kind": "single", "block": B} | {"kind": "mix", '
            '"blocks": {B: weight, ...}, "patch_size"} | {"kind": "banded", "blocks": [B, ...], "band_min", '
            '"band_max", "noise"}, "subsurface": B}; end with a rule without "when". Slopes in degrees, y in blocks.')


def _param_text(name: str, prop: dict, schema: dict) -> str:
    """name, plus allowed values (a|b) or a numeric range [lo..hi] where the schema has them."""
    options = [o for o in prop.get("anyOf", [prop]) if o.get("type") != "null"]
    o = options[0] if len(options) == 1 else {}
    if "$ref" in o:
        o = schema.get("$defs", {}).get(o["$ref"].split("/")[-1], {})
    if "enum" in o and len(o["enum"]) <= 6:
        return f"{name}={'|'.join(map(str, o['enum']))}"
    lo = o.get("minimum", o.get("exclusiveMinimum"))
    hi = o.get("maximum", o.get("exclusiveMaximum"))
    return f"{name} [{lo:g}..{hi:g}]" if lo is not None and hi is not None else name


def _first_sentence(text: str) -> str:
    first = text.strip().split("\n\n")[0].replace("\n", " ")
    end = first.find(". ")
    return first if end < 0 else first[: end + 1]


def hint_problems(issue: Issue, stack) -> list[str]:
    """Why a structured fix could not be applied as written (empty = it names a real tool, real parameters
    and values those parameters accept). `stack` resolves `op_id` for update_op and remove_op."""
    from pydantic import TypeAdapter

    from terrain.ops import get_op
    from terrain.ops.registry import OpError

    ids = {e.id: e.op for e in stack.entries}
    if issue.tool == "none":
        return []
    if issue.tool == "remove_op":
        return [] if issue.op_id in ids else [f"remove_op: no op with id {issue.op_id!r}"]
    if issue.tool == "update_op":
        if issue.op_id not in ids:
            return [f"update_op: no op with id {issue.op_id!r}"]
        op_name = ids[issue.op_id]
    else:
        op_name = issue.tool
    try:
        fields = get_op(op_name).params.model_fields
    except OpError:
        return [f"unknown tool {issue.tool!r}"]
    problems = []
    for c in issue.changes:
        if c.param == "feature" and issue.tool == "update_op":
            continue  # retagging an op with a spec feature
        if c.param not in fields:
            problems.append(f"{op_name} has no parameter {c.param!r}")
            continue
        try:
            value = json.loads(c.value)
        except json.JSONDecodeError:
            value = c.value  # a bare word such as u or thermal
        f = fields[c.param]
        try:
            TypeAdapter(Annotated[f.annotation, *f.metadata] if f.metadata else f.annotation).validate_python(value)
        except ValidationError as e:
            problems.append(f"{op_name}.{c.param}={c.value}: {e.errors(include_url=False)[0]['msg']}")
    return problems
