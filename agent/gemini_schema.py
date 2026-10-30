"""JSON-schema conversion for Gemini (DESIGN §9.4): our pydantic schemas -> the subset Gemini documents.

Function declarations (`parameters_json_schema`) and structured output (`response_json_schema`) take a JSON
Schema subset (https://ai.google.dev/gemini-api/docs/structured-output, checked 2026-10-30). Pydantic emits
keywords outside it, so `to_gemini_schema` rewrites a copy; the Anthropic schemas are never touched. Every
constraint that is loosened here is still enforced on our side (pydantic validates tool inputs and replies),
and constraints the model should know about move into the description.

| Ours | Gemini |
|---|---|
| `$defs` + `$ref` | inlined (recursion raises) |
| `oneOf` + `discriminator` | `anyOf`; the tag joins each branch's `required` (it has a default, but pydantic needs it) |
| `const: x` | `enum: [x]` |
| `exclusiveMinimum` / `exclusiveMaximum` | `minimum` / `maximum` (if absent), "(must be > m)" in the description |
| `default` | "(default: ...)" in the description |
| `propertyNames` | dropped; its enum goes into the description |
| `minLength`, `maxLength` | dropped (documented as ignored) |
"""

from __future__ import annotations

import json
from typing import Any

ALLOWED = frozenset({
    "type", "title", "description", "properties", "required", "additionalProperties", "enum", "format",
    "minimum", "maximum", "items", "prefixItems", "minItems", "maxItems", "anyOf",
})
TYPES = frozenset({"string", "number", "integer", "boolean", "object", "array", "null"})
_DROPPED = frozenset({"minLength", "maxLength", "discriminator", "$defs", "definitions"})


class SchemaConversionError(ValueError):
    pass


def to_gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """A converted copy of `schema` (never mutated) that `check_gemini_schema` accepts."""
    defs = {**schema.get("definitions", {}), **schema.get("$defs", {})}
    return _convert(schema, defs, ())


def _resolve(ref: str, defs: dict[str, Any], seen: tuple[str, ...]) -> tuple[dict[str, Any], tuple[str, ...]]:
    prefix = next((p for p in ("#/$defs/", "#/definitions/") if ref.startswith(p)), None)
    if prefix is None or ref[len(prefix):] not in defs:
        raise SchemaConversionError(f"unsupported $ref {ref!r} (only local $defs are inlined)")
    if ref in seen:
        raise SchemaConversionError(f"recursive $ref {ref!r}: Gemini needs an inlined schema")
    return defs[ref[len(prefix):]], (*seen, ref)


def _convert(node: Any, defs: dict[str, Any], seen: tuple[str, ...]) -> Any:
    if isinstance(node, list):
        return [_convert(x, defs, seen) for x in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        target, seen = _resolve(node["$ref"], defs, seen)
        siblings = {k: v for k, v in node.items() if k != "$ref"}
        node = {**target, **siblings}  # the referencing field's description/title/default win

    out: dict[str, Any] = {}
    notes: list[str] = []
    for key, value in node.items():
        if key in ("properties",):
            out[key] = {name: _convert(sub, defs, seen) for name, sub in value.items()}
        elif key in ("items", "additionalProperties") and isinstance(value, dict):
            out[key] = _convert(value, defs, seen)
        elif key in ("anyOf", "oneOf", "prefixItems"):
            branches = _convert(value, defs, seen)
            if key == "oneOf" and "discriminator" in node:
                branches = [_require_tag(b, node["discriminator"]["propertyName"]) for b in branches]
            out["anyOf" if key == "oneOf" else key] = branches
        elif key == "const":
            out["enum"] = [value]
        elif key in ("exclusiveMinimum", "exclusiveMaximum"):
            bound = "minimum" if key == "exclusiveMinimum" else "maximum"
            if bound not in node:
                out[bound] = value
            notes.append(f"must be {'>' if bound == 'minimum' else '<'} {value}")
        elif key == "default":
            if value is not None:
                notes.append(f"default: {json.dumps(value)}")
        elif key == "propertyNames":
            if isinstance(value, dict) and "enum" in value:
                notes.append("keys: " + ", ".join(map(str, value["enum"])))
        elif key in _DROPPED:
            continue
        else:
            out[key] = value
    if notes:
        text = f"({'; '.join(notes)})"
        out["description"] = f"{out['description']} {text}" if out.get("description") else text
    return out


def _require_tag(branch: dict[str, Any], tag: str) -> dict[str, Any]:
    """A discriminated union's tag must be sent even when it has a default (pydantic can't pick the branch
    without it); with `discriminator` dropped, `required` is the only way to say so."""
    if tag not in branch.get("properties", {}) or tag in branch.get("required", []):
        return branch
    return {**branch, "required": [*branch.get("required", []), tag]}


def check_gemini_schema(schema: Any, path: str = "$") -> None:
    """Raise SchemaConversionError unless `schema` uses only the documented subset."""
    if not isinstance(schema, dict):
        raise SchemaConversionError(f"{path}: schema must be an object, got {type(schema).__name__}")
    unknown = set(schema) - ALLOWED
    if unknown:
        raise SchemaConversionError(f"{path}: unsupported keywords {sorted(unknown)}")
    types = schema.get("type")
    for t in types if isinstance(types, list) else [types] if types is not None else []:
        if t not in TYPES:
            raise SchemaConversionError(f"{path}: unsupported type {t!r}")
    if "enum" in schema and not schema["enum"]:
        raise SchemaConversionError(f"{path}: empty enum")
    props = schema.get("properties", {})
    missing = set(schema.get("required", [])) - set(props)
    if missing:
        raise SchemaConversionError(f"{path}: required names without properties: {sorted(missing)}")
    for name, sub in props.items():
        check_gemini_schema(sub, f"{path}.{name}")
    for key in ("items", "additionalProperties"):
        if isinstance(schema.get(key), dict):
            check_gemini_schema(schema[key], f"{path}.{key}")
    for key in ("anyOf", "prefixItems"):
        for i, sub in enumerate(schema.get(key, [])):
            check_gemini_schema(sub, f"{path}.{key}[{i}]")


def dropped_keywords(schema: Any) -> set[str]:
    """Keywords in `schema` (ours) that `to_gemini_schema` rewrites or drops; for tests and docs."""
    found: set[str] = set()
    if isinstance(schema, list):
        for x in schema:
            found |= dropped_keywords(x)
    elif isinstance(schema, dict):
        found |= set(schema) - ALLOWED
        for key, value in schema.items():
            if key in ("properties", "$defs", "definitions"):
                for sub in value.values():
                    found |= dropped_keywords(sub)
            elif key not in ("enum", "default", "const", "discriminator"):  # values, not schemas
                found |= dropped_keywords(value)
    return found
