"""Gemini schema conversion (agent/gemini_schema.py) over every schema a model sees. No SDK needed."""

import copy

import pytest

from agent.critic import CompareOutput, CritiqueOutput
from agent.gemini_schema import (SchemaConversionError, check_gemini_schema, dropped_keywords,
                                 to_gemini_schema)
from agent.session import Session
from agent.spec import FeatureGraph
from agent.tools import control_tools, observe_tools, op_tools
from terrain.ops import list_ops

SESSION = Session.flat(33, 33, base_y=64, seed=1)
OP_TOOLS = {t.name: t for t in op_tools(SESSION)}
OUTPUTS = {m.__name__: m for m in (CritiqueOutput, CompareOutput, FeatureGraph)}
# What our schemas use outside the documented subset today. A new keyword fails here, so it gets a rule.
CONVERTED = {"$defs", "$ref", "oneOf", "discriminator", "const", "exclusiveMinimum", "default", "propertyNames", "minLength"}


def synthesize(s: dict):
    """A payload built from a converted schema only: first enum value, mid-range numbers, minItems items,
    first non-null anyOf branch, required properties."""
    if "anyOf" in s:
        return synthesize(next(b for b in s["anyOf"] if b.get("type") != "null"))
    if "enum" in s:
        return s["enum"][0]
    t = s.get("type")
    if t == "object":
        return {k: synthesize(s["properties"][k]) for k in s.get("required", [])}
    if t == "array":
        if "prefixItems" in s:
            return [synthesize(x) for x in s["prefixItems"]]
        return [synthesize(s["items"]) for _ in range(max(s.get("minItems", 1), 1))]
    if t in ("number", "integer"):
        lo, hi = s.get("minimum"), s.get("maximum")
        v = (lo + hi) / 2 if lo is not None and hi is not None else (lo + 1 if lo is not None else (hi - 1 if hi is not None else 1))
        return round(v) if t == "integer" else v
    if t == "string":
        return "x"
    if t == "boolean":
        return False
    raise AssertionError(f"cannot synthesize {s}")


# Fields whose cross-field validators the synthesizer can't satisfy from the schema alone.
OVERRIDES = {"raise_region": {"height": 10}}


@pytest.mark.parametrize("name", sorted(OP_TOOLS))
def test_every_op_schema_converts_and_still_describes_valid_params(name):
    schema = OP_TOOLS[name].input_schema
    before = copy.deepcopy(schema)
    converted = to_gemini_schema(schema)
    assert schema == before  # the Anthropic schema is untouched
    check_gemini_schema(converted)
    assert not dropped_keywords(converted)
    payload = synthesize(converted) | OVERRIDES.get(name, {})
    params = next(s.params for s in list_ops() if s.name == name)
    params.model_validate({k: v for k, v in payload.items() if k not in ("id", "feature")})


@pytest.mark.parametrize("tool", [*control_tools(SESSION), *observe_tools(SESSION, ".")], ids=lambda t: t.name)
def test_control_and_observe_tools_convert(tool):
    check_gemini_schema(to_gemini_schema(tool.input_schema))


@pytest.mark.parametrize("name", sorted(OUTPUTS))
def test_structured_outputs_convert_and_synthesized_replies_validate(name):
    model = OUTPUTS[name]
    converted = to_gemini_schema(model.model_json_schema())
    check_gemini_schema(converted)
    assert converted["additionalProperties"] is False
    model.model_validate(synthesize(converted))


def test_only_known_keywords_are_converted():
    found = set().union(*(dropped_keywords(t.input_schema) for t in OP_TOOLS.values()),
                        *(dropped_keywords(m.model_json_schema()) for m in OUTPUTS.values()))
    assert found == CONVERTED


def test_conversion_rules():
    schema = {
        "type": "object",
        "$defs": {"Kind": {"type": "string", "const": "a"}, "Box": {"type": "object", "properties": {"w": {"$ref": "#/$defs/Kind"}}}},
        "properties": {
            "box": {"$ref": "#/$defs/Box", "description": "the box"},
            "u": {"oneOf": [{"$ref": "#/$defs/Kind"}, {"type": "null"}], "discriminator": {"propertyName": "kind"}},
            "width": {"type": "number", "exclusiveMinimum": 0, "maximum": 1, "default": 0.5, "description": "Width."},
            "name": {"type": "string", "minLength": 1},
            "weights": {"type": "object", "propertyNames": {"enum": ["stone", "dirt"]}, "additionalProperties": {"type": "number"}},
        },
    }
    out = to_gemini_schema(schema)
    assert "$defs" not in out
    assert out["properties"]["box"] == {"type": "object", "properties": {"w": {"type": "string", "enum": ["a"]}}, "description": "the box"}
    assert out["properties"]["u"] == {"anyOf": [{"type": "string", "enum": ["a"]}, {"type": "null"}]}
    assert out["properties"]["width"] == {"type": "number", "minimum": 0, "maximum": 1, "description": "Width. (must be > 0; default: 0.5)"}
    assert out["properties"]["name"] == {"type": "string"}
    assert out["properties"]["weights"]["description"] == "(keys: stone, dirt)"
    check_gemini_schema(out)


def test_a_discriminator_tag_with_a_default_becomes_required():
    branch = {"type": "object", "properties": {"kind": {"const": "single", "default": "single"}, "block": {"type": "string"}},
              "required": ["block"]}
    out = to_gemini_schema({"oneOf": [branch], "discriminator": {"propertyName": "kind"}})
    assert out["anyOf"][0]["required"] == ["block", "kind"] and "discriminator" not in out


def test_recursion_and_unknown_refs_fail_loudly():
    with pytest.raises(SchemaConversionError, match="recursive"):
        to_gemini_schema({"$defs": {"N": {"type": "object", "properties": {"next": {"$ref": "#/$defs/N"}}}}, "$ref": "#/$defs/N"})
    with pytest.raises(SchemaConversionError, match="unsupported \\$ref"):
        to_gemini_schema({"$ref": "https://example.com/s.json"})
    with pytest.raises(SchemaConversionError, match="unsupported keywords"):
        check_gemini_schema({"type": "object", "properties": {"a": {"type": "string", "pattern": "x"}}})
