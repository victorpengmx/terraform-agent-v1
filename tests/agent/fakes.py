"""Scripted stand-ins for the model adapter, so harness and critic tests run without the network."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from agent.models import ModelConfig, ModelResponse, Usage

PEAK = {"center": [0.5, 0.5], "radius": 0.25, "height": 40}


def text(t: str) -> Any:
    return SimpleNamespace(type="text", text=t)


def tool_use(name: str, inp: dict, id: str = "tu_1") -> Any:
    return SimpleNamespace(type="tool_use", id=id, name=name, input=inp)


def response(*blocks: Any, stop: str = "end_turn", input_tokens: int = 100) -> ModelResponse:
    return ModelResponse(
        content=list(blocks), stop_reason=stop, usage=Usage(input_tokens, 50), model="claude-haiku-5-5",
        request_id="req", latency_s=0.01,
    )


def critique_json(score: int, verdict: str = "revise") -> ModelResponse:
    body = {
        "issues": [{"location": "raise_peak_1", "severity": "major", "description": "too round", "fix_hint": "raise sharpness"}],
        "scores": {k: score for k in ["prompt_adherence", "layout", "naturalism", "minecraft_quality"]} | {"reference_match": None},
        "verdict": verdict,
    }
    return response(text(json.dumps(body)))


class ScriptedModel:
    """Returns the scripted responses in order and records every call."""

    def __init__(self, *responses: ModelResponse) -> None:
        self.config = ModelConfig(model="claude-haiku-5-5")
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages, *, system=None, tools=None, response_schema=None) -> ModelResponse:
        self.calls.append({"messages": [dict(m) for m in messages], "system": system, "tools": tools, "response_schema": response_schema})
        if not self.responses:
            raise AssertionError("ScriptedModel ran out of responses")
        return self.responses.pop(0)
