"""Module-level (picklable) stand-ins for eval runs in worker processes (tests/eval/test_run_eval.py)."""

import json
import os

from agent.models import ModelConfig
from tests.agent.fakes import PEAK, ScriptedModel, critique_json, feature, graph, response, spec_json, text, tool_use


def make_model(cfg: ModelConfig) -> ScriptedModel:
    if cfg.model == "claude-haiku-5-5":  # builder: its own SPEC, one peak, done
        return ScriptedModel(spec_json(graph(feature("big_peak", x="left"))),
                             response(tool_use("raise_peak", PEAK | {"center": [0.25, 0.4]}), stop="tool_use"),
                             response(text("done")))
    if cfg.model == "claude-opus-5-5":
        return ScriptedModel(critique_json(4, "pass"))
    judge = {"reasons": [f"pid {os.getpid()}"], "verdict": "pass",  # which process judged it
             "scores": {"naturalism": 4, "minecraft_quality": 4, "materials": 4, "reference_match": None}}
    return ScriptedModel(*[response(text(json.dumps(judge))) for _ in range(4)])


def render(stack, views):
    return [(v, b"\x89PNG " + v.encode()) for v in views]
