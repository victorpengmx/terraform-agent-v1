"""Eval judge (DESIGN §16): looks only, with its own prompt (eval/prompts.py), separate from the in-loop critic.

Inputs: the request text, the reference images and neutral L2 renders compiled from the final stack
(overview_sw and overview_ne; image prompts add overview_s, the photo's viewpoint, right after the reference).
No op list, feature graph, statistics or condition label. Output: short reasons, then scores (naturalism,
minecraft_quality, materials, reference_match: 1-5, reference_match null without references) and a verdict.
A run passes when the verdict is "pass" and every score is at least 3; code enforces the rule the prompt states.

Outputs are cached in eval/cache/judge/ keyed by (stack hash, judge-prompt hash, judge model), so re-scoring
never pays twice for the same terrain, request and prompt.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Literal

from agent.models import ModelAdapter, ModelReplyError, ModelResponse, Usage, structured_call
from agent.tools import image_block, text_block
from eval.prompts import JUDGE_SYSTEM
from terrain.opstack import OpStack
from terrain.schema import Strict

CACHE = Path("eval/cache/judge")
TEXT_VIEWS = ("overview_sw", "overview_ne")
IMAGE_VIEWS = ("overview_s", "overview_sw", "overview_ne")  # overview_s first: next to the reference
PASS_MIN = 3

Renderer = Callable[[OpStack, tuple[str, ...]], list[tuple[str, bytes]]]


class JudgeScores(Strict):
    naturalism: int
    minecraft_quality: int
    materials: int
    reference_match: int | None


class JudgeOutput(Strict):
    reasons: list[str]
    scores: JudgeScores
    verdict: Literal["pass", "fail"]


class JudgeError(ModelReplyError):
    """The judge's reply was unusable (refusal, truncation, invalid JSON, scores out of range)."""


def views_for(has_references: bool) -> tuple[str, ...]:
    return IMAGE_VIEWS if has_references else TEXT_VIEWS


def render_views(stack: OpStack, views: tuple[str, ...]) -> list[tuple[str, bytes]]:
    """Neutral L2 renders (titled with the camera only, never the op count)."""
    from terrain.compiler import compile_state
    from terrain.render.l2 import render_volume

    volume = compile_state(stack.state)
    return [(v, render_volume(volume, v, title=v)) for v in views]


def prompt_hash(prompt: str | None, references: list[bytes], views: tuple[str, ...]) -> str:
    """Everything besides the terrain and the model that the judge's answer depends on."""
    h = hashlib.sha256(JUDGE_SYSTEM.encode())
    h.update(json.dumps({"prompt": prompt or "", "references": [hashlib.sha256(r).hexdigest() for r in references],
                         "views": list(views)}).encode())
    return h.hexdigest()


def cache_key(stack: OpStack, prompt: str | None, references: list[bytes], model: str) -> str:
    stack_hash = hashlib.sha256(stack.to_json().encode()).hexdigest()
    p_hash = prompt_hash(prompt, references, views_for(bool(references)))
    return hashlib.sha256(f"{stack_hash}|{p_hash}|{model}".encode()).hexdigest()


def problems(out: JudgeOutput, has_references: bool) -> list[str]:
    s = out.scores
    found = [f"{k} = {v} is outside 1-5" for k, v in s.model_dump().items() if v is not None and not 1 <= v <= 5]
    if has_references and s.reference_match is None:
        found.append("reference_match is null but reference images were given")
    if not has_references and s.reference_match is not None:
        found.append("reference_match must be null without reference images")
    return found


def passes(out: JudgeOutput) -> bool:
    return out.verdict == "pass" and all(v >= PASS_MIN for v in out.scores.model_dump().values() if v is not None)


def ask(model: ModelAdapter, prompt: str | None, references: list[bytes],
        renders: list[tuple[str, bytes]]) -> tuple[JudgeOutput, ModelResponse]:
    """One judge call. Raises JudgeError (with the response, so its cost is still counted)."""
    content: list[dict[str, Any]] = [image_block(r) for r in references]
    request = f"# Request\n{prompt or '(no text; see the reference images)'}"
    if references:
        request += "\n\nReference image(s) are attached above."
    content.append(text_block(request + "\n\n# Renders of the final terrain"))
    for view, png in renders:
        content += [text_block(f"Render: {view}"), image_block(png)]
    out, response = structured_call(model, content, JUDGE_SYSTEM, JudgeOutput, JudgeError)
    if found := problems(out, bool(references)):
        raise JudgeError("invalid scores: " + "; ".join(found), response)
    return out, response


def judge(model: ModelAdapter, stack: OpStack, prompt: str | None, references: list[bytes],
          cache: Path | None = CACHE, render: Renderer = render_views, save_renders: Path | None = None) -> dict[str, Any]:
    """Judge a final stack: a cached record if there is one, else one call retried once on an unusable reply.

    Returns {model, key, cached, output, passed, mean, failed, cost_usd, paid_usd, usage}: cost_usd is what this call
    spent ($0 on a cache hit), paid_usd what the verdict cost when it was made. `passed` is None when both
    tries failed. Only successful verdicts are cached, so a failure is retried by the next rescore."""
    key = cache_key(stack, prompt, references, model.config.model)
    path = cache / f"{key}.json" if cache is not None else None
    if path is not None and path.exists():
        return json.loads(path.read_text(encoding="utf-8")) | {"cached": True, "cost_usd": 0.0}
    renders = render(stack, views_for(bool(references)))
    if save_renders is not None:
        save_renders.mkdir(parents=True, exist_ok=True)
        for view, png in renders:
            (save_renders / f"judge_{view}.png").write_bytes(png)
    usage, cost, errors = Usage(), 0.0, []
    record: dict[str, Any] = {"model": model.config.model, "key": key, "cached": False, "output": None, "passed": None,
                              "mean": None, "failed": None}
    for _ in range(2):
        try:
            out, response = ask(model, prompt, references, renders)
        except JudgeError as e:
            errors.append(str(e))
            response = e.response
            out = None
        if response is not None:
            for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
                setattr(usage, k, getattr(usage, k) + getattr(response.usage, k))
            c = response.usage.cost_usd(response.model)
            cost = None if cost is None or c is None else cost + c
        if out is not None:
            vals = [v for v in out.scores.model_dump().values() if v is not None]
            record |= {"output": out.model_dump(), "passed": passes(out), "mean": sum(vals) / len(vals)}
            break
    else:
        record["failed"] = "; ".join(errors)
    record |= {"cost_usd": cost, "paid_usd": cost, "usage": usage.__dict__.copy(), "errors": errors}
    if path is not None and record["output"] is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({k: v for k, v in record.items() if k not in ("cached", "cost_usd")}, indent=1),
                        encoding="utf-8")
    return record


def call_estimate(model: str, n_references: int) -> float | None:
    """Rough USD per judge call, for the cost gate: ~1.3k input tokens per image plus text, ~1.5k output."""
    images = n_references + len(views_for(bool(n_references)))
    return Usage(input_tokens=1500 + 1300 * images, output_tokens=1500).cost_usd(model)
