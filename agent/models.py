"""Model adapter (DESIGN §9.4): one interface for every role (builder, critic, judge, dev).

The harness talks to `ModelAdapter.complete()` only, so models and providers can be swapped by config
for ablations. Messages and tools use the Anthropic format as the internal lingua franca; adapters for
other providers (e.g. a Gemini critic) translate at this boundary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import anthropic

Effort = Literal["low", "medium", "high", "xhigh", "max"]

# USD per million tokens (Anthropic list prices, Oct 2026). Used for budgets and eval reports only.
PRICES: dict[str, dict[str, float]] = {
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01},  # cache_read assumed 0.1x input
}
CACHE_WRITE_MULTIPLIER = 1.25  # 5-minute TTL cache writes

# Models that accept the server-side refusal fallback (`fallbacks="default"`).
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}


@dataclass(frozen=True)
class ModelConfig:
    model: str = "claude-opus-5-5"
    max_tokens: int = 16000  # thinking counts against this; don't lowball it
    effort: Effort | None = "high"  # Opus 5.5 defaults to "medium" if unset
    refusal_fallback: bool = True  # re-run declined requests on a fallback model (where supported)
    thinking_display: Literal["omitted", "summarized"] | None = None  # "summarized" = readable reasoning in traces


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def cost_usd(self, model: str) -> float | None:
        p = PRICES.get(model)
        if p is None:
            return None
        return (
            self.input_tokens * p["input"]
            + self.output_tokens * p["output"]
            + self.cache_read_tokens * p["cache_read"]
            + self.cache_write_tokens * p["input"] * CACHE_WRITE_MULTIPLIER
        ) / 1e6


@dataclass
class ModelResponse:
    content: list[Any]  # raw content blocks: append these unchanged to the history (keeps thinking blocks)
    stop_reason: str | None  # end_turn | tool_use | max_tokens | refusal | pause_turn | ...
    usage: Usage
    model: str  # the model that actually served the request (may differ after a fallback)
    request_id: str | None
    latency_s: float
    stop_details: Any = None  # populated only for refusals

    @property
    def tool_uses(self) -> list[Any]:
        return [b for b in self.content if b.type == "tool_use"]

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content if b.type == "text")


class ModelAdapter(Protocol):
    config: ModelConfig

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """One model call. `response_schema` constrains the reply to JSON matching that schema
        (no numeric/string-length constraints; every object needs additionalProperties: false)."""
        ...


@dataclass
class AnthropicAdapter:
    config: ModelConfig = field(default_factory=ModelConfig)
    client: Any = None  # anthropic.Anthropic; injectable for tests

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = anthropic.Anthropic()  # credentials from ANTHROPIC_API_KEY or `ant auth login`

    def complete(self, messages, *, system=None, tools=None, response_schema=None) -> ModelResponse:
        cfg = self.config
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "max_tokens": cfg.max_tokens,
            "messages": messages,
            "cache_control": {"type": "ephemeral"},  # auto-cache the longest stable prefix
        }
        if system is not None:
            # Separate breakpoint at the end of tools + system: every new episode and every critic call
            # shares this prefix, but the automatic breakpoint (end of messages) never matches across episodes.
            if isinstance(system, str):
                system = [{"type": "text", "text": system}]
            system = [*system[:-1], {**system[-1], "cache_control": {"type": "ephemeral"}}]
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        output_config: dict[str, Any] = {}
        if cfg.effort:
            output_config["effort"] = cfg.effort
        if response_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": response_schema}
        if output_config:
            kwargs["output_config"] = output_config
        if cfg.thinking_display:
            kwargs["thinking"] = {"type": "adaptive", "display": cfg.thinking_display}

        start = time.perf_counter()
        if cfg.refusal_fallback and cfg.model in FALLBACK_MODELS:
            msg = self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
            )
        else:
            msg = self.client.messages.create(**kwargs)
        latency = time.perf_counter() - start

        u = msg.usage
        return ModelResponse(
            content=list(msg.content),
            stop_reason=msg.stop_reason,
            usage=Usage(
                input_tokens=u.input_tokens or 0,
                output_tokens=u.output_tokens or 0,
                cache_read_tokens=getattr(u, "cache_read_input_tokens", None) or 0,
                cache_write_tokens=getattr(u, "cache_creation_input_tokens", None) or 0,
            ),
            model=msg.model,
            request_id=getattr(msg, "_request_id", None),
            latency_s=latency,
            stop_details=getattr(msg, "stop_details", None),
        )
