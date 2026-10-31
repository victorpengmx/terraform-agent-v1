"""Model adapter (DESIGN §9.4): one interface for every role (builder, critic, judge, dev).

The harness talks to `ModelAdapter.complete()` only, so models and providers can be swapped by config
for ablations. Messages and tools use the Anthropic format as the internal lingua franca; adapters for
other providers (e.g. a Gemini critic) translate at this boundary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

Effort = Literal["low", "medium", "high", "xhigh", "max"]

# USD per million tokens. Used for budgets and eval reports only. A missing price (None) makes a call's
# cost unknown, never $0.
PRICES: dict[str, dict[str, float | None]] = {
    # Anthropic list prices, Oct 2026.
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01},  # cache_read assumed 0.1x input
    # Gemini: ids from https://ai.google.dev/gemini-api/docs/models (2026-10-30). Output includes thinking tokens;
    # cache_read is the context-caching (implicit cache hit) price.
    # TODO(victor): fill in from https://ai.google.dev/gemini-api/docs/pricing (Standard tier).
    # gemini-3.1-pro-preview is priced in two tiers by prompt length; this table holds one rate.
    "gemini-4-argon": {"input": None, "output": None, "cache_read": None},
    "gemini-3.8-flash": {"input": None, "output": None, "cache_read": None},
    "gemini-3.6-flash": {"input": None, "output": None, "cache_read": None},
    "gemini-3.1-pro-preview": {"input": None, "output": None, "cache_read": None},
    # Standard tier, from https://ai.google.dev/gemini-api/docs/pricing (2026-10-31); the M6 eval judge.
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50, "cache_read": 0.025},
}
CACHE_WRITE_MULTIPLIER = 1.25  # 5-minute TTL cache writes

# Models that accept the server-side refusal fallback (`fallbacks="default"`).
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}


Provider = Literal["anthropic", "gemini"]
PROVIDER_PREFIXES: dict[str, Provider] = {"claude-": "anthropic", "gemini-": "gemini"}


@dataclass(frozen=True)
class ModelConfig:
    model: str = "claude-opus-5-5"
    max_tokens: int = 16000  # thinking counts against this; don't lowball it
    effort: Effort | None = "high"  # Opus 5.5 defaults to "medium" if unset
    refusal_fallback: bool = True  # re-run declined requests on a fallback model (where supported; Anthropic only)
    thinking_display: Literal["omitted", "summarized"] | None = None  # "summarized" = readable reasoning in traces
    provider: Provider | None = None  # None: inferred from the model id prefix

    def __post_init__(self) -> None:
        if self.provider is None:
            inferred = next((p for prefix, p in PROVIDER_PREFIXES.items() if self.model.startswith(prefix)), None)
            if inferred is None:
                raise ValueError(f"can't infer the provider of model {self.model!r}: ids start with "
                                 f"{' or '.join(PROVIDER_PREFIXES)}, or pass provider= ({' | '.join(PROVIDER_PREFIXES.values())})")
            object.__setattr__(self, "provider", inferred)
        elif self.provider not in PROVIDER_PREFIXES.values():
            raise ValueError(f"unknown provider {self.provider!r}; use {' or '.join(PROVIDER_PREFIXES.values())}")


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def cost_usd(self, model: str) -> float | None:
        """None when the model or one of its prices is unknown (reported as "unknown", never as $0)."""
        if not price_known(model):
            return None
        p = PRICES[model]
        return (
            self.input_tokens * p["input"]
            + self.output_tokens * p["output"]
            + self.cache_read_tokens * p["cache_read"]
            + self.cache_write_tokens * p["input"] * CACHE_WRITE_MULTIPLIER
        ) / 1e6


def price_known(model: str) -> bool:
    p = PRICES.get(model)
    return p is not None and all(v is not None for v in p.values())


@dataclass
class ModelResponse:
    content: list[Any]  # raw content blocks: append these unchanged to the history (keeps thinking blocks)
    stop_reason: str | None  # end_turn | tool_use | max_tokens | refusal | pause_turn | ...
    usage: Usage
    model: str  # the model that actually served the request (may differ after a fallback)
    request_id: str | None
    latency_s: float
    stop_details: Any = None  # populated only for refusals
    # Failed tries the adapter retried before this reply ({error, try_s, wait_s}); latency_s includes them.
    # Empty for Anthropic, whose SDK retries internally without reporting it.
    retries: list[dict[str, Any]] = field(default_factory=list)

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


class ModelReplyError(Exception):
    """A structured reply was unusable (refusal, truncation, invalid JSON)."""

    def __init__(self, message: str, response: ModelResponse | None = None) -> None:
        super().__init__(message)
        self.response = response  # the unusable reply, so its tokens are still charged and traced


_T = TypeVar("_T", bound=BaseModel)


def structured_call(model: ModelAdapter, content: list[dict[str, Any]], system: str, schema: type[_T],
                    error: type[ModelReplyError] = ModelReplyError) -> tuple[_T, ModelResponse]:
    """One strict-JSON call in a fresh context (one user message). Raises `error` (a ModelReplyError
    subclass, so callers keep their own except clauses), carrying the response so its cost is still charged."""
    response = model.complete([{"role": "user", "content": content}], system=system,
                              response_schema=schema.model_json_schema())
    if response.stop_reason != "end_turn":
        raise error(f"stopped with {response.stop_reason!r} (details: {response.stop_details})", response)
    try:
        return schema.model_validate_json(response.text), response
    except ValidationError as e:
        raise error(f"returned invalid JSON: {e}", response) from None


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


def make_adapter(config: ModelConfig) -> ModelAdapter:
    """The adapter for `config.provider`. Provider SDKs other than Anthropic's are imported only here, lazily,
    so an install without their extras still works."""
    if config.provider == "gemini":
        from agent.gemini import GeminiAdapter

        return GeminiAdapter(config)
    return AnthropicAdapter(config)
