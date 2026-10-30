"""Gemini adapter (DESIGN §9.4): `ModelAdapter` over google-genai's `models.generate_content` (stateless).

The harness keeps Anthropic-style messages (text, image, tool_use, tool_result blocks); this module translates
at the boundary and is the only one that imports the Gemini SDK (lazily, via `agent.models.make_adapter`).
Docs checked 2026-10-30 against google-genai 2.34.0:

- Reply parts become blocks that keep the original `types.Part` and are sent back as that same Part. That
  keeps thought signatures intact: Gemini 3 answers 400 if the signature on a function-call part (only
  the first one of a parallel batch carries it) is missing in the next request
  (https://ai.google.dev/gemini-api/docs/thinking#signatures).
- `FunctionCall.id` is optional, so calls without one get a generated id. The call each tool_result answers
  is looked up by scanning the history on every request; the adapter keeps no state, so episode rotation
  can't break the mapping.
- Function responses use {"output": ...} or {"error": ...} (errors reach the model); images go in
  `FunctionResponse.parts`.
- Prompt-cache hits are implicit (no breakpoints, no write charge): `cached_content_token_count` is
  cache_read and is part of `prompt_token_count`.
- The SDK sends requests without a timeout, and on Windows Ctrl+C can't interrupt a blocked socket read. So
  every request gets a timeout, and it runs on a worker thread while the main thread waits in short slices
  (first smoke runs, 2026-10-30: a request that never returned hung the CLI).
"""

from __future__ import annotations

import base64
import os
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.gemini_schema import to_gemini_schema
from agent.models import ModelConfig, ModelReplyError, ModelResponse, Usage

# Our effort scale -> Gemini thinking_level (minimal/low/medium/high). Gemini has nothing above "high".
THINKING_LEVELS = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH", "xhigh": "HIGH", "max": "HIGH"}
# Finish reasons that mean the model declined or was filtered: the harness's "refusal".
REFUSAL_REASONS = frozenset({"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY",
                             "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION"})
MAX_RETRIES = 4  # after the first try: waits of about 1, 2, 4, 8 s
MAX_TIMEOUT_RETRIES = 1  # a timed-out request is retried once: each try can take the full timeout
RETRY_BASE_S, RETRY_MAX_S = 1.0, 30.0
REQUEST_TIMEOUT_S = 180.0  # slowest successful call so far: 110 s (gemini-3.8-flash, EVIDENCE 2026-10-30)


class GeminiStopError(ModelReplyError):
    """A finish reason the harness has no stop_reason for (e.g. OTHER, LANGUAGE). Carries the response, so its
    tokens are still charged; structured calls retry it once like any unusable reply."""


@dataclass
class GeminiText:
    text: str
    part: Any = None  # the types.Part it came from, re-sent unchanged
    type: str = field(default="text", init=False)


@dataclass
class GeminiToolUse:
    id: str
    name: str
    input: dict[str, Any]
    api_id: str | None = None  # the id Gemini gave the call, if any (generated ids are ours only)
    part: Any = None
    type: str = field(default="tool_use", init=False)


@dataclass
class GeminiThinking:
    thinking: str  # thought summary (only with thinking_display="summarized")
    part: Any = None
    type: str = field(default="thinking", init=False)


@dataclass
class GeminiPart:
    """Any other part (e.g. a bare signature carrier): kept so it goes back unchanged, otherwise ignored."""

    part: Any = None
    type: str = field(default="gemini_part", init=False)


@dataclass
class GeminiAdapter:
    config: ModelConfig = field(default_factory=lambda: ModelConfig(model="gemini-3.8-flash"))
    client: Any = None  # google.genai.Client; injectable for tests
    sleep: Callable[[float], None] = time.sleep  # injectable for retry tests
    timeout_s: float = REQUEST_TIMEOUT_S  # per try; a timeout counts as a retryable transport error
    _tools_cache: tuple[Any, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from google import genai
        except ImportError as e:
            raise ImportError('the Gemini adapter needs the gemini extra: pip install -e ".[gemini]"') from e
        if self.client is None:
            if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
                raise RuntimeError("no Gemini API key: set GEMINI_API_KEY in .env (the SDK also reads GOOGLE_API_KEY, "
                                   "which wins if both are set)")
            self.client = genai.Client()

    def complete(self, messages, *, system=None, tools=None, response_schema=None) -> ModelResponse:
        from google.genai import types

        cfg = self.config
        config: dict[str, Any] = {
            "max_output_tokens": cfg.max_tokens,
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
            "http_options": types.HttpOptions(timeout=int(self.timeout_s * 1000)),  # milliseconds
        }
        if system is not None:
            config["system_instruction"] = system if isinstance(system, str) else "\n\n".join(b["text"] for b in system)
        if tools:
            config["tools"] = self._tools(tools)
        if response_schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_json_schema"] = to_gemini_schema(response_schema)
        thinking: dict[str, Any] = {}
        if cfg.effort:
            thinking["thinking_level"] = types.ThinkingLevel[THINKING_LEVELS[cfg.effort]]
        if cfg.thinking_display == "summarized":
            thinking["include_thoughts"] = True
        if thinking:
            config["thinking_config"] = types.ThinkingConfig(**thinking)

        contents = to_contents(messages)
        start = time.perf_counter()
        raw, retries = self._generate(model=cfg.model, contents=contents, config=types.GenerateContentConfig(**config))
        return from_response(raw, cfg.model, time.perf_counter() - start, retries)

    def _tools(self, tools: list[dict[str, Any]]) -> list[Any]:
        """Function declarations, converted once per tools list (the harness passes the same list every call)."""
        from google.genai import types

        if self._tools_cache is not None and self._tools_cache[0] is tools:
            return self._tools_cache[1]
        declarations = [types.FunctionDeclaration(name=t["name"], description=t["description"],
                                                  parameters_json_schema=to_gemini_schema(t["input_schema"]))
                        for t in tools]
        converted = [types.Tool(function_declarations=declarations)]
        self._tools_cache = (tools, converted)
        return converted

    def _generate(self, **kwargs: Any) -> tuple[Any, list[dict[str, Any]]]:
        """generate_content with bounded exponential backoff on 429, 5xx and transport errors. Returns the reply
        and one record per failed try (error, seconds the try took, backoff wait), so a trace can tell slow
        serving from rate limiting."""
        import httpx
        from google.genai import errors

        retries: list[dict[str, Any]] = []
        for attempt in range(MAX_RETRIES + 1):
            start = time.perf_counter()
            try:
                return interruptible(self.client.models.generate_content, **kwargs), retries
            except (errors.APIError, httpx.TransportError) as e:
                code = getattr(e, "code", None)
                retryable = isinstance(e, httpx.TransportError) or code == 429 or (isinstance(code, int) and code >= 500)
                timeouts = sum(r["error"].endswith("Timeout") for r in retries if isinstance(r["error"], str))
                if isinstance(e, httpx.TimeoutException) and timeouts >= MAX_TIMEOUT_RETRIES:
                    retryable = False
                if not retryable or attempt == MAX_RETRIES:
                    raise
                wait = min(RETRY_MAX_S, RETRY_BASE_S * 2**attempt) * random.uniform(0.5, 1.0)
                retries.append({"error": code if code is not None else type(e).__name__,
                                "try_s": round(time.perf_counter() - start, 3), "wait_s": round(wait, 3)})
                self.sleep(wait)
        raise AssertionError("unreachable")


def interruptible(fn: Callable[..., Any], **kwargs: Any) -> Any:
    """fn(**kwargs) on a daemon thread; the caller waits in 0.2 s slices, so Ctrl+C (KeyboardInterrupt) gets
    through at once. An abandoned request ends with the process."""
    result: dict[str, Any] = {}

    def run() -> None:
        try:
            result["value"] = fn(**kwargs)
        except BaseException as e:  # noqa: BLE001 - re-raised on the caller's thread
            result["error"] = e

    worker = threading.Thread(target=run, name="gemini-request", daemon=True)
    worker.start()
    while worker.is_alive():
        worker.join(0.2)
    if "error" in result:
        raise result["error"]
    return result["value"]


def _get(block: Any, key: str, default: Any = None) -> Any:
    return block.get(key, default) if isinstance(block, dict) else getattr(block, key, default)


def _call_index(messages: list[dict[str, Any]]) -> dict[str, tuple[str, str | None]]:
    """tool_use id -> (function name, Gemini's own call id or None), from the assistant turns in the history."""
    index: dict[str, tuple[str, str | None]] = {}
    for m in messages:
        if m["role"] == "assistant" and not isinstance(m["content"], str):
            for b in m["content"]:
                if _get(b, "type") == "tool_use":
                    index[_get(b, "id")] = (_get(b, "name"), _get(b, "api_id"))
    return index


def to_contents(messages: list[dict[str, Any]]) -> list[Any]:
    """Anthropic-style messages -> Gemini contents. Assistant blocks from this adapter go back as their
    original Part (signatures included); everything else is translated."""
    from google.genai import types

    calls = _call_index(messages)
    contents = []
    for m in messages:
        blocks = [{"type": "text", "text": m["content"]}] if isinstance(m["content"], str) else m["content"]
        parts: list[Any] = []
        for b in blocks:
            parts.extend(_to_parts(b, calls))
        role = "model" if m["role"] == "assistant" else "user"
        if not parts:  # an empty reply (e.g. a malformed call) has nothing to send back
            continue
        if contents and contents[-1].role == role:  # e.g. tool results, then feedback after an empty reply
            contents[-1].parts.extend(parts)
        else:
            contents.append(types.Content(role=role, parts=parts))
    return contents


def _to_parts(block: Any, calls: dict[str, tuple[str, str | None]]) -> list[Any]:
    from google.genai import types

    if _get(block, "part") is not None:
        return [block.part]
    kind = _get(block, "type")
    if kind == "text":
        return [types.Part(text=_get(block, "text"))]
    if kind == "image":
        source = _get(block, "source")
        return [types.Part.from_bytes(data=base64.b64decode(source["data"]), mime_type=source["media_type"])]
    if kind == "tool_use":  # from history this adapter didn't produce: no thought signature to send back
        return [types.Part(function_call=types.FunctionCall(name=_get(block, "name"), args=_get(block, "input")))]
    if kind == "tool_result":
        call_id = _get(block, "tool_use_id")
        if call_id not in calls:
            raise ValueError(f"tool_result for unknown call id {call_id!r}: no tool_use with that id in the history")
        name, api_id = calls[call_id]
        content = _get(block, "content")
        items = [{"type": "text", "text": content}] if isinstance(content, str) else content
        text = "\n".join(i["text"] for i in items if i["type"] == "text")
        images = [types.FunctionResponsePart(inline_data=types.FunctionResponseBlob(
            data=base64.b64decode(i["source"]["data"]), mime_type=i["source"]["media_type"])) for i in items if i["type"] == "image"]
        response = {"error": text} if _get(block, "is_error") else {"output": text}
        return [types.Part(function_response=types.FunctionResponse(id=api_id, name=name, response=response, parts=images or None))]
    raise ValueError(f"the Gemini adapter can't send a {kind!r} block")


def _name(value: Any) -> str | None:
    """Enum member or raw string -> its name. The SDK keeps reasons it doesn't know as pseudo-members."""
    return None if value is None else getattr(value, "value", str(value))


def _dump(items: Any) -> list[dict[str, Any]]:
    return [i.model_dump(mode="json", exclude_none=True) for i in items or []]


def usage_from(metadata: Any) -> Usage:
    """prompt_token_count includes the cached tokens; thinking tokens are billed as output."""
    if metadata is None:
        return Usage()
    cached = metadata.cached_content_token_count or 0
    return Usage(
        input_tokens=(metadata.prompt_token_count or 0) - cached + (metadata.tool_use_prompt_token_count or 0),
        output_tokens=(metadata.candidates_token_count or 0) + (metadata.thoughts_token_count or 0),
        cache_read_tokens=cached,
    )


def _blocks(parts: list[Any], response_id: str | None) -> list[Any]:
    prefix = f"gc_{response_id or uuid.uuid4().hex[:12]}"
    blocks: list[Any] = []
    calls = 0
    for part in parts:
        if part.function_call is not None:
            fc = part.function_call
            blocks.append(GeminiToolUse(id=fc.id or f"{prefix}_{calls}", name=fc.name, input=dict(fc.args or {}),
                                        api_id=fc.id, part=part))
            calls += 1
        elif part.thought:
            blocks.append(GeminiThinking(thinking=part.text or "", part=part))
        elif part.text is not None:
            blocks.append(GeminiText(text=part.text, part=part))
        else:
            blocks.append(GeminiPart(part=part))
    return blocks


def from_response(raw: Any, model: str, latency_s: float, retries: list[dict[str, Any]] | None = None) -> ModelResponse:
    """GenerateContentResponse -> ModelResponse. `model` is the configured id (prices are keyed by it)."""
    response = ModelResponse(content=[], stop_reason=None, usage=usage_from(raw.usage_metadata), model=model,
                             request_id=raw.response_id, latency_s=latency_s, retries=retries or [])
    feedback = raw.prompt_feedback
    if not raw.candidates:
        if feedback is not None and feedback.block_reason is not None:
            response.stop_reason = "refusal"
            response.stop_details = {"block_reason": _name(feedback.block_reason), "message": feedback.block_reason_message,
                                     "safety_ratings": _dump(feedback.safety_ratings)}
            return response
        response.stop_reason = "gemini:no_candidates"
        raise GeminiStopError("Gemini returned no candidates and no block reason", response)

    candidate = raw.candidates[0]
    response.content = _blocks((candidate.content.parts or []) if candidate.content else [], raw.response_id)
    reason = _name(candidate.finish_reason)
    if reason == "STOP":
        response.stop_reason = "tool_use" if response.tool_uses else "end_turn"
    elif reason == "MAX_TOKENS":
        response.stop_reason = "max_tokens"
    elif reason == "MALFORMED_FUNCTION_CALL":  # the harness tells the model and lets it re-issue the call
        response.stop_reason = "malformed_tool_call"
        response.stop_details = {"finish_reason": reason, "message": candidate.finish_message}
    elif reason in REFUSAL_REASONS:
        response.stop_reason = "refusal"
        response.stop_details = {"finish_reason": reason, "message": candidate.finish_message,
                                 "safety_ratings": _dump(candidate.safety_ratings)}
    else:
        response.stop_reason = f"gemini:{reason}"
        raise GeminiStopError(f"Gemini stopped with finish reason {reason!r} (message: {candidate.finish_message!r})", response)
    return response
