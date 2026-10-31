"""Gemini adapter contract tests (agent/gemini.py): recorded-shape responses, no network.

Replies are REST-shaped JSON (camelCase, as the API returns them) parsed by the SDK's own
`GenerateContentResponse`, so field names are checked against the installed google-genai. Skipped without the
gemini extra.
"""

import base64
import json
import threading
from types import SimpleNamespace

import httpx

import pytest

genai_types = pytest.importorskip("google.genai.types")
from google.genai import errors  # noqa: E402

from agent.budget import Budget, BudgetTracker  # noqa: E402
from agent.critic import CritiqueOutput  # noqa: E402
from agent.gemini import GeminiAdapter, GeminiStopError, to_contents  # noqa: E402
from agent.harness import run_task  # noqa: E402
from agent.models import AnthropicAdapter, ModelConfig, Usage, make_adapter, structured_call  # noqa: E402
from agent.session import Session  # noqa: E402
from agent.tools import image_block, op_tools, text_block  # noqa: E402
from agent.tracing import Tracer  # noqa: E402
from fakes import PEAK  # noqa: E402

MODEL = "gemini-3.8-flash"
UNPRICED = "gemini-3.6-flash"  # in PRICES without numbers
SIG = b"\x00opaque-signature\xff"


def reply(*parts, finish="STOP", usage=None, response_id="resp_1", **extra):
    body = {"responseId": response_id, "modelVersion": MODEL,
            "candidates": [{"content": {"role": "model", "parts": list(parts)}, "finishReason": finish, **extra}],
            "usageMetadata": usage or {"promptTokenCount": 1200, "cachedContentTokenCount": 1000,
                                       "candidatesTokenCount": 30, "thoughtsTokenCount": 70}}
    return genai_types.GenerateContentResponse.model_validate(body)


def call(name, args, id=None, signature=None):
    part = {"functionCall": {"name": name, "args": args, **({"id": id} if id else {})}}
    if signature:
        part["thoughtSignature"] = base64.b64encode(signature).decode()
    return part


class FakeClient:
    """Records generate_content kwargs; returns the scripted replies (or raises scripted errors) in order."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, **kwargs):
        self.calls.append(kwargs)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def adapter(*replies, model=MODEL, **config):
    client = FakeClient(*replies)
    return GeminiAdapter(ModelConfig(model=model, **config), client=client, sleep=lambda s: None), client


def test_text_reply_usage_and_request_config():
    model, client = adapter(reply({"text": "hi"}))
    r = model.complete([{"role": "user", "content": "x"}], system="sys")
    assert r.text == "hi" and r.stop_reason == "end_turn" and r.request_id == "resp_1" and r.model == MODEL
    # prompt 1200 includes 1000 cached; thinking tokens are output.
    assert r.usage == Usage(input_tokens=200, output_tokens=100, cache_read_tokens=1000, cache_write_tokens=0)
    config = client.calls[0]["config"]
    assert config.system_instruction == "sys" and config.max_output_tokens == 16000
    assert config.automatic_function_calling.disable is True
    assert config.thinking_config.thinking_level == genai_types.ThinkingLevel.HIGH
    assert config.thinking_config.include_thoughts is None and config.tools is None
    assert [c.role for c in client.calls[0]["contents"]] == ["user"]


def test_tool_round_trip_through_the_harness(tmp_path):
    """Two parallel calls (the first with an id and the batch's signature, the second with neither); one
    fails validation. The next request carries the model turn unchanged, then both responses in call order."""
    session = Session.flat(33, 33, base_y=64)
    first = reply({"text": "Two peaks.", "thoughtSignature": base64.b64encode(b"text-sig").decode()},
                  call("raise_peak", PEAK, id="call_a", signature=SIG),
                  call("raise_peak", PEAK | {"radius": 5}))
    model, client = adapter(first, reply({"text": "Done."}))
    result = run_task(prompt="a hill", images=[], session=session, tools={t.name: t for t in op_tools(session)},
                      model=model, tracer=Tracer(run_id="t", root=tmp_path), budget=Budget(), use_spec=False)
    assert result.stop == "done" and result.summary == "Done." and [e.id for e in session.ops] == ["raise_peak_1"]

    contents = client.calls[1]["contents"]
    assert [c.role for c in contents] == ["user", "model", "user"]
    sent = contents[1].parts
    assert sent == first.candidates[0].content.parts  # the same Parts, nothing rebuilt
    assert sent[0].thought_signature == b"text-sig" and sent[1].thought_signature == SIG and sent[2].thought_signature is None
    responses = [p.function_response for p in contents[2].parts]
    assert [(r.id, r.name) for r in responses] == [("call_a", "raise_peak"), (None, "raise_peak")]
    assert "output" in responses[0].response and "radius" in responses[1].response["error"]
    # Function declarations are converted (no $defs) and identical on every call (prompt caching).
    decl = client.calls[0]["config"].tools[0].function_declarations
    assert "write_note" in [d.name for d in decl] and all("$defs" not in json.dumps(d.parameters_json_schema) for d in decl)
    assert client.calls[1]["config"].tools == client.calls[0]["config"].tools


def test_missing_call_ids_are_generated_and_stable_within_a_reply():
    model, _ = adapter(reply(call("list_ops", {}), call("list_ops", {}), response_id="r9"))
    r = model.complete([{"role": "user", "content": "x"}], tools=[{"name": "list_ops", "description": "d", "input_schema": {"type": "object"}}])
    assert r.stop_reason == "tool_use" and [t.id for t in r.tool_uses] == ["gc_r9_0", "gc_r9_1"]
    assert [t.api_id for t in r.tool_uses] == [None, None]


def test_tool_result_for_an_unknown_call_fails_loudly():
    with pytest.raises(ValueError, match="unknown call id"):
        to_contents([{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "nope", "content": "x", "is_error": False}]}])


def test_images_become_inline_data_in_messages_and_tool_results():
    png, jpeg = b"\x89PNG\r\n\x1a\nfake", b"\xff\xd8\xff fake"
    history = [
        {"role": "user", "content": [image_block(png), text_block("look")]},
        {"role": "assistant", "content": [SimpleNamespace(type="tool_use", id="t1", name="render_view", input={})]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "is_error": False,
                                      "content": [text_block("map view"), image_block(jpeg)]}]},
    ]
    contents = to_contents(history)
    assert (contents[0].parts[0].inline_data.mime_type, contents[0].parts[0].inline_data.data) == ("image/png", png)
    fr = contents[2].parts[0].function_response
    assert fr.response == {"output": "map view"}
    assert (fr.parts[0].inline_data.mime_type, fr.parts[0].inline_data.data) == ("image/jpeg", jpeg)


def test_structured_output_validates_into_our_model():
    body = {"issues": [], "scores": {k: 3 for k in ["prompt_adherence", "layout", "naturalism", "minecraft_quality", "materials"]}
            | {"reference_match": None}, "verdict": "pass"}
    model, client = adapter(reply({"text": json.dumps(body)}))
    out, response = structured_call(model, [text_block("judge")], "sys", CritiqueOutput)
    assert out.verdict == "pass" and response.stop_reason == "end_turn"
    config = client.calls[0]["config"]
    assert config.response_mime_type == "application/json"
    assert "$defs" not in json.dumps(config.response_json_schema) and config.response_json_schema["additionalProperties"] is False


def test_max_tokens_and_safety_and_prompt_blocks():
    model, _ = adapter(reply({"text": "cut"}, finish="MAX_TOKENS"))
    assert model.complete([{"role": "user", "content": "x"}]).stop_reason == "max_tokens"

    ratings = [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "probability": "HIGH", "blocked": True}]
    model, _ = adapter(reply(finish="SAFETY", finishMessage="unsafe", safetyRatings=ratings))
    r = model.complete([{"role": "user", "content": "x"}])
    assert r.stop_reason == "refusal" and r.stop_details["finish_reason"] == "SAFETY" and r.stop_details["message"] == "unsafe"
    assert r.stop_details["safety_ratings"][0]["blocked"] is True

    blocked = genai_types.GenerateContentResponse.model_validate(
        {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}, "usageMetadata": {"promptTokenCount": 10}})
    model, _ = adapter(blocked)
    r = model.complete([{"role": "user", "content": "x"}])
    assert r.stop_reason == "refusal" and r.stop_details["block_reason"] == "PROHIBITED_CONTENT" and r.content == []


# MISSING_THOUGHT_SIGNATURE is in the API reference but not in the SDK's FinishReason enum (2.34.0): the SDK
# warns and keeps it as a pseudo-member, which must still map to its raw name.
@pytest.mark.filterwarnings("ignore:MISSING_THOUGHT_SIGNATURE is not a valid FinishReason")
@pytest.mark.parametrize("reason", ["MISSING_THOUGHT_SIGNATURE", "OTHER", "LANGUAGE"])
def test_other_finish_reasons_raise_with_the_raw_reason(reason):
    model, _ = adapter(reply({"text": ""}, finish=reason))
    with pytest.raises(GeminiStopError, match=reason) as e:
        model.complete([{"role": "user", "content": "x"}])
    assert e.value.response.stop_reason == f"gemini:{reason}" and e.value.response.usage.output_tokens == 100  # still charged


def test_thinking_summaries_and_effort_mapping():
    summary = {"text": "I should raise a peak.", "thought": True}
    model, client = adapter(reply(summary, {"text": "ok"}), effort="max", thinking_display="summarized")
    r = model.complete([{"role": "user", "content": "x"}])
    assert r.text == "ok" and [b.type for b in r.content] == ["thinking", "text"]
    thinking = client.calls[0]["config"].thinking_config
    assert thinking.thinking_level == genai_types.ThinkingLevel.HIGH and thinking.include_thoughts is True
    # The thought part goes back unchanged too.
    assert to_contents([{"role": "assistant", "content": r.content}])[0].parts[0].thought is True


def _api_error(code):
    return errors.ClientError(code, {"error": {"code": code, "message": "m", "status": "S"}}) if code < 500 else \
        errors.ServerError(code, {"error": {"code": code, "message": "m", "status": "S"}})


def test_retries_on_429_and_5xx_but_not_400():
    model, client = adapter(_api_error(429), _api_error(503), reply({"text": "hi"}))
    r = model.complete([{"role": "user", "content": "x"}])
    assert r.text == "hi" and len(client.calls) == 3
    assert [x["error"] for x in r.retries] == [429, 503]
    assert 0.5 <= r.retries[0]["wait_s"] <= 1.0 and 1.0 <= r.retries[1]["wait_s"] <= 2.0  # 1 s, 2 s with jitter

    model, client = adapter(_api_error(400))
    with pytest.raises(errors.ClientError):
        model.complete([{"role": "user", "content": "x"}])
    assert len(client.calls) == 1

    model, client = adapter(*[_api_error(500)] * 5)
    with pytest.raises(errors.ServerError):
        model.complete([{"role": "user", "content": "x"}])
    assert len(client.calls) == 5  # first try + 4 retries


def test_retries_reach_the_trace(tmp_path):
    model, _ = adapter(_api_error(429), reply({"text": "nothing to do"}))
    run_task(prompt="a hill", images=[], session=Session.flat(33, 33, base_y=64), tools={}, model=model,
             tracer=Tracer(run_id="t", root=tmp_path), budget=Budget(), use_spec=False)
    steps = [json.loads(l) for l in (tmp_path / "t.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [x["error"] for x in steps[0]["retries"]] == [429] and steps[-1]["retries"] is None


def test_missing_price_is_unknown_not_zero():
    model, _ = adapter(reply({"text": "hi"}), model=UNPRICED)
    r = model.complete([{"role": "user", "content": "x"}])
    assert r.usage.cost_usd(r.model) is None  # PRICES has the id but no numbers yet
    tracker = BudgetTracker(Budget())
    tracker.charge(r.usage, r.usage.cost_usd(r.model))
    tracker.charge(Usage(1_000_000, 0), Usage(1_000_000, 0).cost_usd("claude-opus-5-5"))
    assert tracker.unpriced_calls == 1 and tracker.cost_text() == "cost unknown (1 unpriced call(s); $4.00 priced)"


def test_run_result_reports_unknown_cost(tmp_path):
    session = Session.flat(33, 33, base_y=64)
    model, _ = adapter(reply({"text": "nothing to do"}), model=UNPRICED)
    result = run_task(prompt="a hill", images=[], session=session, tools={}, model=model,
                      tracer=Tracer(run_id="t", root=tmp_path), budget=Budget(), use_spec=False)
    assert result.cost_usd is None and "cost unknown" in str(result)


def test_the_eval_critic_is_priced():
    model, _ = adapter(reply({"text": "hi"}))
    u = model.complete([{"role": "user", "content": "x"}]).usage
    # gemini-3.8-flash through 2026-12-31: input 0.75, output (incl. thinking) 3.75, cache read 0.075 per 1M.
    assert u.cost_usd(MODEL) == pytest.approx((u.input_tokens * 0.75 + u.output_tokens * 3.75 + u.cache_read_tokens * 0.075) / 1e6)
    assert u.cost_usd(MODEL) > 0


def test_provider_inference_and_factory(monkeypatch):
    assert ModelConfig(model="gemini-3.8-flash").provider == "gemini"
    assert ModelConfig(model="claude-haiku-5-5").provider == "anthropic"
    assert ModelConfig(model="my-tuned-model", provider="gemini").provider == "gemini"
    with pytest.raises(ValueError, match="infer the provider"):
        ModelConfig(model="gpt-x")
    import anthropic
    from google import genai

    # Stub SDK clients: the factory's dispatch is under test; the conftest guard fails real clients.
    monkeypatch.setattr(anthropic, "Anthropic", lambda: "anthropic client")
    monkeypatch.setattr(genai, "Client", lambda: "gemini client")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert isinstance(make_adapter(ModelConfig(model=MODEL)), GeminiAdapter)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    assert isinstance(make_adapter(ModelConfig(model="claude-haiku-5-5")), AnthropicAdapter)


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiAdapter(ModelConfig(model=MODEL))


def test_a_malformed_call_goes_back_to_the_model_and_the_run_continues(tmp_path):
    """gemini-3.5-flash-lite smoke run (2026-10-30) crashed on MALFORMED_FUNCTION_CALL. Now the builder is told
    and re-issues; the feedback joins the previous user turn because the malformed reply has no parts."""
    session = Session.flat(33, 33, base_y=64)
    model, client = adapter(reply(finish="MALFORMED_FUNCTION_CALL", finishMessage="Malformed function call: raise_peak(("),
                            reply(call("raise_peak", PEAK)), reply({"text": "Done."}))
    result = run_task(prompt="a hill", images=[], session=session, tools={t.name: t for t in op_tools(session)},
                      model=model, tracer=Tracer(run_id="t", root=tmp_path), budget=Budget(), use_spec=False)
    assert result.stop == "done" and [e.id for e in session.ops] == ["raise_peak_1"]
    contents = client.calls[1]["contents"]
    assert [c.role for c in contents] == ["user"]  # opening + feedback merged; the empty reply is skipped
    assert "was malformed and was not run (Malformed function call: raise_peak(()" in contents[0].parts[-1].text
    steps = [json.loads(l) for l in (tmp_path / "t.jsonl").read_text(encoding="utf-8").splitlines()]
    assert steps[0]["stop_reason"] == "malformed_tool_call" and "asked to re-issue" in steps[0]["note"]


def test_two_malformed_calls_in_a_row_end_the_stage(tmp_path):
    bad = reply(finish="MALFORMED_FUNCTION_CALL")
    model, client = adapter(bad, bad)
    result = run_task(prompt="a hill", images=[], session=Session.flat(33, 33, base_y=64), tools={}, model=model,
                      tracer=Tracer(run_id="t", root=tmp_path), budget=Budget(), use_spec=False)
    assert result.stop == "malformed tool calls: 2 in a row" and len(client.calls) == 2


def test_a_failed_builder_call_is_traced_and_charged_before_it_propagates(tmp_path):
    model, _ = adapter(reply({"text": ""}, finish="OTHER"))
    with pytest.raises(GeminiStopError):
        run_task(prompt="a hill", images=[], session=Session.flat(33, 33, base_y=64), tools={}, model=model,
                 tracer=Tracer(run_id="t", root=tmp_path), budget=Budget(), use_spec=False)
    model2, _ = adapter(_api_error(400))
    with pytest.raises(errors.ClientError):
        run_task(prompt="a hill", images=[], session=Session.flat(33, 33, base_y=64), tools={}, model=model2,
                 tracer=Tracer(run_id="t2", root=tmp_path), budget=Budget(), use_spec=False)
    stopped = json.loads((tmp_path / "t.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert stopped["usage"]["output"] == 100 and stopped["note"].startswith("model call failed: GeminiStopError")
    rejected = json.loads((tmp_path / "t2.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert rejected["usage"] is None and rejected["note"].startswith("model call failed: ClientError")


def test_requests_have_a_timeout_and_a_timeout_is_retried_once():
    model, client = adapter(httpx.ReadTimeout("slow"), reply({"text": "hi"}))
    r = model.complete([{"role": "user", "content": "x"}])
    assert client.calls[0]["config"].http_options.timeout == 180_000  # ms
    assert [x["error"] for x in r.retries] == ["ReadTimeout"]

    model, client = adapter(httpx.ReadTimeout("slow"), httpx.ReadTimeout("slow"), reply({"text": "never"}))
    with pytest.raises(httpx.ReadTimeout):
        model.complete([{"role": "user", "content": "x"}])
    assert len(client.calls) == 2


def test_requests_run_on_a_worker_thread_so_ctrl_c_gets_through():
    threads = []
    model, client = adapter(reply({"text": "hi"}))
    inner = client.models.generate_content
    client.models.generate_content = lambda **kw: threads.append(threading.current_thread().name) or inner(**kw)
    assert model.complete([{"role": "user", "content": "x"}]).text == "hi"
    assert threads == ["gemini-request"]


def test_the_timeout_grows_with_the_output_limit():
    from agent.gemini import request_timeout

    assert request_timeout(ModelConfig(model=MODEL)) == 180.0  # 16K at 100 tokens/s = 160 s: the floor holds
    assert request_timeout(ModelConfig(model=MODEL, max_tokens=32_000)) == 320.0
    model, client = adapter(reply({"text": "hi"}), max_tokens=32_000)
    model.complete([{"role": "user", "content": "x"}])
    assert client.calls[0]["config"].http_options.timeout == 320_000  # ms
    assert GeminiAdapter(ModelConfig(model=MODEL), client=FakeClient(), timeout_s=5.0).timeout_s == 5.0
