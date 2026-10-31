"""Eval judge (eval/judge.py): parsing, retries, the pass rule, what it sees, and the cache. Fake adapters only."""

import json

from agent.session import Session
from eval import judge as judging
from eval.prompts import JUDGE_SYSTEM
from agent.prompts import CRITIC_SYSTEM
from tests.agent.fakes import PEAK, ScriptedModel, response, text


def reply(verdict="pass", ref=None, **scores):
    body = {"reasons": ["clean slopes"], "scores": {"naturalism": 4, "minecraft_quality": 4, "materials": 3,
                                                    "reference_match": ref} | scores, "verdict": verdict}
    return response(text(json.dumps(body)))


def fake_render(stack, views):
    return [(v, b"\x89PNG " + v.encode()) for v in views]


def stack():
    s = Session.flat(33, 33, base_y=64, seed=1)
    s.apply("raise_peak", PEAK, id="raise_peak_1")
    return s.stack


def texts(call):
    return " ".join(b["text"] for b in call["messages"][0]["content"] if b["type"] == "text")


def test_a_verdict_is_parsed_and_cached(tmp_path):
    model = ScriptedModel(reply())
    rec = judging.judge(model, stack(), "a peak", [], cache=tmp_path, render=fake_render, save_renders=tmp_path / "run")
    assert rec["passed"] is True and rec["mean"] == 11 / 3 and rec["failed"] is None and rec["cached"] is False
    assert sorted(p.name for p in (tmp_path / "run").iterdir()) == ["judge_overview_ne.png", "judge_overview_sw.png"]
    # Same stack, request and model: served from the cache, no call (the script is empty now).
    again = judging.judge(model, stack(), "a peak", [], cache=tmp_path, render=fake_render)
    assert again["cached"] is True and again["cost_usd"] == 0.0 and again["output"] == rec["output"]
    # A different request or model is a different key.
    assert judging.cache_key(stack(), "a peak", [], "m") != judging.cache_key(stack(), "two peaks", [], "m")
    assert judging.cache_key(stack(), "a peak", [], "m") != judging.cache_key(stack(), "a peak", [], "n")


def test_the_pass_rule_needs_every_score_at_least_3(tmp_path):
    rec = judging.judge(ScriptedModel(reply(materials=2)), stack(), "a peak", [], cache=None, render=fake_render)
    assert rec["passed"] is False  # the model said pass, but materials is 2
    rec = judging.judge(ScriptedModel(reply("fail")), stack(), "a peak", [], cache=None, render=fake_render)
    assert rec["passed"] is False


def test_out_of_range_scores_are_retried_once_then_recorded_as_failed(tmp_path):
    model = ScriptedModel(reply(naturalism=6), reply())
    rec = judging.judge(model, stack(), "a peak", [], cache=tmp_path, render=fake_render)
    assert rec["passed"] is True and len(model.calls) == 2 and "outside 1-5" in rec["errors"][0]
    model = ScriptedModel(response(text("not json")), reply(ref=4))  # reference_match without references
    rec = judging.judge(model, stack(), "another peak", [], cache=tmp_path, render=fake_render)
    assert rec["passed"] is None and rec["output"] is None and "must be null" in rec["failed"]
    assert len(list(tmp_path.glob("*.json"))) == 1  # failures are not cached, so a rescore retries them


def test_the_judge_sees_request_references_and_neutral_renders_only(tmp_path):
    model = ScriptedModel(reply(ref=3))
    ref = b"\x89PNG reference photo"
    judging.judge(model, stack(), "like this photo", [ref], cache=None, render=fake_render)
    call = model.calls[0]
    content = call["messages"][0]["content"]
    assert [b["type"] for b in content][:2] == ["image", "text"]  # the reference first, then the request
    assert "Render: overview_s" in texts(call)  # the photo's viewpoint, right after the request
    words = texts(call)
    for leak in ("raise_peak", "critic_on", "critic_off", "spec_off", "ops", "checklist"):
        assert leak not in words
    assert call["system"] == JUDGE_SYSTEM and call["system"] != CRITIC_SYSTEM


def test_the_cost_estimate_uses_the_judge_price():
    assert judging.call_estimate("gemini-3.1-flash-lite", 0) > 0
    assert judging.call_estimate("gemini-3.1-flash-lite", 1) > judging.call_estimate("gemini-3.1-flash-lite", 0)
    assert judging.call_estimate("gemini-3.8-flash", 0) is None  # unpriced: unknown, never $0
