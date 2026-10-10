import pytest

from agent.context import ContextManager, Episode, RollbackFeedback
from agent.critic import Critic, CriticError, CritiqueOutput, CritiqueResult
from agent.session import Session
from fakes import PEAK, ScriptedModel, critique_json, response, text

PNG = b"\x89PNG fake"


@pytest.fixture
def session():
    s = Session.flat(33, 33, base_y=64)
    s.apply("raise_peak", PEAK)
    return s


def _walk(node, path="$"):
    if isinstance(node, dict):
        yield path, node
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")


def test_critic_schema_fits_structured_output_limits():
    for path, node in _walk(CritiqueOutput.model_json_schema()):
        assert not {"minimum", "maximum", "minLength", "maxLength"} & node.keys(), path
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False, path


def test_blocking_check_skips_model_call():
    model = ScriptedModel()
    result, resp = Critic(model).critique({"prompt": "x"}, Session.flat(9, 9))
    assert resp is None and model.calls == [] and result.verdict == "revise" and result.score == 0


def test_spike_check_locates_spikes(session):
    session.state.height[10, 20] += 10
    checks = {c.name: c for c in Critic(ScriptedModel()).run_checks(session)}
    assert not checks["no_spikes"].passed and "u=0.62, v=0.31" in checks["no_spikes"].detail


def test_critique_parses_model_json(session):
    model = ScriptedModel(critique_json(4, "pass"))
    result, _ = Critic(model).critique({"prompt": "a hill"}, session)
    assert result.verdict == "pass" and result.score == 4 and result.issues[0].location == "raise_peak_1"
    assert model.calls[0]["response_schema"]["additionalProperties"] is False


def test_failed_major_check_overrides_pass(session):
    session.state.height[10, 20] += 10
    result, _ = Critic(ScriptedModel(critique_json(5, "pass"))).critique({"prompt": "x"}, session)
    assert result.verdict == "revise"


@pytest.mark.parametrize("bad", [response(text("not json")), response(text("{}"), stop="max_tokens")])
def test_unusable_critic_reply_fails_loudly(session, bad):
    with pytest.raises(CriticError):
        Critic(ScriptedModel(bad)).critique({"prompt": "x"}, session)


def test_episode_opening_message(session):
    ctx = ContextManager({"prompt": "a hill"}, area=(33, 33), references=[PNG])
    ctx.notes.append("keep it small")
    critique = CritiqueResult(verdict="revise", issues=[])
    rollback = RollbackFeedback(round=2, why="blind comparison preferred the accepted attempt",
                                diff=["changed raise_peak_1: height 15 -> 90"], problems=["[major] at raise_peak_1: a needle"])
    content = ctx.build_episode(session, "terrain", critique=critique, renders=[PNG], rollback=rollback).messages[0]["content"]
    assert [b["type"] for b in content] == ["image", "text", "image"]  # references, text, renders
    body = content[1]["text"]
    for fragment in ['"prompt": "a hill"', "raise_peak_1", "Verdict: revise", "Round 2 was not kept: blind comparison",
                     "height 15 -> 90", "a needle", "keep it small"]:
        assert fragment in body


def test_system_prompt_is_stable_and_has_area():
    ctx = ContextManager({"prompt": "x"}, area=(256, 128))
    assert ctx.system_prompt() == ctx.system_prompt() and "256 x 128 blocks" in ctx.system_prompt()


def test_note_tool_and_rotation():
    ctx = ContextManager({"prompt": "x"}, area=(9, 9), max_input_tokens=1000, max_messages=3)
    assert ctx.note_tool().handler({"note": " plan: one peak "}) == "saved note 1" and ctx.notes == ["plan: one peak"]
    assert ctx.should_rotate(Episode(messages=[{}]), 1000)
    assert ctx.should_rotate(Episode(messages=[{}] * 3), 10)
    assert not ctx.should_rotate(Episode(messages=[{}]), 10)
