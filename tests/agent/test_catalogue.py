"""Tool catalogue in the critic prompt (DESIGN §11.5) and the schema check for structured fixes."""

import json

import pytest

from agent.critic import Change, Critic, Issue, hint_problems, tool_catalogue
from agent.session import Session
from fakes import PEAK, ScriptedModel, critique_json, response, text, tool_use
from terrain.ops import list_ops


@pytest.fixture
def session():
    s = Session.flat(33, 33, base_y=64)
    s.apply("raise_peak", PEAK, id="mountain_1")
    s.apply("carve_valley", {"path": [[0.1, 0.1], [0.9, 0.9]], "width": 0.2, "depth": 5}, id="valley")
    return s


def test_catalogue_lists_every_tool_once_with_values_and_ranges():
    text = tool_catalogue()
    lines = [l for l in text.splitlines() if l.startswith("- ")]
    assert [l[2:].split("(")[0] for l in lines] == [s.name for s in list_ops()] + ["update_op", "remove_op"]
    valley = next(l for l in lines if l.startswith("- carve_valley("))
    assert "profile=v|u" in valley and "width [0..1]" in valley and "character=gentle|rolling|rugged|jagged" in valley
    assert "raise_region(region, height [-256..256], edge [0..0.5], mode=add|plateau|basin)" in text
    assert valley.endswith("lowering the terrain (never raising it).")  # first sentence only


def test_catalogue_is_in_the_critic_prompt_and_byte_stable(session):
    model = ScriptedModel(critique_json(4))
    critic = Critic(model)
    assert critic.system == Critic(ScriptedModel()).system and tool_catalogue() in critic.system
    assert tool_catalogue() not in Critic(ScriptedModel(), catalogue=False).system
    critic.critique({"prompt": "a hill"}, session)
    assert model.calls[0]["system"] == critic.system


def issue(tool, location="0.5,0.5", op_id=None, **changes):
    if op_id is None:
        op_id = "mountain_1" if tool in ("update_op", "remove_op") else ""
    return Issue(location=location, severity="major", description="d", tool=tool, op_id=op_id, fix_hint="f",
                 changes=[Change(param=k, value=v) for k, v in changes.items()])


@pytest.mark.parametrize("fix, problem", [
    (issue("update_op", roughness="0.8", height="30"), None),
    (issue("carve_valley", location="0.5,0.5", profile="u", width="0.3"), None),  # bare words are values too
    (issue("none"), None),
    (issue("remove_op"), None),
    (issue("update_op", irregularity="1.2"), "less than or equal to 1"),
    (issue("raise_region", location="0.5,0.5", mode='"ridge"'), "'add', 'plateau' or 'basin'"),
    (issue("update_op", width="0.3"), "raise_peak has no parameter 'width'"),
    (issue("sculpt_3d", location="0.5,0.5"), "unknown tool 'sculpt_3d'"),
    (issue("update_op", op_id="mountain_9", height="30"), "no op with id 'mountain_9'"),
    (issue("update_op", op_id="0.5,0.5", height="30"), "no op with id '0.5,0.5'"),  # a position is not an op id
    (issue("remove_op", op_id="mountain_1, valley"), "no op with id 'mountain_1, valley'"),
])
def test_hint_problems_checks_tools_parameters_and_values(session, fix, problem):
    problems = hint_problems(fix, session.stack)
    assert (problems == []) if problem is None else any(problem in p for p in problems), problems


def test_the_builder_sees_the_structured_fix():
    from agent.critic import CritiqueResult

    text = CritiqueResult(verdict="revise", issues=[issue("update_op", height="120"), issue("none")]).as_text()
    assert "(update_op mountain_1: height=120)" in text and "(no tool can do this yet)" in text


def test_invalid_fixes_are_flagged_and_hidden_from_the_builder(session):
    issues = [issue("update_op", width="0.3"), issue("update_op", irregularity="1.2"),
              issue("update_op", op_id="mountain_9", height="30"), issue("update_op", roughness="0.8", height="30"),
              issue("none")]
    issues = [i.model_copy(update={"description": f"problem {n}"}) for n, i in enumerate(issues)]
    body = {"issues": [i.model_dump() for i in issues], "verdict": "revise",
            "scores": {k: 3 for k in ["prompt_adherence", "layout", "naturalism", "minecraft_quality", "materials"]} | {"reference_match": None}}
    result, _ = Critic(ScriptedModel(response(text(json.dumps(body))))).critique({"prompt": "a hill"}, session)

    assert [bool(p) for p in result.fix_problems] == [True, True, True, False, False]
    assert "raise_peak has no parameter 'width'" in result.fix_problems[0][0]
    assert "less than or equal to 1" in result.fix_problems[1][0]
    assert "no op with id 'mountain_9'" in result.fix_problems[2][0]
    assert result.fix_stats() == {"issues": 5, "with_fix": 4, "valid": 1, "none": 1}

    lines = {l.split(": ", 1)[1].split(" Fix")[0]: l for l in result.as_text().splitlines() if l.startswith("- [major] at")}
    for n in range(3):  # description kept, bad parameters replaced by the reason
        assert "(fix not applicable as written: " in lines[f"problem {n}"] and "(update_op " not in lines[f"problem {n}"]
    assert lines["problem 3"].endswith("Fix: f (update_op mountain_1: roughness=0.8, height=30)")  # valid fix unchanged
    assert lines["problem 4"].endswith("(no tool can do this yet)")


def test_the_trace_records_fix_validity_per_critique(session, tmp_path):
    from agent.budget import Budget
    from agent.harness import run_task
    from agent.tools import op_tools
    from agent.tracing import Tracer

    builder = ScriptedModel(response(tool_use("raise_peak", PEAK), stop="tool_use"), response(text("done")))
    s = Session.flat(33, 33, base_y=64)
    run_task(prompt="a hill", images=[], session=s, tools={t.name: t for t in op_tools(s)}, model=builder,
             critic=Critic(ScriptedModel(critique_json(3))), tracer=Tracer(run_id="t", root=tmp_path), budget=Budget(),
             max_rounds=1, use_spec=False)
    record = next(json.loads(l) for l in (tmp_path / "t.jsonl").read_text().splitlines() if '"terrain/critique"' in l)
    assert record["scores"]["fixes"] == {"issues": 1, "with_fix": 1, "valid": 1, "none": 0}
    assert record["scores"]["fix_problems"] == [[]]


def test_missing_tools_are_ranked_from_traces(tmp_path):
    import json

    from scripts.missing_tools import missing

    record = {"stage": "terrain/critique", "scores": {"issues": [
        {"tool": "none", "fix_hint": "a ridge tool"}, {"tool": "none", "fix_hint": "a ridge tool"},
        {"tool": "none", "fix_hint": "river confluences"}, {"tool": "update_op", "fix_hint": "x"}]}}
    (tmp_path / "t.jsonl").write_text(json.dumps(record))
    assert missing([tmp_path / "t.jsonl"]).most_common() == [("a ridge tool", 2), ("river confluences", 1)]
