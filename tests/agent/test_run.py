"""agent.run output: the trace header, final op stack and renders a run leaves behind (no network)."""

import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from agent.budget import Budget
from agent.critic import Critic
from agent.harness import run_task
from agent.run import RunConfig, l2_view, make_renderer, run_header, run_seed, save_final
from agent.session import Session
from agent.tools import control_tools, op_tools
from agent.tracing import Tracer, TraceRecord
from fakes import PEAK, ScriptedModel, compare_json, critique_json, response, text, tool_use
from terrain.opstack import OpStack


def _args(**kw):
    defaults = dict(prompt="a lone volcanic peak", model="claude-haiku-5-5", effort="low", rounds=2, render="l1")
    return RunConfig(**{**defaults, **kw})


def test_run_header_records_prompt_references_and_config(tmp_path):
    session = Session.flat(64, 48, base_y=70, seed=5)
    tools = {t.name: t for t in [*op_tools(session), *control_tools(session)]}
    ref = b"\x89PNG fake reference"
    header = run_header(_args(images=(Path("refs/peak.png"),)), [ref], session, Budget(max_steps=10), tools)

    assert header["prompt"] == "a lone volcanic peak"
    assert header["references"] == [{"path": "refs/peak.png", "sha256": hashlib.sha256(ref).hexdigest()}]
    assert header["builder_model"] == header["critic_model"] == "claude-haiku-5-5"
    assert header["canvas"] == {"width": 64, "depth": 48, "base_y": 70, "seed": 5}
    assert header["budget"]["max_steps"] == 10 and header["max_rounds"] == 2 and header["compare_both_orders"] is False and header["spec"] == "model"

    # The tool hash changes with the schemas the model sees, and the header round-trips through a trace.
    fewer = {k: v for k, v in tools.items() if k != "remove_op"}
    assert run_header(_args(), [], session, Budget(), fewer)["tools_sha256"] != header["tools_sha256"]
    tracer = Tracer(run_id="t", root=tmp_path)
    tracer.log(TraceRecord(run_id="t", stage="run", step=0, header=header))
    assert json.loads(tracer.path.read_text(encoding="utf-8"))["header"] == header


def test_run_seed_comes_from_the_flag_or_the_run_id():
    assert run_seed(7, "20261029-120000-abcdef") == 7 and run_seed(0, "x") == 0
    derived = run_seed(None, "20261029-120000-abcdef")
    assert derived == run_seed(None, "20261029-120000-abcdef")  # reproducible from the trace's run id
    assert len({run_seed(None, f"20261029-120000-{i:06x}") for i in range(10)}) == 10
    assert 0 <= derived < 2**31
    session = Session.flat(16, 16, seed=derived)
    assert run_header(_args(), [], session, Budget(), {})["canvas"]["seed"] == derived


def test_run_header_without_critic():
    session = Session.flat(16, 16)
    assert run_header(_args(critic=False, critic_model="x"), [], session, Budget(), {})["critic_model"] is None


def test_run_header_records_provider_and_model_per_role():
    session = Session.flat(16, 16)
    models = run_header(_args(critic_model="gemini-3.8-flash"), [], session, Budget(), {})["models"]
    builder, critic = {"provider": "anthropic", "model": "claude-haiku-5-5"}, {"provider": "gemini", "model": "gemini-3.8-flash"}
    assert models == {"builder": builder, "spec": builder, "critic": critic, "comparison": critic}
    models = run_header(_args(critic=False, use_spec=False), [], session, Budget(), {})["models"]
    assert models["spec"] is None and models["critic"] is None and models["comparison"] is None

PNG = b"\x89PNG"


@pytest.mark.parametrize("references, camera", [([b"\x89PNG reference photo"], "overview_s"), ([], "overview_sw")])
def test_the_critic_and_comparison_see_the_reference_camera(tmp_path, monkeypatch, references, camera):
    import terrain.render.l2 as l2

    monkeypatch.setattr(l2, "render_volume", lambda volume, view, title="", **kw: PNG + b" l2 " + view.encode())
    session = Session.flat(33, 33, base_y=64)
    builder = ScriptedModel(response(tool_use("raise_peak", PEAK), stop="tool_use"), response(text("r1")),
                            response(tool_use("raise_peak", PEAK | {"center": [0.25, 0.25], "radius": 0.2, "height": 10}, id="tu_2"),
                                     stop="tool_use"), response(text("r2")))
    critic_model = ScriptedModel(critique_json(3), critique_json(3), compare_json("same"))
    run_task(prompt="like this photo", images=references, session=session, tools={t.name: t for t in op_tools(session)},
             model=builder, critic=Critic(critic_model), tracer=Tracer(run_id="t", root=tmp_path), budget=Budget(),
             render=make_renderer("l2", tmp_path / "r", view=l2_view(references)), max_rounds=2, use_spec=False)

    def l2_views(call):
        data = [base64.b64decode(b["source"]["data"]) for b in call["messages"][0]["content"] if b["type"] == "image"]
        return {d.split(b" l2 ")[1].decode() for d in data if b" l2 " in d}

    assert [l2_views(c) for c in critic_model.calls] == [{camera}] * 3  # both critiques and the comparison
    assert l2_views(builder.calls[2]) == {camera}  # round 2's opening render too


def test_save_final_writes_replayable_stack_and_renders(tmp_path):
    session = Session.flat(33, 33, base_y=64, seed=3)
    session.apply("raise_peak", {"center": [0.5, 0.5], "radius": 0.25, "height": 40}, id="peak")
    out = tmp_path / "run_1"

    written = save_final(session, out)

    assert [p.name for p in written] == ["stack.json", "final_l1.png", "final_l2_overview_ne.png", "final_l2_overview_sw.png"]
    assert all(p.read_bytes()[:4] == PNG for p in written[1:])
    replayed = OpStack.from_json((out / "stack.json").read_text(encoding="utf-8"))
    assert [e.id for e in replayed.entries] == ["peak"] and replayed.base.seed == 3
    np.testing.assert_array_equal(replayed.state.height, session.state.height)


def test_save_final_keeps_stack_when_l2_fails(tmp_path, capsys):
    session = Session.flat(17, 17)
    written = save_final(session, tmp_path, l2_views=("not_a_view",))
    assert [p.name for p in written] == ["stack.json", "final_l1.png"]
    assert "final L2 render failed" in capsys.readouterr().out
