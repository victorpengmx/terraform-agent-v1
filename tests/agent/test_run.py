"""agent.run output: the trace header, final op stack and renders a run leaves behind (no network)."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from agent.budget import Budget
from agent.run import run_header, save_final
from agent.session import Session
from agent.tools import control_tools, op_tools
from agent.tracing import Tracer, TraceRecord
from terrain.opstack import OpStack


def _args(**kw):
    defaults = dict(prompt="a lone volcanic peak", image=[], model="claude-haiku-5-5", effort="low", critic_model=None,
                    no_critic=False, rounds=2, render="l1")
    return argparse.Namespace(**{**defaults, **kw})


def test_run_header_records_prompt_references_and_config(tmp_path):
    session = Session.flat(64, 48, base_y=70, seed=5)
    tools = {t.name: t for t in [*op_tools(session), *control_tools(session)]}
    ref = b"\x89PNG fake reference"
    header = run_header(_args(image=[Path("refs/peak.png")]), [ref], session, Budget(max_steps=10), tools)

    assert header["prompt"] == "a lone volcanic peak"
    assert header["references"] == [{"path": "refs/peak.png", "sha256": hashlib.sha256(ref).hexdigest()}]
    assert header["builder_model"] == header["critic_model"] == "claude-haiku-5-5"
    assert header["canvas"] == {"width": 64, "depth": 48, "base_y": 70, "seed": 5}
    assert header["budget"]["max_steps"] == 10 and header["max_rounds"] == 2

    # The tool hash changes with the schemas the model sees, and the header round-trips through a trace.
    fewer = {k: v for k, v in tools.items() if k != "remove_op"}
    assert run_header(_args(), [], session, Budget(), fewer)["tools_sha256"] != header["tools_sha256"]
    tracer = Tracer(run_id="t", root=tmp_path)
    tracer.log(TraceRecord(run_id="t", stage="run", step=0, header=header))
    assert json.loads(tracer.path.read_text(encoding="utf-8"))["header"] == header


def test_run_header_without_critic():
    session = Session.flat(16, 16)
    assert run_header(_args(no_critic=True, critic_model="x"), [], session, Budget(), {})["critic_model"] is None

PNG = b"\x89PNG"


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
