"""Every text a model sees, pinned by hash (refactor guard; prompt caching needs byte-stable prompts anyway).

The hashes were taken before the 2026-10-30 refactor. Values are reached only through public entry points
(Critic, write_spec, ContextManager, agent.tools), so moving constants between modules can't change this
file. A failure means a prompt, schema or model-facing text changed: if that was intended, it is a prompt
change (EVIDENCE row), and the hash is updated deliberately.
"""

import hashlib
import json

import pytest

from agent.context import ContextManager, RollbackFeedback
from agent.critic import AttemptView, Critic
from agent.session import Session, terrain_stats
from agent.spec import FeatureGraph, write_spec
from agent.tools import control_tools, execute, observe_tools, op_tools
from tests.agent.fakes import ScriptedModel, compare_json, critique_json, feature, graph, spec_json, tool_use

EXPECTED = {
    "builder_tools": "64c908f45c1e4765579ea83a6114ae3d5514f44ba9118e4aefd2a77cf83f8cd5",
    "write_note_tool": "2a2f56a7e9dadf48cecb16846bafde8bdc6fe82eaaee271ec3ae4fd706de272a",
    "builder_system": "bb8ba282ea8eab63d8f469c96da635f26f7e912aab90433063a2f08bb521527f",
    "critic_system": "b148a1283cc095e4d3de24584e852e8225aacd9351dbbe1dcba6197c8d129bc6",
    "tool_catalogue": "25cbe072f42c0ed050a8a29815815bcff8c53e8d45867b608445f0e1a5d701cb",
    "compare_system": "13927ae89cc4ef6e1dea0e6476a9c4801f94bc7f749bf0737b9dc4c3876c1a87",
    "spec_system": "d626c2bdb61c939e037c86a1a8c721b5693a09275e110e5a4693a3098b67224e",
    "critique_schema": "46ff595c1900b0aa2c93221939eab0481c3557b8b7607215353180f552b333b4",
    "compare_schema": "5fb85951ebd48304cb6cf415a7903ba4e42ce24cd4232670f72e500e528ca1fb",
    "spec_schema": "f8bf814d0a4578ee75eadfa9254cfdd4bcef0c93b84853175bef343443b95ecf",
    "tool_results": "82b89cb632aec2fda8d860b9e737fe575befe0164a10943c69fa5e605a90a45d",
    "critique_request": "b8fd797b145b39ef9b69845d65b8bfab245a022bf52872ab10477f8efe772546",
    "compare_request": "dd569fa203892b0119ec1b2936660af3171e5720d0fdea7a5ab46c89cd1e1afc",
    "spec_request": "6fcb71165dedd12d59062a0757ccfc506589918173d4898287875d048ec2bb7b",
    "episode_opening": "081fee91fe03ea3bbb8f464ab8d3ccae3e6ab1e0f2e0345aac7a65c193b6873f",
}


def _sha(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def current(tmp_path) -> dict[str, str]:
    s = Session.flat(65, 65, base_y=64, seed=3)
    tools = {t.name: t for t in [*op_tools(s), *control_tools(s), *observe_tools(s, tmp_path)]}
    g = FeatureGraph.model_validate(graph(feature("peak_1", x="left"), feature("river_1", "river"),
                                          relations=[{"subject": "river_1", "relation": "reaches", "object": "south_edge"}]))
    spec = {"prompt": "a peak and a river", "graph": g}
    results = [execute(tools, tool_use(op, params, id=f"t{i}"))["content"] for i, (op, params) in enumerate([
        ("raise_peak", {"center": [0.2, 0.3], "radius": 0.15, "height": 15, "feature": "peak_1"}),
        ("carve_river", {"path": [[0.6, 0.1], [0.6, 1.0]], "width": 0.06, "depth": 3, "feature": "river_1"}),
        ("list_ops", {}),
    ])]

    critic_model = ScriptedModel(critique_json(3))
    critique, _ = Critic(critic_model).critique(spec, s)
    compare_model = ScriptedModel(compare_json("same"))
    Critic(compare_model).compare(spec, AttemptView([], terrain_stats(s.state)), AttemptView([], "area 65x65 blocks"))
    spec_model = ScriptedModel(spec_json(graph(feature("peak_1"))))
    write_spec(spec_model, "a peak and a river", images=[b"\x89PNG fake"])

    context = ContextManager(spec, area=(65, 65))
    episode = context.build_episode(s, "terrain", critique=critique,
                                    rollback=RollbackFeedback(round=2, why="why", diff=["added x"], problems=["[major] p"]))
    full, bare = Critic(ScriptedModel()).system, Critic(ScriptedModel(), catalogue=False).system
    assert full.startswith(bare + "\n\n")
    calls = {"critique": critic_model.calls[0], "compare": compare_model.calls[0], "spec": spec_model.calls[0]}
    return {
        "builder_tools": _sha([t.to_param() for t in tools.values()]),  # as run_header's tools_sha256
        "write_note_tool": _sha(context.note_tool().to_param()),
        "builder_system": _sha(context.system_prompt()),
        "critic_system": _sha(full),
        "tool_catalogue": _sha(full[len(bare) + 2:]),
        "compare_system": _sha(calls["compare"]["system"]),
        "spec_system": _sha(calls["spec"]["system"]),
        **{f"{k}_schema": _sha(c["response_schema"]) for k, c in calls.items()},
        "tool_results": _sha(results),
        **{f"{k}_request": _sha(c["messages"]) for k, c in calls.items()},
        "episode_opening": _sha(episode.messages),
    }


@pytest.fixture(scope="module")
def hashes(tmp_path_factory):
    return current(tmp_path_factory.mktemp("views"))


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_model_facing_text_is_unchanged(hashes, key):
    assert hashes[key] == EXPECTED[key], f"{key} changed: a prompt, schema or model-facing text differs"
