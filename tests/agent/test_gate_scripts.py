import json

from agent.critic import AttemptView, Critic
from agent.tracing import Tracer
from fakes import ScriptedModel, compare_json
from scripts.gate_check import check_pair
from scripts.gate_stats import comparisons, summarize

VIEW = AttemptView(renders=[], stats="area 9x9 blocks")


def test_broken_copy_check_needs_the_intact_terrain_to_win_both_orders(tmp_path):
    # Order 1: intact is Terrain 1; order 2: intact is Terrain 2.
    passed, rows = check_pair(Critic(ScriptedModel(compare_json("1"), compare_json("2"))), {"prompt": "x"}, VIEW, VIEW,
                              tracer=Tracer(run_id="g", root=tmp_path))
    assert passed and [r["winner"] for r in rows] == ["intact", "intact"]
    passed, rows = check_pair(Critic(ScriptedModel(compare_json("1"), compare_json("1"))), {"prompt": "x"}, VIEW, VIEW)
    assert not passed and [r["winner"] for r in rows] == ["intact", "broken"]  # position bias, caught
    # gate_check records are not gate comparisons, so they don't skew the win rates.
    assert comparisons([tmp_path / "g.jsonl"]) == []


def _record(run_id, revision_is, better):
    order = {"1": "revision", "2": "previous"} if revision_is == "1" else {"1": "previous", "2": "revision"}
    winner = "same" if better == "same" else order[better]
    return {"run_id": run_id, "stage": "terrain/compare", "scores": {"order": order, "winner": winner, "better": better}}


def test_gate_stats_flags_a_biased_gate(tmp_path):
    path = tmp_path / "t.jsonl"
    biased = [_record("a", "1" if i % 2 else "2", "1" if i % 2 else "2") for i in range(20)]  # revision always wins
    path.write_text("\n".join(json.dumps(r) for r in biased))
    s = summarize(comparisons([path]))
    assert s["n"] == 20 and s["revision_rate"] == 1.0 and any("biased" in f for f in s["flags"])
    assert s["terrain1_rate"] == 0.5


def test_gate_stats_passes_a_balanced_gate():
    records = [_record("a", "1", b)["scores"] for b in ["1", "2", "same", "2"] * 5]
    s = summarize(records)
    assert s["n"] == 20 and s["revision_rate"] == 0.25 and s["flags"] == []
