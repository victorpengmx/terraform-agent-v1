"""Success rule (eval/scoring.py), paired statistics (eval/stats.py) and eval configs (eval/config.py)."""

import pytest

from eval import stats
from eval.benchmark import load
from eval.config import EvalConfig, load_config, same_family, shared_family
from eval.scoring import answer_key, checks, success
from scripts.demos import build


def test_success_needs_all_three_parts():
    ok_key, bad_key = {"major_failures": []}, {"major_failures": ["peak_1.exists"]}
    ok_checks, bad_checks = {"failed": []}, {"failed": ["no_spikes"]}
    passed, failed, unknown = {"passed": True}, {"passed": False}, {"passed": None}
    assert success(ok_key, ok_checks, passed) == {"checklist": True, "checks": True, "judge": True, "overall": True}
    assert success(bad_key, ok_checks, passed)["overall"] is False
    assert success(ok_key, bad_checks, passed)["overall"] is False
    assert success(ok_key, ok_checks, failed)["overall"] is False
    # No usable judge verdict: overall unknown, never a failure; the other parts are still stored.
    assert success(bad_key, ok_checks, unknown) == {"checklist": False, "checks": True, "judge": None, "overall": None}
    assert success(ok_key, ok_checks, None)["overall"] is None


def test_answer_key_and_checks_on_a_fitting_demo():
    t01 = {e.id: e for e in load()}["t01"]
    stack = build("valley", seed=1)
    a = answer_key(t01.graph, stack)
    assert a["pass_rate"] == 1.0 and a["major_failures"] == [] and a["passed"] == a["total"] == 11
    assert a["matched"]["river_1"] == ["river"]
    assert checks(stack)["failed"] == []


def test_sign_test_known_answers():
    assert stats.sign_test(9, 1) == pytest.approx(0.021484375)  # 2 * (1 + 10) / 1024
    assert stats.sign_test(1, 9) == pytest.approx(0.021484375)
    assert stats.sign_test(0, 0) == 1.0
    assert stats.sign_test(5, 5) == 1.0
    assert stats.sign_test(6, 0) == pytest.approx(2 / 64)


def test_bootstrap_ci_is_exact_for_constant_differences_and_reproducible():
    assert stats.bootstrap_ci({"a": [0.2, 0.2], "b": [0.2], "c": [0.2]}) == pytest.approx((0.2, 0.2, 0.2))
    diffs = {f"p{i}": [(-1) ** i * 0.1 * i, 0.05] for i in range(10)}
    assert stats.bootstrap_ci(diffs) == stats.bootstrap_ci(diffs)  # fixed RNG seed
    m, lo, hi = stats.bootstrap_ci(diffs)
    assert lo < m < hi
    assert stats.bootstrap_ci({}) is None


def test_bootstrap_resamples_prompts_not_seeds():
    """Each prompt counts once: its seeds are averaged first."""
    m, _, _ = stats.bootstrap_ci({"a": [1.0, 1.0, 1.0, 1.0], "b": [0.0]})
    assert m == pytest.approx(0.5)


def test_labels_are_plain_language():
    assert stats.label((0.1, -0.05, 0.2), "critic_on", "critic_off") == "no clear difference at this N"
    assert stats.label((0.3, 0.1, 0.5), "critic_on", "critic_off") == "critic_on higher"
    assert stats.label((-0.3, -0.5, -0.1), "critic_on", "critic_off") == "critic_off higher"
    assert stats.percentile([5, 1, 3, 2, 4], 50) == 3 and stats.percentile([1, 2, 3, 4], 95) == 4


def test_configs_select_prompts_and_override_budgets():
    entries = load()
    e1, e2 = load_config("eval/configs/e1_critic.yaml"), load_config("eval/configs/e2_spec.yaml")
    assert len(e1.select(entries)) * len(e1.seeds) * len(e1.conditions) == 60
    assert len(e2.select(entries)) * len(e2.seeds) * len(e2.conditions) == 20
    assert {e.id for e in e2.select(entries)} == {e.id for e in entries if e.spec_ablation}
    assert e1.budget_for("critic_off").max_cost_usd == 0.25 and e1.budget_for("critic_on").max_cost_usd == 1.00
    assert e1.budget_for("critic_off").max_steps == 40
    with pytest.raises(ValueError, match="unknown prompt ids"):
        EvalConfig(name="x", prompts=["t99"], conditions={"a": {"critic": True}}).select(entries)


def test_a_mistyped_config_key_is_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("name: x\nconditions: {a: {critic: true, round: 3}}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="round"):
        load_config(path)


def test_judge_family():
    assert same_family("claude-opus-5-5", "claude-haiku-5-5", None)
    assert not same_family("gemini-3.1-flash-lite", "claude-haiku-5-5", "claude-opus-5-5")
    assert shared_family({"builder": "claude-haiku-5-5", "critic": "gemini-3.8-flash", "judge": "claude-opus-5-5"}) == ["builder"]
    assert shared_family({"builder": "claude-opus-5-5", "critic": "claude-opus-5-5", "judge": "gemini-3.8-flash"}) == []
    assert shared_family({"builder": "claude-haiku-5-5", "critic": None, "judge": "claude-opus-5-5"}) == ["builder"]


def test_the_eval_configs_use_the_eval_models():
    for name in ("e1_critic", "e2_spec", "pilot", "regression"):
        m = load_config(f"eval/configs/{name}.yaml").models
        assert (m.builder, m.critic, m.judge) == ("claude-haiku-5-5", "gemini-3.8-flash", "claude-opus-5-5"), name
    m = load_config("eval/configs/opus.yaml").models
    assert (m.builder, m.critic, m.judge) == ("claude-opus-5-5", "claude-opus-5-5", "gemini-3.8-flash")
    haiku, pilot = load_config("eval/configs/pilot_haiku.yaml"), load_config("eval/configs/pilot.yaml")
    assert {haiku.models.builder, haiku.models.critic, haiku.judge_model} == {"claude-haiku-5-5"}
    assert (haiku.prompts, haiku.seeds, list(haiku.conditions)) == (pilot.prompts, pilot.seeds, list(pilot.conditions))


def test_a_judge_equal_to_the_critic_is_rejected_unless_allowed():
    cfg = {"name": "x", "models": {"critic": "claude-opus-5-5", "judge": "claude-opus-5-5"}, "conditions": {"a": {"critic": True}}}
    with pytest.raises(ValueError, match="judge model \\(claude-opus-5-5\\) is the critic model"):
        EvalConfig.model_validate(cfg)
    assert EvalConfig.model_validate(cfg | {"judge_same_as_critic_allowed": True}).judge_model == "claude-opus-5-5"
