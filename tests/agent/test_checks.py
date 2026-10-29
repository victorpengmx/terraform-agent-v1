"""Programmatic checks (DESIGN §11.2) on small synthetic terrains: each one passes on sane ground and fails,
with a location, on the artifact it targets."""

import numpy as np
import pytest

from agent.checks import CheckResult, run_checks
from agent.critic import Critic, CritiqueResult
from agent.session import Session
from fakes import PEAK, ScriptedModel


def checks(session) -> dict[str, CheckResult]:
    return {c.name: c for c in run_checks(session)}


@pytest.fixture
def hill():
    s = Session.flat(65, 65, base_y=64)
    s.apply("raise_peak", PEAK)
    return s


def test_a_sane_hill_passes_everything(hill):
    assert [c.name for c in run_checks(hill) if not c.passed] == []


def test_a_barely_changed_area_is_blocking_and_skips_the_model():
    s = Session.flat(65, 65, base_y=64)
    s.apply("raise_peak", {"center": [0.5, 0.5], "radius": 0.02, "height": 3})
    c = checks(s)["meaningful_change"]
    assert not c.passed and c.severity == "blocking"
    model = ScriptedModel()
    result, response = Critic(model).critique({"prompt": "x"}, s)
    assert response is None and model.calls == [] and result.verdict == "revise"


def test_sheer_terrain_fails_steep_share(hill):
    hill.state.height[:, ::2] += 20  # 20-block walls between every pair of columns
    c = checks(hill)["steep_share"]
    assert not c.passed and c.severity == "major" and "e.g. at u=" in c.detail


def test_thin_wall_fails_thin_features():
    s = Session.flat(65, 65, base_y=64)
    s.apply("raise_region", {"region": {"polygon": [[0.2, 0.5], [0.8, 0.5], [0.8, 0.52], [0.2, 0.52]]}, "height": 20, "edge": 0})
    c = checks(s)["thin_features"]
    assert not c.passed and "v=0.5" in c.detail


def test_water_too_shallow_to_survive_rounding_fails_containment(hill):
    s = hill.state
    s.water[5:15, 5:15] = s.height[5:15, 5:15] + 0.3  # under half a block: rounds away
    c = checks(hill)["water_containment"]
    assert not c.passed and "100 of 100" in c.detail


def test_gentle_steps_is_informational(hill):
    hill.state.height[5, 5] += 1  # a 1-block bump on the flat ground around the hill
    c = checks(hill)["gentle_steps"]
    assert c.passed and c.severity == "info" and "u=0.08, v=0.08" in c.detail


def test_regressions_are_new_blocking_or_major_failures():
    before = CritiqueResult(verdict="revise", checks=[
        CheckResult(name="no_spikes", passed=True, severity="major"),
        CheckResult(name="steep_share", passed=False, severity="major"),
        CheckResult(name="gentle_steps", passed=True, severity="info")])
    after = CritiqueResult(verdict="revise", checks=[
        CheckResult(name="no_spikes", passed=False, severity="major"),
        CheckResult(name="steep_share", passed=False, severity="major"),
        CheckResult(name="gentle_steps", passed=False, severity="info")])
    assert [c.name for c in after.regressions(before)] == ["no_spikes"]
