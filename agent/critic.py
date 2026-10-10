"""Critic (DESIGN §11): scores a stage result and returns located, actionable issues.

Two steps: cheap programmatic checks first; if none is blocking, one model call in a fresh context with
the spec, references, renders, terrain stats and check results. The critic never sees the builder's
conversation. It only judges; accept/rollback decisions stay in the harness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from scipy.ndimage import grey_opening

from agent.models import ModelAdapter, ModelResponse
from agent.session import Session
from agent.tools import image_block, text_block
from terrain.compiler.compile import surface_y, water_top
from terrain.features import feature_report, measure_features
from terrain.materials import surface_metrics
from terrain.state import WorldState

Score = Literal[1, 2, 3, 4, 5]  # enum, not min/max: structured outputs reject numeric constraints

# Programmatic check thresholds (DESIGN §11.2). Demos / baseline run vs. broken terrains, 2026-10-22:
MIN_CHANGED_SHARE = 0.01  # demos change 57-98% of columns; a 3-block bump changes 0%
SPIKE_BLOCKS = 3.0  # a column this much higher than all 4 neighbours reads as a 1-block spike in game
SHEER_DROP, MAX_SHEER_SHARE = 3.0, 0.25  # 3 blocks down to a neighbour ≈ 72°. Demos 0.2-10.4% (mesa cliffs); a wreck 81%
GENTLE_DEG = 15.0  # "gentle" slope class (terrain/features.py)
MAX_WATER_LOST = 0.25  # demos lose 0.7-8.6% (shallow lake fringes, a known hard case)
THIN_BLOCKS, MAX_THIN_SHARE = 4.0, 0.002  # demos 0.02-0.05% of columns; a 1.5-block wall 0.47%, the wreck 34%


class CriticError(Exception):
    """The critic model's reply was unusable (refusal, truncation, invalid JSON)."""

    def __init__(self, message: str, response: ModelResponse | None = None) -> None:
        super().__init__(message)
        self.response = response  # the unusable reply, so its tokens are still charged and traced


class CheckResult(BaseModel):
    """Output of one programmatic check."""

    name: str
    passed: bool
    severity: Literal["info", "major", "blocking"]  # blocking = skip the model call, must revise
    detail: str = ""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")  # additionalProperties: false, required by structured outputs


_T = TypeVar("_T", bound=_Strict)


class Issue(_Strict):
    location: str = Field(description="Op id (e.g. 'raise_peak_1') or normalized 'u,v' position")
    severity: Literal["minor", "major", "blocking"]
    description: str
    fix_hint: str = Field(description="Concrete change, ideally naming the tool and parameter")


class RubricScores(_Strict):
    prompt_adherence: Score
    layout: Score
    naturalism: Score
    minecraft_quality: Score
    reference_match: Score | None = Field(description="null when there is no reference image")


class CritiqueOutput(_Strict):
    """What the critic model returns. Issues come before scores so the model reasons before rating."""

    issues: list[Issue]
    scores: RubricScores
    verdict: Literal["pass", "revise"]


class CritiqueResult(BaseModel):
    checks: list[CheckResult] = []
    issues: list[Issue] = []
    scores: RubricScores | None = None  # None when a blocking check short-circuited the model call
    verdict: Literal["pass", "revise"]
    failed: str | None = None  # why the critic model gave no usable critique this round (checks still ran)

    def regressions(self, previous: CritiqueResult) -> list[CheckResult]:
        """Blocking or major checks that fail here but passed in `previous` (acceptance gate rule 1)."""
        passed_before = {c.name for c in previous.checks if c.passed}
        return [c for c in self.checks if not c.passed and c.severity in ("blocking", "major") and c.name in passed_before]

    @property
    def score(self) -> float:
        """Mean rubric score; 0 when blocked. Used for score-gated acceptance."""
        if self.scores is None:
            return 0.0
        vals = [v for v in self.scores.model_dump().values() if v is not None]
        return sum(vals) / len(vals)

    def as_text(self) -> str:
        lines = [f"Verdict: {self.verdict}. Score {self.score:.2f}/5."]
        if self.failed:
            lines = [f"The critic gave no usable critique this round ({self.failed}); only the programmatic checks below are available."]
        if self.scores is not None:
            lines.append("Scores: " + ", ".join(f"{k}={v}" for k, v in self.scores.model_dump().items() if v is not None))
        failed = [c for c in self.checks if not c.passed]
        if failed:
            lines.append("Failed checks:")
            lines += [f"- [{c.severity}] {c.name}: {c.detail}" for c in failed]
        if self.issues:
            lines.append("Issues:")
            lines += [f"- [{i.severity}] at {i.location}: {i.description} Fix: {i.fix_hint}" for i in self.issues]
        return "\n".join(lines)


def run_checks(session: Session) -> list[CheckResult]:
    """Cheap, objective checks on the canvas (DESIGN §11.2); no model call. Thresholds are calibrated so the
    three demos pass and constructed broken terrains fail (EVIDENCE.md, 2026-10-22)."""
    s = session.state
    h = s.height.astype(np.float64)
    r = np.round(h)  # what stands in the world
    checks = [CheckResult(name="built_something", passed=bool(session.ops), severity="blocking",
                          detail=f"{len(session.ops)} ops" if session.ops else "no ops were applied")]

    changed = (np.abs(h - session.stack.base.base_y) >= 1.0) | ~np.isnan(s.water)
    checks.append(CheckResult(
        name="meaningful_change", passed=bool(changed.mean() >= MIN_CHANGED_SHARE), severity="blocking",
        detail=f"{changed.mean():.1%} of columns differ from the flat base by at least a block (need {MIN_CHANGED_SHARE:.0%})"))

    spikes = np.argwhere(h - _neighbours(h).max(axis=0) > SPIKE_BLOCKS)
    checks.append(CheckResult(
        name="no_spikes", passed=len(spikes) == 0, severity="major",
        detail=f"{len(spikes)} columns stand >{SPIKE_BLOCKS:.0f} blocks above all 4 neighbours" + _example(s, spikes)))

    # Per block, not a central gradient: that compares a column's two neighbours and misses 1-column walls.
    sheer = r - _neighbours(r).min(axis=0) >= SHEER_DROP
    checks.append(CheckResult(
        name="steep_share", passed=bool(sheer.mean() <= MAX_SHEER_SHARE), severity="major",
        detail=f"{sheer.mean():.1%} of columns drop {SHEER_DROP:.0f}+ blocks to a neighbour (steeper than ~70°; "
        f"limit {MAX_SHEER_SHARE:.0%})" + _example(s, np.argwhere(sheer))))

    gentle = surface_metrics(s)[0] < GENTLE_DEG
    nb = _neighbours(r)
    speck = gentle & (np.all(nb == r - 1, axis=0) | np.all(nb == r + 1, axis=0))
    checks.append(CheckResult(
        name="gentle_steps", passed=True, severity="info",
        detail=f"{speck.sum() / max(int(gentle.sum()), 1):.2%} of gentle columns are isolated 1-block bumps or pits"
        + _example(s, np.argwhere(speck))))

    wet = ~np.isnan(s.water) & (s.water > s.height)
    lost = wet & ~(water_top(s) > surface_y(s))
    lost_share = lost.sum() / max(int(wet.sum()), 1)
    checks.append(CheckResult(
        name="water_containment", passed=bool(lost_share <= MAX_WATER_LOST), severity="major",
        detail=f"{int(lost.sum())} of {int(wet.sum())} water columns ({lost_share:.0%}) are too shallow to survive "
        f"rounding to blocks (limit {MAX_WATER_LOST:.0%})" + _example(s, np.argwhere(lost))))

    thin = r - grey_opening(r, size=(3, 3)) >= THIN_BLOCKS
    checks.append(CheckResult(
        name="thin_features", passed=bool(thin.mean() <= MAX_THIN_SHARE), severity="major",
        detail=f"{int(thin.sum())} columns ({thin.mean():.2%}) belong to pillars or walls 1-2 blocks wide and at least "
        f"{THIN_BLOCKS:.0f} high (limit {MAX_THIN_SHARE:.1%})" + _example(s, np.argwhere(thin))))
    return checks


def _neighbours(a: np.ndarray) -> np.ndarray:
    p = np.pad(a, 1, mode="edge")
    return np.stack([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])


def _example(state: WorldState, zx: np.ndarray) -> str:
    if not len(zx):
        return ""
    z, x = zx[len(zx) // 2]
    return f"; e.g. at u={x / (state.width - 1):.2f}, v={z / (state.depth - 1):.2f}"


class CompareOutput(_Strict):
    """Blind comparison reply. Symmetric fields, strengths before the decision."""

    strengths_1: str
    strengths_2: str
    better: Literal["1", "2", "same"]
    why: str


@dataclass(frozen=True)
class AttemptView:
    """What the comparison sees of one attempt: renders and terrain statistics, nothing that reveals recency."""

    renders: list[bytes]
    stats: str


COMPARE_SYSTEM = """\
You compare two Minecraft landscapes, Terrain 1 and Terrain 2, built for the same spec. You built neither. \
Judge which better fulfils the spec: the features it asks for and their layout, natural-looking landforms, \
and clean block-scale quality (no spikes, no floating water). First list the strengths of each terrain, \
then decide. Answer "same" when neither is clearly better; small differences are "same"."""


CRITIC_SYSTEM = """\
You are a strict critic of Minecraft landscapes produced by a terrain-generation agent. You judge the \
result against the user's spec. You did not build it.

Rubric (1-5 each):
- prompt_adherence: does it contain what the spec asks for, at the asked scale? 1 = unrelated, 3 = main \
features present with gaps, 5 = everything asked for.
- layout: placement and proportions of features across the area. 1 = arbitrary, 3 = workable, 5 = \
deliberate and balanced.
- naturalism: would this read as a natural landform? 1 = obviously synthetic (perfect cones, uniform \
spacing), 3 = plausible, 5 = convincing.
- minecraft_quality: block-scale quality: no spikes, no floating water, walkable transitions where \
expected. 1 = broken, 3 = minor artifacts, 5 = clean.
- reference_match: similarity to the reference images; null if there are none.

3 means acceptable. Reserve 5 for results you would show as a showcase. Locate every issue with an op id \
or a normalized 'u,v' position ((0,0) = north-west corner) and give a concrete fix. Say "pass" only if \
there are no major or blocking issues and every score is at least 4."""


class Critic:
    def __init__(self, model: ModelAdapter) -> None:
        self.model = model

    def run_checks(self, session: Session) -> list[CheckResult]:
        return run_checks(session)

    def critique(
        self, spec: dict, session: Session, renders: list[bytes] | None = None, references: list[bytes] | None = None
    ) -> tuple[CritiqueResult, ModelResponse | None]:
        """Returns the critique and the model response (for budget and tracing; None if short-circuited)."""
        checks = self.run_checks(session)
        if any(not c.passed and c.severity == "blocking" for c in checks):
            return CritiqueResult(checks=checks, verdict="revise"), None

        text = [
            "# Spec\n" + json.dumps(spec, indent=2, sort_keys=True),
            f"# Terrain statistics\n{session.stats()}",
            f"# Features (measured by code)\n{feature_report(measure_features(session.stack))}",
            f"# Ops applied\n{session.summary()}",
            "# Programmatic checks\n" + "\n".join(f"- {c.name}: {'pass' if c.passed else 'FAIL'} ({c.detail})" for c in checks),
        ]
        if references:
            text.append("Reference images are attached above.")
        if renders:
            text.append("Renders of the result are attached below.")
        else:
            text.append("No render is available yet: judge from the statistics and op parameters, and score "
                        "naturalism and minecraft_quality conservatively.")
        content = [image_block(r) for r in references or []]
        content.append(text_block("\n\n".join(text)))
        content.extend(image_block(r) for r in renders or [])

        out, response = self._structured(content, CRITIC_SYSTEM, CritiqueOutput)
        failed_major = any(not c.passed and c.severity == "major" for c in checks)
        verdict = "revise" if failed_major else out.verdict
        return CritiqueResult(checks=checks, issues=out.issues, scores=out.scores, verdict=verdict), response

    def compare(
        self, spec: dict, first: AttemptView, second: AttemptView, references: list[bytes] | None = None
    ) -> tuple[CompareOutput, ModelResponse]:
        """Blind comparison for the acceptance gate (DESIGN §11.4): two terrains as "Terrain 1" and "Terrain 2",
        with renders and statistics only. Nothing says which is newer: no op lists, no previous critique."""
        content = [image_block(r) for r in references or []]
        intro = "# Spec\n" + json.dumps(spec, indent=2, sort_keys=True)
        if references:
            intro += "\n\nReference images are attached above."
        content.append(text_block(intro))
        for label, view in (("1", first), ("2", second)):
            content.append(text_block(f"# Terrain {label}\n{view.stats}" + ("\nRenders follow." if view.renders else "")))
            content.extend(image_block(r) for r in view.renders)
        content.append(text_block("Which terrain better fulfils the spec? Answer \"same\" if neither is clearly better."))
        return self._structured(content, COMPARE_SYSTEM, CompareOutput)

    def _structured(self, content: list[dict], system: str, schema: type[_T]) -> tuple[_T, ModelResponse]:
        """One strict-JSON call. Raises CriticError (carrying the response, so its cost is still charged)."""
        response = self.model.complete([{"role": "user", "content": content}], system=system,
                                       response_schema=schema.model_json_schema())
        if response.stop_reason != "end_turn":
            raise CriticError(f"stopped with {response.stop_reason!r} (details: {response.stop_details})", response)
        try:
            return schema.model_validate_json(response.text), response
        except ValidationError as e:
            raise CriticError(f"returned invalid JSON: {e}", response) from None
