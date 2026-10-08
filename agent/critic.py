"""Critic (DESIGN §11): scores a stage result and returns located, actionable issues.

Two steps: cheap programmatic checks first; if none is blocking, one model call in a fresh context with
the spec, references, renders, terrain stats and check results. The critic never sees the builder's
conversation. It only judges; accept/rollback decisions stay in the harness.
"""

from __future__ import annotations

import json
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.models import ModelAdapter, ModelResponse
from agent.session import Session
from agent.tools import image_block, text_block

Score = Literal[1, 2, 3, 4, 5]  # enum, not min/max: structured outputs reject numeric constraints
SPIKE_BLOCKS = 3.0  # a column this much higher than all 4 neighbours reads as a 1-block spike in game


class CriticError(Exception):
    """The critic model's reply was unusable (refusal, truncation, invalid JSON)."""


class CheckResult(BaseModel):
    """Output of one programmatic check."""

    name: str
    passed: bool
    severity: Literal["info", "major", "blocking"]  # blocking = skip the model call, must revise
    detail: str = ""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")  # additionalProperties: false, required by structured outputs


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

    @property
    def score(self) -> float:
        """Mean rubric score; 0 when blocked. Used for score-gated acceptance."""
        if self.scores is None:
            return 0.0
        vals = [v for v in self.scores.model_dump().values() if v is not None]
        return sum(vals) / len(vals)

    def as_text(self) -> str:
        lines = [f"Verdict: {self.verdict}. Score {self.score:.2f}/5."]
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
        """Cheap, objective checks on the canvas; no model call."""
        h = session.state.height
        checks = [
            CheckResult(
                name="built_something",
                passed=bool(session.ops),
                severity="blocking",
                detail="no ops were applied" if not session.ops else f"{len(session.ops)} ops",
            )
        ]

        inner = h[1:-1, 1:-1]
        neighbours = np.stack([h[:-2, 1:-1], h[2:, 1:-1], h[1:-1, :-2], h[1:-1, 2:]])
        spikes = np.argwhere(inner - neighbours.max(axis=0) > SPIKE_BLOCKS)
        detail = f"{len(spikes)} columns stand >{SPIKE_BLOCKS:.0f} blocks above all neighbours"
        if len(spikes):
            z, x = spikes[0] + 1
            detail += f", e.g. at u={x / (session.state.width - 1):.2f}, v={z / (session.state.depth - 1):.2f}"
        checks.append(CheckResult(name="no_spikes", passed=len(spikes) == 0, severity="major", detail=detail))
        return checks

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

        response = self.model.complete(
            [{"role": "user", "content": content}],
            system=CRITIC_SYSTEM,
            response_schema=CritiqueOutput.model_json_schema(),
        )
        if response.stop_reason != "end_turn":
            raise CriticError(f"critic stopped with {response.stop_reason!r} (details: {response.stop_details})")
        try:
            out = CritiqueOutput.model_validate_json(response.text)
        except ValidationError as e:
            raise CriticError(f"critic returned invalid JSON: {e}") from None

        failed_major = any(not c.passed and c.severity == "major" for c in checks)
        verdict = "revise" if failed_major else out.verdict
        return CritiqueResult(checks=checks, issues=out.issues, scores=out.scores, verdict=verdict), response
