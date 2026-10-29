"""Critic (DESIGN §11): scores a stage result and returns located, actionable issues.

Two steps: cheap programmatic checks first; if none is blocking, one model call in a fresh context with
the spec, references, renders, terrain stats and check results. The critic never sees the builder's
conversation. It only judges; accept/rollback decisions stay in the harness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from agent.catalogue import hint_problems, tool_catalogue
from agent.checks import CheckResult, run_checks
from agent.models import ModelAdapter, ModelReplyError, ModelResponse, structured_call
from agent.session import Session
from agent.checklist import Item, checklist, summary as checklist_summary
from agent.spec import spec_text
from agent.tools import image_block, text_block
from terrain.features import feature_report, measure_features
from terrain.schema import Strict
from terrain.state import WorldState

Score = Literal[1, 2, 3, 4, 5]  # enum, not min/max: structured outputs reject numeric constraints


class CriticError(ModelReplyError):
    """The critic model's reply was unusable (refusal, truncation, invalid JSON)."""


class Change(Strict):
    param: str = Field(description="Parameter name of the tool")
    value: str = Field(description='New value as JSON, e.g. "0.3", "\\"u\\"" or "[[0.2, 0.5], [0.8, 0.5]]"')


class Issue(Strict):
    location: str = Field(description="Op id (e.g. 'raise_peak_1') or normalized 'u,v' position")
    severity: Literal["minor", "major", "blocking"]
    description: str
    tool: str = Field(description="The tool that makes the fix: an op name, update_op (change an existing op) or "
                      "remove_op; 'none' if no tool can make it")
    op_id: str = Field(description="For update_op and remove_op: the id of the one op to change or remove, exactly "
                       "as in the op list (e.g. 'mountain_1'); empty for other tools. One fix per op.")
    changes: list[Change] = Field(description="Parameter values to set with that tool; empty for remove_op or none")
    fix_hint: str = Field(description="The fix in words; for tool 'none', the capability that is missing")


class RubricScores(Strict):
    prompt_adherence: Score
    layout: Score
    naturalism: Score
    minecraft_quality: Score
    materials: Score
    reference_match: Score | None = Field(description="null when there is no reference image")


class CritiqueOutput(Strict):
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
    checklist: list[Item] = []  # spec checklist (DESIGN §11.3), computed by code; empty without a feature graph
    fix_problems: list[list[str]] = []  # per issue: why its structured fix can't be applied as written (hint_problems)

    def fix_stats(self) -> dict[str, int]:
        """Structured-fix validity for reports: issues, those naming a tool, those valid as written, tool "none"."""
        problems = self.fix_problems or [[] for _ in self.issues]
        with_fix = [p for i, p in zip(self.issues, problems) if i.tool != "none"]
        return {"issues": len(self.issues), "with_fix": len(with_fix), "valid": sum(not p for p in with_fix),
                "none": len(self.issues) - len(with_fix)}

    @property
    def checklist_passes(self) -> int:
        return sum(i.status == "pass" for i in self.checklist)

    def checklist_losses(self, previous: CritiqueResult) -> list[Item]:
        """Checklist items that pass in `previous` but not here (shown when gate rule 2 rolls back)."""
        passed_before = {i.id for i in previous.checklist if i.status == "pass"}
        return [i for i in self.checklist if i.status != "pass" and i.id in passed_before]

    def checklist_gains(self, previous: CritiqueResult) -> list[Item]:
        """Checklist items that pass here but not in `previous` (named in the gate's why on a checklist gain)."""
        passed_before = {i.id for i in previous.checklist if i.status == "pass"}
        return [i for i in self.checklist if i.status == "pass" and i.id not in passed_before]

    def regressions(self, previous: CritiqueResult) -> list[CheckResult]:
        """Blocking or major checks that fail here but passed in `previous` (acceptance gate rule 1)."""
        passed_before = {c.name for c in previous.checks if c.passed}
        return [c for c in self.checks if not c.passed and c.severity in ("blocking", "major") and c.name in passed_before]

    @property
    def score(self) -> float:
        """Mean rubric score; 0 when blocked. For reports only: it never gates acceptance (DESIGN §11.4)."""
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
        if self.checklist:
            lines.append(checklist_summary(self.checklist))
        if self.issues:
            lines.append("Issues:")
            problems = self.fix_problems or [[] for _ in self.issues]
            lines += [f"- [{i.severity}] at {i.location}: {i.description} Fix: {i.fix_hint}"
                      + (f" (fix not applicable as written: {'; '.join(p)})" if p else _fix_text(i))
                      for i, p in zip(self.issues, problems)]
        return "\n".join(lines)


def _palette_text(state: WorldState) -> str:
    from terrain.materials import DEFAULT_RULES

    rules = state.palette_rules if state.palette_rules is not None else [r.model_dump(mode="json", exclude_none=True) for r in DEFAULT_RULES]
    label = "custom" if state.palette_rules is not None else "default"
    return f"{label}: " + json.dumps(rules, separators=(",", ":"))


def spec_checklist(spec: dict, session: Session) -> list[Item]:
    """The spec checklist for the current terrain, or [] when the run has no feature graph."""
    return checklist(spec["graph"], session.stack) if spec.get("graph") is not None else []


class CompareOutput(Strict):
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
- materials: the surface blocks, judged in the 3D render: do they suit each landform and each part of the \
scene (wherever it is: either side, background or foreground, summits or lower slopes), and do they vary \
naturally? With a reference, compare every landform's surface with what the reference shows at that place, \
not only those the spec gives a material for: slopes the reference shows green stay vegetated even when \
steep, and rock belongs where it shows bare rock. Without one, rock suits steep or high ground, soil and \
grass gentle ground, sand or gravel the water's edge. 1 = wrong or one block everywhere, 3 = plausible with \
artifacts (rings or stripes of one block, bare stone walls, abrupt switches at one slope), 5 = convincing. \
Fix materials with set_palette_rules or paint_region; a palette WARNING in the op list (a rule mostly \
overridden by an earlier rule) is fixed by reordering the rules.
- reference_match: how well the terrain matches the reference images, null if there are none. The photo is a \
perspective view; with reference images the 3D render is taken from the south edge looking north, the \
viewpoint the spec assumes for the photo (left = west, near = south), so compare the main landforms, their \
arrangement (left/right, near/far), proportions, mood and palette, not exact geometry.

3 means acceptable. Reserve 5 for results you would show as a showcase. Locate every issue with an op id \
or a normalized 'u,v' position ((0,0) = north-west corner) and give a concrete fix: the tool that makes it \
and the parameter values to set, or tool "none" when no tool can make it. Say "pass" only if there are no \
major or blocking issues and every score is at least 4."""


def _fix_text(issue: Issue) -> str:
    """The structured fix as the builder sees it, e.g. " (update_op mountain_1: height=120, roughness=0.8)"."""
    if issue.tool == "none":
        return " (no tool can do this yet)"
    target = f" {issue.op_id}" if issue.tool in ("update_op", "remove_op") else ""
    changes = ", ".join(f"{c.param}={c.value}" for c in issue.changes)
    return f" ({issue.tool}{target}{': ' + changes if changes else ''})"


class Critic:
    def __init__(self, model: ModelAdapter, catalogue: bool = True) -> None:
        """`catalogue`: append the generated tool catalogue to the system prompt (DESIGN §11.5); off only for
        the A/B in scripts/catalogue_ab.py. The prompt is built once, so it stays byte-stable (caching)."""
        self.model = model
        self.system = CRITIC_SYSTEM + ("\n\n" + tool_catalogue() if catalogue else "")

    def run_checks(self, session: Session) -> list[CheckResult]:
        return run_checks(session)

    def critique(
        self, spec: dict, session: Session, renders: list[bytes] | None = None, references: list[bytes] | None = None
    ) -> tuple[CritiqueResult, ModelResponse | None]:
        """Returns the critique and the model response (for budget and tracing; None if short-circuited)."""
        checks = self.run_checks(session)
        items = spec_checklist(spec, session)
        if any(not c.passed and c.severity == "blocking" for c in checks):
            return CritiqueResult(checks=checks, checklist=items, verdict="revise"), None

        text = [
            "# Spec\n" + spec_text(spec),
            f"# Terrain statistics\n{session.stats()}",
            f"# Features (measured by code)\n{feature_report(measure_features(session.stack))}",
            f"# Ops applied\n{session.summary()}",
            *(["# Spec checklist (code-checked; failed items are issues)\n" + checklist_summary(items)] if items else []),
            "# Surface palette in use (set_palette_rules replaces all of it)\n" + _palette_text(session.state),
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

        out, response = structured_call(self.model, content, self.system, CritiqueOutput, CriticError)
        failed_major = any(not c.passed and c.severity == "major" for c in checks)
        failed_major |= any(i.status == "fail" and i.severity == "major" for i in items)
        verdict = "revise" if failed_major else out.verdict
        # Every fix is checked against the stack it was written for, so the builder never gets one that
        # names a missing op, parameter or invalid value (DESIGN §11.5).
        problems = [hint_problems(i, session.stack) for i in out.issues]
        return CritiqueResult(checks=checks, issues=out.issues, scores=out.scores, verdict=verdict, checklist=items,
                              fix_problems=problems), response

    def compare(
        self, spec: dict, first: AttemptView, second: AttemptView, references: list[bytes] | None = None
    ) -> tuple[CompareOutput, ModelResponse]:
        """Blind comparison for the acceptance gate (DESIGN §11.4): two terrains as "Terrain 1" and "Terrain 2",
        with renders and statistics only. Nothing says which is newer: no op lists, no previous critique."""
        content = [image_block(r) for r in references or []]
        intro = "# Spec\n" + spec_text(spec)
        if references:
            intro += "\n\nReference images are attached above."
        content.append(text_block(intro))
        for label, view in (("1", first), ("2", second)):
            content.append(text_block(f"# Terrain {label}\n{view.stats}" + ("\nRenders follow." if view.renders else "")))
            content.extend(image_block(r) for r in view.renders)
        content.append(text_block("Which terrain better fulfils the spec? Answer \"same\" if neither is clearly better."))
        return structured_call(self.model, content, COMPARE_SYSTEM, CompareOutput, CriticError)
