"""Agent harness (DESIGN §9): the loop that turns a prompt into terrain.

Grown from a port of Thorsten Ball's "How to Build an Agent". Two layers:

- `run_stage`: the inner tool loop. Model call -> execute tool calls -> feed results back, until the
  model ends its turn, the budget runs out or the model refuses. Branches on stop_reason explicitly.
- `run_task`: the outer loop. BUILD -> CRITIQUE -> (REVISE | ACCEPT), up to `max_rounds`, with the
  acceptance gate of DESIGN §11.4: the first attempt is the baseline; a revision is rolled back if it
  fails a blocking/major check the accepted attempt passed, or if a blind comparison prefers the accepted
  attempt. Rolled-back rounds tell the builder why and what they had changed. Absolute rubric scores never
  gate (they are noisy and tie often), and the run always ends on the latest accepted attempt.

The SPEC step (DESIGN §9.3) comes in M5, CLARIFY and multi-stage runs (§9.1) later; for now the spec is
the prompt and there is one stage.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from agent.budget import Budget, BudgetTracker
from agent.context import ContextManager, Episode, RollbackFeedback
from agent.critic import AttemptView, Critic, CriticError, CritiqueResult
from agent.models import ModelAdapter, ModelResponse
from agent.session import Session, Snapshot, op_diff, terrain_stats
from agent.tools import Tool, execute, tool_result
from agent.tracing import Tracer, TraceRecord
from terrain.state import WorldState

# (state, title) -> PNG/JPEG bytes per view; L1/L2 renderers plug in here. title None = a neutral default,
# used for the blind comparison so no image says how many ops built it.
Renderer = Callable[[WorldState, str | None], list[bytes]]
Winner = Literal["revision", "previous", "same"]


@dataclass
class RunResult:
    run_id: str
    stop: str  # why the run ended: accepted | max_rounds | done | budget: ... | refused: ...
    rounds: int
    session: Session
    critiques: list[CritiqueResult] = field(default_factory=list)
    accepted_round: int | None = None  # the attempt the run ended on (None without a critic)
    score: float | None = None  # its mean rubric score, for reports only (never used to gate)
    cost_usd: float = 0.0
    summary: str = ""  # builder's final text

    def __str__(self) -> str:
        kept = "n/a" if self.accepted_round is None else f"round {self.accepted_round} (score {self.score:.2f})"
        return (f"run {self.run_id}: {self.stop} after {self.rounds} round(s); kept {kept}; "
                f"${self.cost_usd:.2f}\n{self.session.stats()}\n{self.session.summary()}\n\n{self.summary}")


@dataclass
class _Run:
    """Per-run wiring shared by the inner and outer loops."""

    session: Session
    tools: dict[str, Tool]
    tool_params: list[dict[str, Any]]  # computed once: byte-stable for the prompt cache
    model: ModelAdapter
    context: ContextManager
    tracer: Tracer
    tracker: BudgetTracker
    step: int = 0


@dataclass
class StageOutcome:
    stop: str  # done | budget: ... | refused: ...
    summary: str


@dataclass(frozen=True)
class Attempt:
    """One judged build: the canvas checkpoint and its critique."""

    round: int
    checkpoint: Snapshot
    critique: CritiqueResult


def run_task(
    *,
    prompt: str,
    images: list[bytes],
    session: Session,
    tools: dict[str, Tool],
    model: ModelAdapter,
    tracer: Tracer,
    budget: Budget,
    critic: Critic | None = None,
    render: Renderer | None = None,
    max_rounds: int = 3,
    stage: str = "terrain",
    compare_both_orders: bool = False,
) -> RunResult:
    spec = {"prompt": prompt}
    context = ContextManager(spec, area=(session.state.width, session.state.depth), references=images)
    tools = {**tools, "write_note": context.note_tool()}
    run = _Run(
        session=session,
        tools=tools,
        tool_params=[t.to_param() for t in tools.values()],
        model=model,
        context=context,
        tracer=tracer,
        tracker=BudgetTracker(budget),
    )
    result = RunResult(run_id=tracer.run_id, stop="", rounds=0, session=session)

    accepted: Attempt | None = None
    rollback: RollbackFeedback | None = None

    for round_no in range(1, max_rounds + 1):
        result.rounds = round_no
        renders = render(session.state, f"{len(session.ops)} ops") if render and session.ops else []
        feedback = accepted.critique if accepted else None
        episode = context.build_episode(session, stage, critique=feedback, renders=renders, rollback=rollback)
        outcome = run_stage(run, episode, stage, render)
        result.summary = outcome.summary or result.summary
        if outcome.stop != "done" or critic is None:
            result.stop = outcome.stop
            break

        attempt = Attempt(round_no, session.snapshot(), _critique(run, critic, spec, stage, render, images))
        result.critiques.append(attempt.critique)
        if accepted is None:
            keep, why = True, "first attempt (baseline)"
        else:
            keep, why = _gate(run, critic, spec, stage, render, images, accepted, attempt, compare_both_orders)

        if keep:
            accepted, rollback = attempt, None
            _trace(run, stage, note=f"accepted round {round_no}: {why}")
            if attempt.critique.verdict == "pass":
                result.stop = "accepted"
                break
        else:
            problems = [f"[{i.severity}] at {i.location}: {i.description}" for i in attempt.critique.issues]
            problems += [f"[{c.severity}] check {c.name}: {c.detail}" for c in attempt.critique.checks if not c.passed]
            rollback = RollbackFeedback(round=round_no, why=why, problems=problems,
                                        diff=op_diff(accepted.checkpoint.entries, attempt.checkpoint.entries))
            session.restore(accepted.checkpoint)
            _trace(run, stage, note=f"rolled back round {round_no}: {why}")
        if (reason := run.tracker.exceeded()) is not None:
            result.stop = f"budget: {reason}"
            break
    else:
        result.stop = "max_rounds"

    if accepted is not None:
        session.restore(accepted.checkpoint)  # always finish on the latest accepted attempt
        result.accepted_round, result.score = accepted.round, accepted.critique.score
    result.cost_usd = run.tracker.cost_usd
    _trace(run, stage, note=f"run ended: {result.stop}")
    return result


def run_stage(run: _Run, episode: Episode, stage: str, render: Renderer | None = None) -> StageOutcome:
    """Inner tool loop for one stage. Appends to `episode` and changes the session through tools."""
    last_text = ""
    while True:
        if (reason := run.tracker.exceeded()) is not None:
            return StageOutcome(stop=f"budget: {reason}", summary=last_text)

        response = run.model.complete(episode.messages, system=run.context.system_prompt(), tools=run.tool_params)
        _charge(run, response)
        episode.messages.append({"role": "assistant", "content": response.content})
        last_text = response.text or last_text

        match response.stop_reason:
            case "tool_use":
                try:
                    results = [execute(run.tools, tu) for tu in response.tool_uses]
                except Exception as e:  # a bug: record the calls that triggered it, then let it propagate
                    calls = [{"name": tu.name, "input": tu.input} for tu in response.tool_uses]
                    _trace(run, stage, response, tool_calls=calls, note=f"crash in a tool: {type(e).__name__}: {e}")
                    raise
                _trace(run, stage, response, tool_calls=_tool_log(response, results))
                u = response.usage
                if run.context.should_rotate(episode, u.input_tokens + u.cache_read_tokens + u.cache_write_tokens):
                    # Results are already applied to the session; the new episode is built from it.
                    renders = render(run.session.state, f"{len(run.session.ops)} ops") if render else []
                    fresh = run.context.build_episode(run.session, stage, renders=renders, continuing=True)
                    episode.messages[:] = fresh.messages  # a new conversation, not an edit of the old one
                    _trace(run, stage, note="episode rotated")
                else:
                    episode.messages.append({"role": "user", "content": results})
            case "end_turn":
                _trace(run, stage, response)
                return StageOutcome(stop="done", summary=last_text)
            case "max_tokens":
                # A tool call in a truncated reply may be incomplete: don't run it, ask for a re-issue.
                _trace(run, stage, response, note="hit max_tokens")
                cut = [tool_result(tu.id, "This call was cut off by the output limit and was not run. Re-issue it.", is_error=True)
                       for tu in response.tool_uses]
                episode.messages.append({"role": "user", "content": cut or "Your reply was cut off by the output limit. Continue, more briefly."})
            case "pause_turn":
                _trace(run, stage, response, note="pause_turn; resuming")  # resend as-is so the model continues
            case "refusal":
                _trace(run, stage, response, note=f"refusal: {response.stop_details}")
                return StageOutcome(stop=f"refused: {response.stop_details}", summary=last_text)
            case other:
                raise RuntimeError(f"unhandled stop_reason {other!r}")


def _critique(run: _Run, critic: Critic, spec: dict, stage: str, render: Renderer | None, references: list[bytes]) -> CritiqueResult:
    """Critique the current terrain. An unusable critic reply is retried once; a second failure records a
    failed round (programmatic checks only) instead of crashing the run."""
    renders = render(run.session.state, f"{len(run.session.ops)} ops") if render else []
    error = ""
    for attempt in (1, 2):
        try:
            critique, response = critic.critique(spec, run.session, renders=renders, references=references)
        except CriticError as e:
            if e.response is not None:
                _charge(run, e.response)
            error = str(e)
            _trace(run, f"{stage}/critique", e.response, note=f"critic failed (try {attempt} of 2): {e}")
            continue
        if response is not None:
            _charge(run, response)
        _trace(run, f"{stage}/critique", response, scores=critique.model_dump(), note=f"score {critique.score:.2f}, {critique.verdict}")
        return critique
    critique = CritiqueResult(checks=critic.run_checks(run.session), verdict="revise", failed=error)
    _trace(run, f"{stage}/critique", scores=critique.model_dump(), note="critic failed twice; round recorded with checks only")
    return critique


def _gate(run: _Run, critic: Critic, spec: dict, stage: str, render: Renderer | None, references: list[bytes],
          accepted: Attempt, revision: Attempt, both_orders: bool) -> tuple[bool, str]:
    """Acceptance gate (DESIGN §11.4): (keep the revision?, why)."""
    regressed = revision.critique.regressions(accepted.critique)
    if regressed:  # rule 1
        return False, "it fails checks the accepted attempt passed: " + "; ".join(f"{c.name} ({c.detail})" for c in regressed)
    # Rule 2 (spec checklist pass count) arrives with the checklist in M5.

    # Rule 3: blind comparison in a seeded random order; only a win for the accepted attempt rolls back.
    previous = AttemptView(render(accepted.checkpoint.state, None) if render else [], terrain_stats(accepted.checkpoint.state))
    current = AttemptView(render(revision.checkpoint.state, None) if render else [], terrain_stats(revision.checkpoint.state))
    revision_first = random.Random(run.session.stack.base.seed * 1000 + revision.round).random() < 0.5
    orders = [revision_first, not revision_first] if both_orders else [revision_first]
    verdicts = [_compare(run, critic, spec, stage, references, previous, current, rev_first, revision.round) for rev_first in orders]
    winners = {w for w, _ in verdicts}
    why = verdicts[0][1]
    if winners == {"previous"}:
        return False, f"a blind comparison preferred the accepted attempt: {why}"
    if winners == {"revision"}:
        return True, f"a blind comparison preferred it: {why}"
    return True, f"no clear winner in the blind comparison, so the revision is kept: {why}"


def _compare(run: _Run, critic: Critic, spec: dict, stage: str, references: list[bytes], previous: AttemptView,
             current: AttemptView, revision_first: bool, round_no: int) -> tuple[Winner, str]:
    """One blind comparison, retried once on an unusable reply; two failures count as no winner."""
    first, second = (current, previous) if revision_first else (previous, current)
    order = {"1": "revision", "2": "previous"} if revision_first else {"1": "previous", "2": "revision"}
    for attempt in (1, 2):
        try:
            out, response = critic.compare(spec, first, second, references)
        except CriticError as e:
            if e.response is not None:
                _charge(run, e.response)
            _trace(run, f"{stage}/compare", e.response, note=f"comparison failed (try {attempt} of 2): {e}")
            continue
        _charge(run, response)
        winner: Winner = "same" if out.better == "same" else order[out.better]  # type: ignore[assignment]
        _trace(run, f"{stage}/compare", response, scores={"round": round_no, "order": order, "winner": winner, **out.model_dump()},
               note=f"blind comparison: {winner}")
        return winner, out.why
    return "same", "the comparison failed twice"


def _charge(run: _Run, response: ModelResponse) -> None:
    run.tracker.charge(response.usage, response.usage.cost_usd(response.model))


def _tool_log(response: ModelResponse, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"name": tu.name, "input": tu.input, "is_error": r["is_error"],
         "result": r["content"] if isinstance(r["content"], str) else "[content blocks]"}
        for tu, r in zip(response.tool_uses, results)
    ]


def _trace(run: _Run, stage: str, response: ModelResponse | None = None, **fields: Any) -> None:
    run.step += 1
    record = TraceRecord(run_id=run.tracer.run_id, stage=stage, step=run.step, **fields)
    if response is not None:
        u = response.usage
        record.model = response.model
        record.request_id = response.request_id
        record.stop_reason = response.stop_reason
        record.usage = {"input": u.input_tokens, "output": u.output_tokens, "cache_read": u.cache_read_tokens, "cache_write": u.cache_write_tokens}
        record.cost_usd = u.cost_usd(response.model)
        record.latency_s = round(response.latency_s, 3)
        record.text = response.text or None
    run.tracer.log(record)
