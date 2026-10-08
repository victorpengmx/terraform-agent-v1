"""Agent harness (DESIGN §9): the loop that turns a prompt into terrain.

Grown from a port of Thorsten Ball's "How to Build an Agent". Two layers:

- `run_stage`: the inner tool loop. Model call -> execute tool calls -> feed results back, until the
  model ends its turn, the budget runs out or the model refuses. Branches on stop_reason explicitly.
- `run_task`: the outer loop. BUILD -> CRITIQUE -> (REVISE | ACCEPT), up to `max_rounds`, with
  score-gated acceptance: a revision is kept only if it scores higher than the best attempt so far;
  otherwise the session is rolled back and the builder is told its fix made things worse.

The SPEC step (DESIGN §9.3) comes in M5, CLARIFY and multi-stage runs (§9.1) later; for now the spec is
the prompt and there is one stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agent.budget import Budget, BudgetTracker
from agent.context import ContextManager, Episode
from agent.critic import Critic, CritiqueResult
from agent.models import ModelAdapter, ModelResponse
from agent.session import Session, Snapshot
from agent.tools import Tool, execute, tool_result
from agent.tracing import Tracer, TraceRecord

Renderer = Callable[[Session], list[bytes]]  # PNG/JPEG bytes per view; L1/L2 renderers plug in here


@dataclass
class RunResult:
    run_id: str
    stop: str  # why the run ended: accepted | max_rounds | done | budget: ... | refused: ...
    rounds: int
    session: Session
    critiques: list[CritiqueResult] = field(default_factory=list)
    best_score: float | None = None
    cost_usd: float = 0.0
    summary: str = ""  # builder's final text

    def __str__(self) -> str:
        score = "n/a" if self.best_score is None else f"{self.best_score:.2f}"
        return (f"run {self.run_id}: {self.stop} after {self.rounds} round(s); best score {score}; "
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

    best: tuple[float, Snapshot, CritiqueResult] | None = None
    feedback: CritiqueResult | None = None
    rolled_back_from: float | None = None

    for round_no in range(1, max_rounds + 1):
        result.rounds = round_no
        renders = render(session) if render and session.ops else []
        episode = context.build_episode(session, stage, critique=feedback, renders=renders, rolled_back_from=rolled_back_from)
        outcome = run_stage(run, episode, stage, render)
        result.summary = outcome.summary or result.summary
        if outcome.stop != "done" or critic is None:
            result.stop = outcome.stop
            break

        critique = _critique(run, critic, spec, stage, render, images)
        result.critiques.append(critique)
        if best is not None and critique.score <= best[0]:
            # Score-gated acceptance: this revision is no better; roll back to the best attempt.
            session.restore(best[1])
            feedback, rolled_back_from = best[2], critique.score
            _trace(run, stage, note=f"rolled back: score {critique.score:.2f} <= best {best[0]:.2f}")
        else:
            best = (critique.score, session.snapshot(), critique)
            feedback, rolled_back_from = critique, None
            if critique.verdict == "pass":
                result.stop = "accepted"
                break
        if (reason := run.tracker.exceeded()) is not None:
            result.stop = f"budget: {reason}"
            break
    else:
        result.stop = "max_rounds"

    if best is not None:
        session.restore(best[1])  # always finish on the best-scoring attempt
        result.best_score = best[0]
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
                results = [execute(run.tools, tu) for tu in response.tool_uses]
                _trace(run, stage, response, tool_calls=_tool_log(response, results))
                u = response.usage
                if run.context.should_rotate(episode, u.input_tokens + u.cache_read_tokens + u.cache_write_tokens):
                    # Results are already applied to the session; the new episode is built from it.
                    renders = render(run.session) if render else []
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
    renders = render(run.session) if render else []
    critique, response = critic.critique(spec, run.session, renders=renders, references=references)
    if response is not None:
        _charge(run, response)
    _trace(run, f"{stage}/critique", response, scores=critique.model_dump(), note=f"score {critique.score:.2f}, {critique.verdict}")
    return critique


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
