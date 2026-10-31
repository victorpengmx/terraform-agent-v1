"""Context manager: decides what the builder sees on each model call.

Episodes: each BUILD round starts a fresh conversation whose opening message is written from current
state (spec, op list, terrain stats, last critique, notes, latest renders). Inside an episode the
history is append-only. 
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent.models import ROTATION_TOKENS
from agent.prompts import BUILDER_SYSTEM
from agent.session import Session
from agent.spec import spec_text
from agent.tools import Tool, image_block, text_block
from terrain.config import WORLD_MAX_Y, WORLD_MIN_Y

if TYPE_CHECKING:
    from agent.critic import CritiqueResult


@dataclass
class Episode:
    """One append-only conversation with the builder."""

    messages: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class RollbackFeedback:
    """Why the acceptance gate rejected the last revision and what it had changed (DESIGN §11.4 step 4)."""

    round: int
    why: str
    diff: list[str]  # op_diff(accepted, rejected): what the rejected revision changed, now undone
    problems: list[str]  # the rejected attempt's critic issues and failed checks

    def as_text(self) -> str:
        lines = [f"# Your last revision was rolled back\nRound {self.round} was not kept: {self.why}"]
        if self.diff:
            lines.append("What that revision changed (now undone):\n" + "\n".join(f"- {d}" for d in self.diff))
        if self.problems:
            lines.append("Problems found in it:\n" + "\n".join(f"- {p}" for p in self.problems))
        lines.append("You are back on the last accepted terrain, described above. Try a different fix.")
        return "\n\n".join(lines)


class ContextManager:
    def __init__(
        self,
        spec: dict[str, Any],
        area: tuple[int, int],
        references: list[bytes] | None = None,
        max_input_tokens: int = ROTATION_TOKENS,
        max_messages: int = 80,
    ) -> None:
        self.spec = spec  # pinned for the whole run (DESIGN §9.3)
        self.area = area
        self.references = references or []
        self.max_input_tokens = max_input_tokens
        self.max_messages = max_messages
        self.notes: list[str] = []  # scratchpad: model-written, survives rotation
        self._system = BUILDER_SYSTEM.format(width=area[0], depth=area[1], min_y=WORLD_MIN_Y, max_y=WORLD_MAX_Y - 1)

    def system_prompt(self) -> str:
        """Fixed for the run, so the cached prefix (tools + system) stays valid across episodes."""
        return self._system

    def note_tool(self) -> Tool:
        def write_note(p: dict[str, Any]) -> str:
            note = str(p.get("note", "")).strip()
            if not note:
                return "note was empty; nothing saved"
            self.notes.append(note)
            return f"saved note {len(self.notes)}"

        return Tool(
            name="write_note",
            description="Save a short progress note (plan, decision, open problem). Notes are shown to you at the start of every new conversation.",
            input_schema={
                "type": "object",
                "properties": {"note": {"type": "string", "description": "One or two sentences."}},
                "required": ["note"],
                "additionalProperties": False,
            },
            handler=write_note,
        )

    def build_episode(
        self,
        session: Session,
        stage: str,
        critique: CritiqueResult | None = None,
        renders: list[bytes] | None = None,
        rollback: RollbackFeedback | None = None,
        continuing: bool = False,
    ) -> Episode:
        """Opening message: references, task, spec, terrain state, critique (of the accepted terrain), rollback
        feedback, notes, latest renders."""
        parts = [f"# Task\nStage: {stage}. Build this stage of the terrain described by the spec."]
        if continuing:
            parts.append("This continues earlier work in this stage; the previous conversation was closed to save context. "
                         "The state below includes everything done so far.")
        parts.append("# Spec\n" + spec_text(self.spec))
        parts.append(f"# Current terrain\n{session.stats()}\n\nOps so far:\n{session.summary()}")
        if critique is not None:
            parts.append("# Critique of the current terrain\n" + critique.as_text())
        if rollback is not None:
            parts.append(rollback.as_text())
        if critique is not None:
            parts.append("Address the issues above, most severe first.")
        if self.notes:
            parts.append("# Your notes\n" + "\n".join(f"- {n}" for n in self.notes))
        if self.references:
            parts.append("Reference images from the user are attached above.")
        if renders:
            parts.append("Latest render(s) of the terrain are attached below.")

        content = [image_block(r) for r in self.references]
        content.append(text_block("\n\n".join(parts)))
        content.extend(image_block(r) for r in renders or [])
        return Episode(messages=[{"role": "user", "content": content}])

    def should_rotate(self, episode: Episode, last_input_tokens: int) -> bool:
        """Start a fresh episode when this one is too long to keep paying for."""
        return last_input_tokens >= self.max_input_tokens or len(episode.messages) >= self.max_messages
