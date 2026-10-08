"""Run budgets: stop runaway loops on steps, cost or wall-clock time.

The harness calls `charge()` after every model call and checks `exceeded()` before the next one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from agent.models import Usage


@dataclass(frozen=True)
class Budget:
    max_steps: int = 40  # model calls per run (builder and critic)
    max_cost_usd: float = 2.00
    max_seconds: float = 900.0


@dataclass
class BudgetTracker:
    budget: Budget
    steps: int = 0
    cost_usd: float = 0.0
    usage: Usage = field(default_factory=Usage)
    started: float = field(default_factory=time.monotonic)

    def charge(self, usage: Usage, cost_usd: float | None) -> None:
        self.steps += 1
        self.cost_usd += cost_usd or 0.0
        self.usage.input_tokens += usage.input_tokens
        self.usage.output_tokens += usage.output_tokens
        self.usage.cache_read_tokens += usage.cache_read_tokens
        self.usage.cache_write_tokens += usage.cache_write_tokens

    def exceeded(self) -> str | None:
        """Reason the budget is spent, or None. Phrased so it can be shown to the agent or the user."""
        b = self.budget
        if self.steps >= b.max_steps:
            return f"step budget reached ({self.steps}/{b.max_steps} model calls)"
        if self.cost_usd >= b.max_cost_usd:
            return f"cost budget reached (${self.cost_usd:.2f}/${b.max_cost_usd:.2f})"
        elapsed = time.monotonic() - self.started
        if elapsed >= b.max_seconds:
            return f"time budget reached ({elapsed:.0f}s/{b.max_seconds:.0f}s)"
        return None
