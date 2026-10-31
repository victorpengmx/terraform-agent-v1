"""Eval configs (`eval/configs/*.yaml`, DESIGN §16): which prompts, seeds and conditions an experiment runs, with
which models, render level and per-run budgets.

A condition is one way of running the agent (critic on or off, SPEC on or off). Every condition runs every
selected (prompt, seed) pair with the same canvas seed, so runs pair up across conditions and configs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml

from agent.budget import Budget
from agent.models import ModelConfig
from eval.benchmark import BenchmarkEntry
from terrain.schema import Strict

DEFAULT_JUDGES = ("gemini-3.1-flash-lite", "claude-opus-5-5")  # the first whose adapter is available


class BudgetConfig(Strict):
    max_steps: int = Budget.max_steps
    max_cost_usd: float = 1.00
    max_seconds: float = Budget.max_seconds


class BudgetOverride(Strict):
    max_steps: int | None = None
    max_cost_usd: float | None = None
    max_seconds: float | None = None


class Condition(Strict):
    critic: bool
    rounds: int = 3
    use_spec: bool = True
    budget: BudgetOverride | None = None  # fields set here replace the config's budget for this condition


class Models(Strict):
    builder: str = "claude-haiku-5-5"
    critic: str = "claude-opus-5-5"
    judge: str | None = None  # None: default_judge()


class EvalConfig(Strict):
    name: str
    prompts: Literal["all", "spec_ablation", "regression"] | list[str] = "all"
    seeds: list[int] = [1, 2]
    models: Models = Models()
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"  # builder and critic
    judge_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    render: Literal["l1", "l2", "none"] = "l2"
    size: tuple[int, int] = (256, 256)
    base_y: float = 64.0
    compare_both_orders: bool = True  # DESIGN §11.4: for eval runs
    budget: BudgetConfig = BudgetConfig()
    conditions: dict[str, Condition]

    def select(self, entries: list[BenchmarkEntry]) -> list[BenchmarkEntry]:
        """The benchmark entries this config runs, in benchmark order."""
        if isinstance(self.prompts, list):
            known = {e.id for e in entries}
            if unknown := [p for p in self.prompts if p not in known]:
                raise ValueError(f"config {self.name}: unknown prompt ids {unknown}")
            return [e for e in entries if e.id in self.prompts]
        if self.prompts == "all":
            return list(entries)
        return [e for e in entries if getattr(e, self.prompts)]

    def budget_for(self, condition: str) -> Budget:
        base = self.budget.model_dump()
        override = self.conditions[condition].budget
        if override is not None:
            base |= {k: v for k, v in override.model_dump().items() if v is not None}
        return Budget(**base)

    @property
    def judge_model(self) -> str:
        return self.models.judge or default_judge()


def default_judge() -> str:
    """A Gemini judge when the gemini extra is installed (a different family from the Claude builder and critic),
    else Claude Opus; reports then mark the judge as same-family."""
    try:
        import google.genai  # noqa: F401
    except ImportError:
        return DEFAULT_JUDGES[1]
    return DEFAULT_JUDGES[0]


def same_family(judge: str, *others: str | None) -> bool:
    provider = ModelConfig(model=judge).provider
    return any(o is not None and ModelConfig(model=o).provider == provider for o in others)


def load_config(path: Path) -> EvalConfig:
    try:
        return EvalConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    except Exception as e:
        raise ValueError(f"{path}: {e}") from None
