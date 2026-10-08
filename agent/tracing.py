"""Tracing (DESIGN §9.5): one JSONL record per agent step under traces/<run_id>.jsonl.

Traces feed debugging, evals, the recipe library and future fine-tuning, so record exact model ids
and token usage on every step. Records are flushed per line so a crashed run still leaves a trace.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


@dataclass
class TraceRecord:
    run_id: str
    stage: str
    step: int
    model: str | None = None
    request_id: str | None = None
    stop_reason: str | None = None
    usage: dict[str, int] | None = None
    cost_usd: float | None = None
    latency_s: float | None = None
    text: str | None = None  # assistant text this step (not thinking)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # [{name, input, result, is_error}]
    render_paths: list[str] = field(default_factory=list)
    scores: dict[str, Any] | None = None  # critic output, when this step is a critique
    note: str | None = None  # anything else (budget stops, errors)
    header: dict[str, Any] | None = None  # step 0 only: prompt, references and run config (agent.run)
    ts: float = field(default_factory=time.time)


class Tracer:
    def __init__(self, run_id: str | None = None, root: Path | str = "traces") -> None:
        self.run_id = run_id or new_run_id()
        self.path = Path(root) / f"{self.run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: TraceRecord) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), default=_jsonable) + "\n")


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "model_dump"):  # pydantic / SDK objects
        return obj.model_dump()
    return str(obj)
