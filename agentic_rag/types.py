from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchHit:
    content: str
    source: str
    page: int
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"{self.source}，第 {self.page} 页"

    @property
    def chunk_id(self) -> str | None:
        """Stable chunk record id, used to trace a step back to stored evidence."""
        value = self.metadata.get("chunk_id")
        return str(value) if value else None


@dataclass
class AgentStep:
    """One orchestration step.

    record_id/evidence make the step traceable: record_id identifies the step
    record written to runs/agent_runs.jsonl, evidence lists the chunk records
    (or citation numbers) the step actually relied on.
    """

    name: str
    detail: str
    status: str = "completed"
    record_id: str | None = None
    evidence: list[str] = field(default_factory=list)


@dataclass
class ToolCallRecord:
    tool: str
    success: bool
    latency_ms: float
    attempts: int
    error_type: str | None = None


@dataclass
class AgentResult:
    answer: str
    hits: list[SearchHit]
    steps: list[AgentStep]
    grounded: bool
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    latency_ms: float = 0.0
    context_chars: int = 0
    context_raw_chars: int = 0
    dropped_messages: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error_type: str | None = None
    run_id: str | None = None
