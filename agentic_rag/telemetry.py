from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .types import AgentResult


def append_run_log(
    path: Path,
    question: str,
    result: AgentResult,
    request_id: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "question": question,
        "success": bool(result.answer) and result.error_type is None,
        "grounded": result.grounded,
        "latency_ms": round(result.latency_ms, 2),
        "steps": len(result.steps),
        "context_chars": result.context_chars,
        "context_raw_chars": result.context_raw_chars,
        "dropped_messages": result.dropped_messages,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "error_type": result.error_type,
        "tool_calls": [asdict(item) for item in result.tool_calls],
        "sources": sorted({hit.source for hit in result.hits}),
        "run_id": result.run_id,
        # 每步的 record_id 与 evidence 落盘，使轨迹可以回溯到具体记录。
        "step_records": [
            {
                "record_id": step.record_id,
                "name": step.name,
                "status": step.status,
                "evidence": list(step.evidence),
            }
            for step in result.steps
        ],
        "evidence_records": [
            {
                "chunk_id": hit.chunk_id,
                "source": hit.source,
                "page": hit.page,
                "citation": hit.citation,
            }
            for hit in result.hits
        ],
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
