import json
from pathlib import Path

from agentic_rag.telemetry import append_run_log
from agentic_rag.types import AgentResult, AgentStep, SearchHit


def test_run_log_keeps_request_id_for_http_correlation(tmp_path: Path):
    path = tmp_path / "runs.jsonl"
    append_run_log(path, "问题", AgentResult("拒答", [], [], False), "request-123")
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["request_id"] == "request-123"
    assert record["question"] == "问题"
# RESUME_ALIGN_ACCEPTANCE


def test_run_log_keeps_traceable_step_and_evidence_records(tmp_path: Path):
    path = tmp_path / "runs.jsonl"
    result = AgentResult(
        answer="答案 [1]",
        hits=[SearchHit("证据", "规则.pdf", 2, 0.9, {"chunk_id": "chunk-1"})],
        steps=[AgentStep("工具执行", "knowledge_search 成功", "completed", "run-abc:step-3", ["chunk-1"])],
        grounded=True,
        run_id="run-abc",
    )

    append_run_log(path, "问题", result, "request-9")
    record = json.loads(path.read_text(encoding="utf-8"))

    assert record["run_id"] == "run-abc"
    assert record["step_records"] == [
        {
            "record_id": "run-abc:step-3",
            "name": "工具执行",
            "status": "completed",
            "evidence": ["chunk-1"],
        }
    ]
    assert record["evidence_records"][0]["chunk_id"] == "chunk-1"
