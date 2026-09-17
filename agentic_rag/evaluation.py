from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .types import AgentResult


class Runner(Protocol):
    def run(self, question: str, history=None) -> AgentResult: ...


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    expected_keywords: list[str]
    expected_sources: list[str]
    require_grounding: bool = True
    forbidden_keywords: list[str] = field(default_factory=list)
    history: list[dict[str, str]] = field(default_factory=list)
    expect_no_answer: bool = False
    note: str = ""


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            data = json.loads(line)
            try:
                cases.append(EvalCase(**data))
            except TypeError as exc:
                raise ValueError(f"评测集第 {line_number} 行字段错误：{exc}") from exc
    if not cases:
        raise ValueError("评测集不能为空")
    return cases


def citation_accuracy(answer: str, hit_count: int) -> float | None:
    """Share of citation numbers that map to a real record of this round.

    Returns None when the answer carries no citation at all, so no-answer cases
    are excluded from the aggregate instead of dragging it to zero.
    """
    if "原始证据（只能使用这些事实）" in answer:
        return 0.0
    citations = [int(value) for value in re.findall(r"\[(\d+)]", answer)]
    if not citations:
        return None
    return sum(1 for value in citations if 1 <= value <= hit_count) / len(citations)


def judge(case: EvalCase, result: AgentResult) -> dict:
    answer_lower = result.answer.lower()
    keyword_pass = all(item.lower() in answer_lower for item in case.expected_keywords)
    retrieved_sources = {hit.source for hit in result.hits}
    source_pass = all(any(expected in actual for actual in retrieved_sources) for expected in case.expected_sources)
    matched_sources = {
        expected
        for expected in case.expected_sources
        if any(expected in actual for actual in retrieved_sources)
    }
    retrieval_recall_at_k = (
        len(matched_sources) / len(case.expected_sources) if case.expected_sources else None
    )
    first_relevant_rank = next(
        (
            rank
            for rank, hit in enumerate(result.hits, 1)
            if any(expected in hit.source for expected in case.expected_sources)
        ),
        None,
    )
    reciprocal_rank = (1 / first_relevant_rank) if first_relevant_rank else (0.0 if case.expected_sources else None)
    retrieval_hit = bool(matched_sources) if case.expected_sources else None
    answer_citation_accuracy = (
        None if case.expect_no_answer else citation_accuracy(result.answer, len(result.hits))
    )
    forbidden_pass = not any(item.lower() in answer_lower for item in case.forbidden_keywords)
    if case.expect_no_answer:
        abstained = not result.grounded and bool(result.answer) and not result.error_type
        grounding_pass = abstained
        task_success = abstained and keyword_pass and source_pass and forbidden_pass
    else:
        grounding_pass = result.grounded if case.require_grounding else True
        task_success = keyword_pass and source_pass and grounding_pass and forbidden_pass and not result.error_type
    return {
        "id": case.id,
        "question": case.question,
        "task_success": task_success,
        "keyword_pass": keyword_pass,
        "source_pass": source_pass,
        "retrieval_recall_at_k": retrieval_recall_at_k,
        "retrieval_hit": retrieval_hit,
        "reciprocal_rank": reciprocal_rank,
        "citation_accuracy": answer_citation_accuracy,
        "grounding_pass": grounding_pass,
        "forbidden_pass": forbidden_pass,
        "latency_ms": round(result.latency_ms, 2),
        "steps": len(result.steps),
        "tool_success": all(call.success for call in result.tool_calls),
        "recovered": any(step.status == "recovered" for step in result.steps),
        "policy_violation": any(call.tool not in {"knowledge_search", "list_documents"} for call in result.tool_calls),
        "error_type": result.error_type,
        "answer": result.answer,
        "sources": sorted(retrieved_sources),
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "context_chars": result.context_chars,
        "context_raw_chars": result.context_raw_chars,
        "dropped_messages": result.dropped_messages,
    }


def evaluate(runner: Runner, cases: list[EvalCase], repeats: int = 1, pricing: dict | None = None) -> dict:
    """Run cases against a runner and aggregate metrics.

    pricing: optional {"input_per_1m": float, "output_per_1m": float} USD per
    1M tokens to also report dollar cost; omitted for local models where cost is 0.
    """
    rows: list[dict] = []
    for case in cases:
        for repeat in range(repeats):
            row = judge(case, runner.run(case.question, case.history))
            row["repeat"] = repeat + 1
            rows.append(row)
    count = len(rows)
    latencies = [row["latency_ms"] for row in rows]
    prompt_tokens = [row["prompt_tokens"] for row in rows]
    completion_tokens = [row["completion_tokens"] for row in rows]
    total_tokens = [p + c for p, c in zip(prompt_tokens, completion_tokens)]
    context_chars = [row["context_chars"] for row in rows]
    context_raw = [row["context_raw_chars"] for row in rows]
    successful_rows = [row for row in rows if row["task_success"]]
    retrieval_rows = [row for row in rows if row["retrieval_recall_at_k"] is not None]
    retrieval_hit_rows = [row for row in rows if row["retrieval_hit"] is not None]
    citation_rows = [row for row in rows if row["citation_accuracy"] is not None]
    successful_by_case = {
        case.id: [row["task_success"] for row in rows if row["id"] == case.id]
        for case in cases
    }
    summary = {
        "runs": count,
        "cases": len(cases),
        "repeats": repeats,
        "task_success_rate": sum(row["task_success"] for row in rows) / count,
        "grounding_rate": sum(row["grounding_pass"] for row in rows) / count,
        "tool_success_rate": sum(row["tool_success"] for row in rows) / count,
        "recovery_trigger_rate": sum(row["recovered"] for row in rows) / count,
        "recovery_success_rate": (
            sum(row["recovered"] and row["task_success"] for row in rows)
            / max(1, sum(row["recovered"] for row in rows))
        ),
        "policy_violation_rate": sum(row["policy_violation"] for row in rows) / count,
        "mean_steps": statistics.mean(row["steps"] for row in rows),
        "mean_latency_ms": statistics.mean(latencies),
        "p95_latency_ms": sorted(latencies)[max(0, math.ceil(count * 0.95) - 1)],
        "stability_at_n": sum(all(values) for values in successful_by_case.values()) / len(cases),
        "retrieval_recall_at_k": (
            statistics.mean(row["retrieval_recall_at_k"] for row in retrieval_rows)
            if retrieval_rows
            else None
        ),
        "mrr": (
            statistics.mean(row["reciprocal_rank"] for row in retrieval_rows)
            if retrieval_rows
            else None
        ),
        # 固定问题集上的「检索命中率」与「引用准确性」。
        "retrieval_hit_rate": (
            sum(row["retrieval_hit"] for row in retrieval_hit_rows) / len(retrieval_hit_rows)
            if retrieval_hit_rows
            else None
        ),
        "citation_accuracy": (
            statistics.mean(row["citation_accuracy"] for row in citation_rows)
            if citation_rows
            else None
        ),
        "mean_prompt_tokens": statistics.mean(prompt_tokens),
        "mean_completion_tokens": statistics.mean(completion_tokens),
        "mean_total_tokens": statistics.mean(total_tokens),
        "tokens_per_success": (
            statistics.mean([row["prompt_tokens"] + row["completion_tokens"] for row in successful_rows])
            if successful_rows
            else 0.0
        ),
        "mean_context_chars": statistics.mean(context_chars),
        "mean_context_raw_chars": statistics.mean(context_raw),
        "context_compression_ratio": (
            1 - sum(context_chars) / sum(context_raw) if sum(context_raw) > 0 else 0.0
        ),
    }
    if pricing:
        mean_cost = (
            summary["mean_prompt_tokens"] * pricing.get("input_per_1m", 0.0)
            + summary["mean_completion_tokens"] * pricing.get("output_per_1m", 0.0)
        ) / 1_000_000
        summary["mean_cost_usd"] = mean_cost
        summary["cost_per_success_usd"] = (
            mean_cost * count / len(successful_rows) if successful_rows else None
        )
    return {"summary": summary, "results": rows}


def write_report(path: Path, report: dict, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, **report}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = report["summary"]
    markdown = [
        "# RAG 知识库问答 Agent 评测报告",
        "",
        "> 本文件由真实运行自动生成；未运行评测前不应引用其中的提升百分比。",
        "> 检索命中率与引用准确性为本项目固定问题集的核心验收口径。",
        "",
        "## 实验条件",
        "",
        *[f"- {key}: `{value}`" for key, value in metadata.items()],
        "",
        "## 汇总指标",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| Task Success Rate | {summary['task_success_rate']:.2%} |",
        f"| Grounding Rate | {summary['grounding_rate']:.2%} |",
        f"| Tool Success Rate | {summary['tool_success_rate']:.2%} |",
        f"| Recovery Success Rate | {summary['recovery_success_rate']:.2%} |",
        f"| Policy Violation Rate | {summary['policy_violation_rate']:.2%} |",
        f"| Stability@N | {summary['stability_at_n']:.2%} |",
        "| 检索命中率 Retrieval Hit Rate | "
        + (
            f"{summary['retrieval_hit_rate']:.2%}"
            if summary["retrieval_hit_rate"] is not None
            else "N/A"
        )
        + " |",
        "| 引用准确性 Citation Accuracy | "
        + (
            f"{summary['citation_accuracy']:.2%}"
            if summary["citation_accuracy"] is not None
            else "N/A"
        )
        + " |",
        "| Retrieval Recall@K | "
        + (
            f"{summary['retrieval_recall_at_k']:.2%}"
            if summary["retrieval_recall_at_k"] is not None
            else "N/A"
        )
        + " |",
        "| MRR | "
        + (f"{summary['mrr']:.4f}" if summary["mrr"] is not None else "N/A")
        + " |",
        f"| Mean Steps | {summary['mean_steps']:.2f} |",
        f"| Mean Latency | {summary['mean_latency_ms']:.0f} ms |",
        f"| P95 Latency | {summary['p95_latency_ms']:.0f} ms |",
        f"| Mean Prompt Tokens | {summary['mean_prompt_tokens']:.0f} |",
        f"| Mean Completion Tokens | {summary['mean_completion_tokens']:.0f} |",
        f"| Mean Total Tokens | {summary['mean_total_tokens']:.0f} |",
        f"| Tokens per Success | {summary['tokens_per_success']:.0f} |",
        f"| Mean Context Chars | {summary['mean_context_chars']:.0f} |",
        f"| Mean Raw Context Chars | {summary['mean_context_raw_chars']:.0f} |",
        f"| Context Compression Ratio | {summary['context_compression_ratio']:.1%} |",
    ]
    if "mean_cost_usd" in summary:
        markdown.extend(
            [
                "",
                "## 成本（按配置的 API 单价估算）",
                "",
                "| 指标 | 结果 |",
                "|---|---:|",
                f"| Mean Cost / Task | ${summary['mean_cost_usd']:.6f} |",
                "| Cost / Successful Task | "
                + (
                    f"${summary['cost_per_success_usd']:.6f}"
                    if summary["cost_per_success_usd"] is not None
                    else "无成功任务"
                )
                + " |",
            ]
        )
    path.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
