from __future__ import annotations

import argparse
import hashlib
import platform
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import torch

from agentic_rag import AgenticRAG, Settings
from agentic_rag.baseline import BaselineRAG
from agentic_rag.evaluation import evaluate, load_cases, write_report
from agentic_rag.knowledge_base import KnowledgeBase
from agentic_rag.models import ModelRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="运行可复现的 RAG/Agent 评测")
    parser.add_argument("--dataset", type=Path, default=Path("eval/dataset.example.jsonl"))
    parser.add_argument("--mode", choices=["agent", "baseline"], default="agent")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tag", type=str, default=None, help="输出文件名标签，如 reranker-on/reranker-off")
    parser.add_argument("--disable-reranker", action="store_true", help="关闭 Reranker，用于消融对比")
    args = parser.parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats 必须大于 0")

    settings = Settings.from_env()
    if args.disable_reranker:
        settings = replace(settings, enable_reranker=False)
    runtime = ModelRuntime(settings)
    kb = KnowledgeBase(settings, runtime)
    runner = AgenticRAG(runtime, kb, settings) if args.mode == "agent" else BaselineRAG(runtime, kb)
    pricing = None
    if settings.cost_per_1m_input_tokens or settings.cost_per_1m_output_tokens:
        pricing = {
            "input_per_1m": settings.cost_per_1m_input_tokens,
            "output_per_1m": settings.cost_per_1m_output_tokens,
        }
    report = evaluate(runner, load_cases(args.dataset), args.repeats, pricing)
    tag = args.tag or ("reranker-off" if args.disable_reranker else "reranker-on")
    output = args.output or Path("reports") / f"{args.mode}-{tag}-{datetime.now():%Y%m%d-%H%M%S}.json"
    metadata = {
        "mode": args.mode,
        "tag": tag,
        "dataset": str(args.dataset),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "model": settings.llm_model,
        "embedding": settings.embedding_model,
        "reranker": settings.reranker_model if settings.enable_reranker else "disabled",
        "device": runtime.device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "repeats": args.repeats,
        "cost_per_1m_input_tokens": settings.cost_per_1m_input_tokens,
        "cost_per_1m_output_tokens": settings.cost_per_1m_output_tokens,
    }
    write_report(output, report, metadata)
    print(f"评测完成：{output} / {output.with_suffix('.md')}")
    print(report["summary"])


if __name__ == "__main__":
    main()
