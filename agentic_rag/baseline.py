from __future__ import annotations

import re
import time

from .knowledge_base import KnowledgeBase
from .models import ModelRuntime
from .types import AgentResult, AgentStep


class BaselineRAG:
    """Single-pass RAG baseline used only for controlled comparison."""

    def __init__(self, runtime: ModelRuntime, knowledge_base: KnowledgeBase):
        self.runtime = runtime
        self.kb = knowledge_base

    def run(self, question: str, history=None) -> AgentResult:
        started = time.perf_counter()
        self.runtime.reset_usage()
        hits = self.kb.search(question)
        if not hits:
            return AgentResult(
                "未检索到相关信息。", [], [AgentStep("单次检索", "无结果")], False,
                prompt_tokens=self.runtime.prompt_tokens,
                completion_tokens=self.runtime.completion_tokens,
            )
        evidence = "\n\n".join(
            f"[{index}] {hit.source} 第{hit.page}页\n{hit.content}"
            for index, hit in enumerate(hits, 1)
        )
        answer = self.runtime.generate(
            "仅依据证据回答并使用 [编号] 引用。",
            f"问题：{question}\n证据：\n{evidence}",
            1200,
        )
        citations = [int(value) for value in re.findall(r"\[(\d+)]", answer)]
        grounded = bool(citations) and all(1 <= value <= len(hits) for value in citations)
        return AgentResult(
            answer=answer,
            hits=hits,
            steps=[AgentStep("单次检索生成", f"使用 {len(hits)} 条证据")],
            grounded=grounded,
            latency_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=self.runtime.prompt_tokens,
            completion_tokens=self.runtime.completion_tokens,
        )
