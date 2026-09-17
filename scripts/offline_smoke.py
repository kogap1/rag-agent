"""离线端到端冒烟：不下载任何模型权重，验证改造后的主链路。

覆盖：版本化入库 → LangGraph 问答编排 → 引用核对 → 每步可追溯 → 版本回退 → MCP 检索工具。
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document  # noqa: E402
from mcp import Client  # noqa: E402

import agentic_rag.knowledge_base as kb_module  # noqa: E402
from agentic_rag.agent import AgenticRAG  # noqa: E402
from agentic_rag.knowledge_base import KnowledgeBase  # noqa: E402
from agentic_rag.mcp_server import create_mcp_server  # noqa: E402

ANSWER = "综合测评按加权求和计算 [1]"


class OfflineRuntime:
    """确定性替身模型，避免评测/冒烟依赖真实权重。"""

    def __init__(self) -> None:
        self.reranker = None
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def reset_usage(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def generate(self, system: str, prompt: str, max_new_tokens: int = 256) -> str:
        self.prompt_tokens += 10
        self.completion_tokens += 5
        if "只读工具规划器" in system:
            question = prompt.rsplit("当前问题：", 1)[-1].strip()
            return json.dumps({"tool": "knowledge_search", "arguments": {"query": question}}, ensure_ascii=False)
        if "查询改写器" in system:
            return "综合测评 计算 办法"
        if "引用校验修复器" in system:
            return json.dumps({"claims": [{"text": "综合测评按加权求和计算", "citation": 1}]}, ensure_ascii=False)
        return ANSWER


class OfflineVectorStore:
    def __init__(self) -> None:
        self.rows: dict[str, Document] = {}

    def get(self, where=None, include=None):
        rows = list(self.rows.items())
        if where:
            rows = [
                (identifier, document)
                for identifier, document in rows
                if all(document.metadata.get(key) == value for key, value in where.items())
            ]
        return {
            "ids": [identifier for identifier, _ in rows],
            "metadatas": [document.metadata for _, document in rows],
        }

    def add_documents(self, documents, ids):
        self.rows.update(dict(zip(ids, documents)))

    def delete(self, ids):
        for identifier in ids:
            self.rows.pop(identifier, None)

    def similarity_search_with_relevance_scores(self, query, k, **kwargs):
        version_id = (kwargs.get("filter") or {}).get("version_id")
        rows = [doc for doc in self.rows.values() if doc.metadata.get("version_id") == version_id]
        return [(doc, 0.9) for doc in rows[:k]]


class OfflineKnowledgeBase(KnowledgeBase):
    def __init__(self, settings, runtime, store):
        self._store = store
        super().__init__(settings, runtime)

    @property
    def db(self):
        return self._store


def _settings(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        metadata_db_path=root / "state.sqlite3",
        ingestion_lease_seconds=60,
        chunk_size=200,
        chunk_overlap=20,
        recall_k=5,
        final_k=3,
        keep_history_versions=2,
        max_context_chars=6000,
        max_history_messages=6,
        tool_timeout_seconds=5,
        tool_max_retries=0,
        max_tool_output_chars=12000,
        max_agent_steps=2,
        max_retrieval_rounds=2,
        run_log_path=root / "agent_runs.jsonl",
    )


def _loader_factory(text: str):
    class OfflineLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            return [Document(page_content=text, metadata={"page": 2})]

    return OfflineLoader


def _check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    if not condition:
        raise SystemExit(f"离线冒烟失败：{label}")


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        store = OfflineVectorStore()
        runtime = OfflineRuntime()
        settings = _settings(root)
        kb_module.PyPDFLoader = _loader_factory("综合测评按加权求和计算。")
        kb = OfflineKnowledgeBase(settings, runtime, store)

        first = root / "v1.pdf"
        first.write_bytes(b"%PDF-v1")
        second = root / "v2.pdf"
        second.write_bytes(b"%PDF-v2")

        print("== 1. 版本化入库")
        _check("首次入库写入分块", kb.ingest(first, source_name="测评细则.pdf") == 1)
        first_version = kb.catalog.active_version_ids()[0]
        _check("重复入库被唯一约束拦下", kb.ingest(first, source_name="测评细则.pdf") == 0)
        _check("新版本入库并切换 active", kb.ingest(second, source_name="测评细则.pdf") == 1)
        second_version = kb.catalog.active_version_ids()[0]
        _check("active 已切到新版本", second_version != first_version, second_version[:12])
        _check("旧版本仍在回退窗口内", [item.version_id for item in kb.history_versions("测评细则.pdf")][1] == first_version)

        print("== 2. LangGraph 问答编排与引用核对")
        agent = AgenticRAG(runtime, kb, settings)
        result = agent.run("综合测评怎么算？", request_id="smoke-1")
        _check("回答通过引用校验", result.grounded is True, result.answer)
        _check("证据带来源页码", result.hits[0].citation.endswith("第 3 页"), result.hits[0].citation)
        _check("检索步骤可回溯到 chunk 记录", any(step.evidence for step in result.steps))

        print("== 3. 每步可回溯到具体记录")
        record_ids = [step.record_id for step in result.steps]
        _check("所有步骤都有 record_id", all(record_ids))
        _check("record_id 挂在本轮 run_id 下", all(rid.startswith(result.run_id) for rid in record_ids))
        log_lines = settings.run_log_path.read_text(encoding="utf-8").strip().splitlines()
        record = json.loads(log_lines[-1])
        _check("运行日志写入 step_records", bool(record["step_records"]))
        _check("运行日志写入 evidence_records", record["evidence_records"][0]["page"] == 3)

        print("== 4. 升级失败后回退到旧版本")
        restored = kb.rollback("测评细则.pdf")
        _check("回退到上一个已激活版本", restored.version_id == first_version)
        _check("回退后查询命中旧版本向量", kb.search("综合测评")[0].metadata["version_id"] == first_version)

        print("== 5. MCP 只读检索工具")
        captured = {}

        def search_handler(query, top_k=None):
            captured["query"] = query
            hits = kb.search(query)
            return [
                {
                    "citation": hit.citation,
                    "chunk_id": hit.chunk_id,
                    "content": hit.content,
                    "score": hit.score,
                    "source": hit.source,
                    "page": hit.page,
                }
                for hit in (hits[:top_k] if top_k else hits)
            ]

        def query_handler(question, history, request_id):
            return AgenticRAG(runtime, kb, settings).run(question, history, request_id=request_id)

        server = create_mcp_server(
            query_handler,
            search_handler,
            lambda: [item.__dict__ for item in kb.document_records()],
            lambda: {"status": "ready", "catalog": kb.health()},
        )

        async def list_mcp_tools() -> list[str]:
            async with Client(server) as client:
                tools = await client.list_tools()
                return sorted(tool.name for tool in tools.tools)

        tool_names = asyncio.run(list_mcp_tools())
        _check(
            "MCP 暴露检索工具",
            "search_knowledge_base" in tool_names,
            ", ".join(tool_names),
        )

    print("\n离线冒烟全部通过。")


if __name__ == "__main__":
    main()
