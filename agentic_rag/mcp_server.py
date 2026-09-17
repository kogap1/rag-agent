from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from .types import AgentResult


QueryHandler = Callable[[str, list[dict[str, str]], str], AgentResult]
SearchHandler = Callable[[str, int | None], list[dict[str, Any]]]
DocumentsHandler = Callable[[], list[dict[str, Any]]]
HealthHandler = Callable[[], dict[str, Any]]


def _normalize_history(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    if not history:
        return []
    if len(history) > 20:
        raise ValueError("history 最多包含 20 条消息")
    normalized: list[dict[str, str]] = []
    for item in history:
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role not in {"user", "assistant"}:
            raise ValueError("history.role 只能是 user 或 assistant")
        if not content or len(content) > 4000:
            raise ValueError("history.content 长度必须在 1~4000 字符之间")
        normalized.append({"role": role, "content": content})
    return normalized


def _result_payload(result: AgentResult, request_id: str) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "answer": result.answer,
        "grounded": result.grounded,
        "latency_ms": result.latency_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "evidence": [
            {
                "citation": hit.citation,
                "content": hit.content,
                "score": hit.score,
                "source": hit.source,
                "page": hit.page,
            }
            for hit in result.hits
        ],
        "trace": [step.__dict__ for step in result.steps],
    }


def create_mcp_server(
    query_handler: QueryHandler,
    search_handler: SearchHandler,
    documents_handler: DocumentsHandler,
    health_handler: HealthHandler,
) -> MCPServer:
    """Create the read-only MCP boundary over the same service used by FastAPI."""

    server = MCPServer(
        name="agentic-rag-knowledge-base",
        title="Agentic RAG Knowledge Base",
        version="3.0.0",
        instructions=(
            "Use search_knowledge_base to retrieve ranked evidence without generating an answer. "
            "Use query_knowledge_base for evidence-grounded answers. "
            "Use list_knowledge_documents before asking about corpus coverage. "
            "All tools are read-only; ingestion, version switch and rollback remain authenticated FastAPI operations."
        ),
    )
    read_only = ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )

    @server.tool(
        name="query_knowledge_base",
        title="Query knowledge base",
        description="Answer a question from indexed evidence and return citations plus the LangGraph trace.",
        annotations=read_only,
        structured_output=True,
    )
    async def query_knowledge_base(
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        clean_question = question.strip()
        if not clean_question or len(clean_question) > 500:
            raise ValueError("question 长度必须在 1~500 字符之间")
        clean_history = _normalize_history(history)
        request_id = f"mcp-{uuid4().hex}"
        result = await asyncio.to_thread(
            query_handler,
            clean_question,
            clean_history,
            request_id,
        )
        return _result_payload(result, request_id)

    @server.tool(
        name="search_knowledge_base",
        title="Search knowledge base",
        description=(
            "Retrieve reranked evidence chunks with source, page and citation for a query, "
            "without calling the answer model."
        ),
        annotations=read_only,
        structured_output=True,
    )
    async def search_knowledge_base(
        query: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        clean_query = query.strip()
        if not clean_query or len(clean_query) > 500:
            raise ValueError("query 长度必须在 1~500 字符之间")
        if top_k is not None and not 1 <= top_k <= 50:
            raise ValueError("top_k 必须在 1~50 之间")
        evidence = await asyncio.to_thread(search_handler, clean_query, top_k)
        return {"query": clean_query, "evidence": evidence}

    @server.tool(
        name="list_knowledge_documents",
        title="List knowledge documents",
        description="List indexed documents, active versions, ingestion status, and chunk counts.",
        annotations=read_only,
        structured_output=True,
    )
    async def list_knowledge_documents() -> dict[str, Any]:
        documents = await asyncio.to_thread(documents_handler)
        return {"documents": documents}

    @server.tool(
        name="knowledge_base_health",
        title="Knowledge base health",
        description="Read SQLite catalog integrity and active/building/failed version counts.",
        annotations=read_only,
        structured_output=True,
    )
    async def knowledge_base_health() -> dict[str, Any]:
        return await asyncio.to_thread(health_handler)

    return server
