import asyncio

from mcp import Client

from agentic_rag.mcp_server import create_mcp_server
from agentic_rag.types import AgentResult, AgentStep, SearchHit


def test_mcp_lists_and_calls_read_only_tools():
    captured = {}

    def query_handler(question, history, request_id):
        captured.update(question=question, history=history, request_id=request_id)
        return AgentResult(
            answer="答案 [1]",
            hits=[SearchHit("证据", "规则.pdf", 2, 0.9)],
            steps=[AgentStep("答案校验", "引用编号校验通过")],
            grounded=True,
            latency_ms=12.5,
        )

    search_calls = {}

    def search_handler(query, top_k=None):
        search_calls.update(query=query, top_k=top_k)
        return [
            {
                "citation": "规则.pdf，第 2 页",
                "chunk_id": "chunk-1",
                "content": "证据",
                "score": 0.9,
                "source": "规则.pdf",
                "page": 2,
            }
        ]

    server = create_mcp_server(
        query_handler,
        search_handler,
        lambda: [{"source": "规则.pdf", "status": "active", "chunk_count": 8}],
        lambda: {"status": "ready", "catalog": {"integrity": "ok"}},
    )

    async def scenario():
        async with Client(server) as client:
            tools = await client.list_tools()
            assert {tool.name for tool in tools.tools} == {
                "search_knowledge_base",
                "query_knowledge_base",
                "list_knowledge_documents",
                "knowledge_base_health",
            }

            search = await client.call_tool(
                "search_knowledge_base",
                {"query": "规则是什么？", "top_k": 1},
            )
            assert search.structured_content["evidence"][0]["chunk_id"] == "chunk-1"

            query = await client.call_tool(
                "query_knowledge_base",
                {
                    "question": "规则是什么？",
                    "history": [{"role": "user", "content": "先看规则"}],
                },
            )
            assert query.structured_content["answer"] == "答案 [1]"
            assert query.structured_content["evidence"][0]["page"] == 2

            documents = await client.call_tool("list_knowledge_documents", {})
            assert documents.structured_content["documents"][0]["status"] == "active"

            health = await client.call_tool("knowledge_base_health", {})
            assert health.structured_content["catalog"]["integrity"] == "ok"

    asyncio.run(scenario())
    assert captured["question"] == "规则是什么？"
    assert captured["history"][0]["role"] == "user"
    assert captured["request_id"].startswith("mcp-")
    assert search_calls == {"query": "规则是什么？", "top_k": 1}
