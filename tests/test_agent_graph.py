from types import SimpleNamespace

from agentic_rag.agent import AgenticRAG
from agentic_rag.types import SearchHit


class FakeRuntime:
    prompt_tokens = 0
    completion_tokens = 0

    def reset_usage(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def generate(self, system, prompt, max_new_tokens):
        self.prompt_tokens += 10
        self.completion_tokens += 3
        if "只读工具规划器" in system:
            return '{"tool":"knowledge_search","arguments":{"query":"奖学金条件"}}'
        if "查询改写器" in system:
            return "奖学金 申请 条件"
        if "引用校验修复器" in system:
            return '{"claims":[{"text":"申请需要满足条件","citation":1}]}'
        return "申请需要满足条件 [1]"


class FakeKnowledgeBase:
    def __init__(self, settings, hits):
        self.settings = settings
        self.hits = hits
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return list(self.hits)

    def list_documents(self):
        return ["规则.pdf"]


def make_settings(tmp_path, max_agent_steps=2, max_retrieval_rounds=2):
    return SimpleNamespace(
        max_context_chars=6000,
        max_history_messages=6,
        tool_timeout_seconds=1,
        tool_max_retries=0,
        max_tool_output_chars=12000,
        max_agent_steps=max_agent_steps,
        max_retrieval_rounds=max_retrieval_rounds,
        run_log_path=tmp_path / "runs.jsonl",
    )


def test_langgraph_compiles_expected_nodes_and_returns_grounded_answer(tmp_path):
    settings = make_settings(tmp_path)
    kb = FakeKnowledgeBase(settings, [SearchHit("申请需要满足条件", "规则.pdf", 3, 0.9)])
    agent = AgenticRAG(FakeRuntime(), kb, settings)

    graph_nodes = set(agent.graph.get_graph().nodes)
    assert {
        "prepare_context",
        "plan_intent",
        "execute_tool",
        "rewrite_query",
        "generate_answer",
        "validate_citations",
        "repair_citations",
    }.issubset(graph_nodes)

    result = agent.run("奖学金怎么申请？", request_id="graph-test")

    assert result.grounded is True
    assert result.answer.endswith("[1]")
    assert result.hits[0].source == "规则.pdf"
    assert any("LangGraph" in step.detail for step in result.steps)


def test_langgraph_rewrites_once_then_refuses_without_evidence(tmp_path):
    settings = make_settings(tmp_path, max_agent_steps=2)
    kb = FakeKnowledgeBase(settings, [])
    agent = AgenticRAG(FakeRuntime(), kb, settings)

    result = agent.run("不存在的制度？")

    assert result.grounded is False
    assert "没有找到足以回答" in result.answer
    assert kb.queries == ["奖学金条件", "奖学金 申请 条件"]
    assert any(step.name == "查询改写" for step in result.steps)

# RESUME_ALIGN_ACCEPTANCE


def test_every_step_is_traceable_to_a_record(tmp_path):
    settings = make_settings(tmp_path)
    hit = SearchHit(
        "申请需要满足条件",
        "规则.pdf",
        3,
        0.9,
        {"chunk_id": "chunk-42"},
    )
    kb = FakeKnowledgeBase(settings, [hit])
    agent = AgenticRAG(FakeRuntime(), kb, settings)

    result = agent.run("奖学金怎么申请？", request_id="trace-test")

    assert result.run_id and result.run_id.startswith("run-")
    record_ids = [step.record_id for step in result.steps]
    assert all(record_id and record_id.startswith(result.run_id) for record_id in record_ids)
    assert len(set(record_ids)) == len(record_ids)

    retrieval_step = next(step for step in result.steps if step.name == "工具执行")
    assert retrieval_step.evidence == ["chunk-42"]

    validation_step = next(step for step in result.steps if step.name == "答案校验")
    assert validation_step.evidence == ["citation:1"]


def test_retrieval_rounds_are_bounded_by_configuration(tmp_path):
    settings = make_settings(tmp_path, max_retrieval_rounds=3)
    kb = FakeKnowledgeBase(settings, [])
    agent = AgenticRAG(FakeRuntime(), kb, settings)

    result = agent.run("不存在的制度？")

    # 三轮检索 + 两次改写后达到上限，明确拒答而不是继续空转。
    tool_steps = [step for step in result.steps if step.name == "工具执行"]
    rewrite_steps = [step for step in result.steps if step.name == "查询改写"]
    assert len(tool_steps) == 3
    assert len(rewrite_steps) == 2
    assert result.grounded is False
    assert any(step.name == "证据门控" for step in result.steps)
    assert [step.record_id for step in result.steps] == [
        f"{result.run_id}:step-{index}" for index in range(1, len(result.steps) + 1)
    ]
    assert any(step.name == "证据门控" for step in result.steps)
