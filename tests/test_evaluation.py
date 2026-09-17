from agentic_rag.evaluation import EvalCase, evaluate, judge
from agentic_rag.types import AgentResult, SearchHit


def test_judge_requires_keyword_source_and_grounding():
    case = EvalCase("1", "问题", ["答案"], ["规则.pdf"], True)
    result = AgentResult(
        answer="这是答案 [1]",
        hits=[SearchHit("证据", "规则.pdf", 1, 1.0)],
        steps=[],
        grounded=True,
    )
    assert judge(case, result)["task_success"] is True


def test_judge_reports_retrieval_recall_and_reciprocal_rank():
    case = EvalCase("retrieval", "问题", [], ["规则.pdf", "附件.pdf"], True)
    result = AgentResult(
        answer="答案 [1]",
        hits=[
            SearchHit("无关", "其他.pdf", 1, 0.9),
            SearchHit("证据", "规则.pdf", 2, 0.8),
        ],
        steps=[],
        grounded=True,
    )
    row = judge(case, result)
    assert row["retrieval_recall_at_k"] == 0.5
    assert row["reciprocal_rank"] == 0.5


def test_judge_rejects_known_hallucination():
    case = EvalCase("2", "文档列表", ["规则.pdf"], [], False, ["OSDI"])
    result = AgentResult("- 规则.pdf\n- OSDI", [], [], True)
    judged = judge(case, result)
    assert judged["forbidden_pass"] is False
    assert judged["task_success"] is False


def test_no_answer_case_succeeds_when_abstained():
    case = EvalCase("nab-1", "出国政策？", [], [], False, [], [], True)
    result = AgentResult("当前知识库中没有找到足以回答该问题的证据。", [], [], False)
    row = judge(case, result)
    assert row["grounding_pass"] is True
    assert row["task_success"] is True


def test_no_answer_case_fails_when_hallucinated():
    case = EvalCase("nab-2", "出国政策？", [], [], False, ["资助"], [], True)
    result = AgentResult("我校提供每年3万元出国资助。", [SearchHit("证据", "规则.pdf", 1, 1.0)], [], True)
    row = judge(case, result)
    assert row["task_success"] is False


def test_history_is_passed_to_runner():
    captured = {}
    history = [{"role": "user", "content": "综合测评总分怎么计算？"}]

    class FakeRunner:
        def run(self, question, history=None):
            captured["history"] = history
            return AgentResult("答案 [1]", [SearchHit("证据", "规则.pdf", 1, 1.0)], [], True)

    case = EvalCase("coref-1", "那体育占比？", ["5%"], ["规则.pdf"], True, [], history)
    evaluate(FakeRunner(), [case])
    assert captured["history"] == history


def test_summary_reports_tokens_and_context_metrics():
    class FakeRunner:
        def run(self, question, history=None):
            return AgentResult(
                "答案 [1]",
                [SearchHit("证据", "规则.pdf", 1, 1.0)],
                [],
                True,
                prompt_tokens=100,
                completion_tokens=50,
                context_chars=200,
                context_raw_chars=1000,
            )

    report = evaluate(FakeRunner(), [EvalCase("x", "问题", ["答案"], ["规则.pdf"])], repeats=2)
    summary = report["summary"]
    assert summary["mean_prompt_tokens"] == 100
    assert summary["mean_completion_tokens"] == 50
    assert summary["mean_total_tokens"] == 150
    assert summary["tokens_per_success"] == 150
    assert summary["context_compression_ratio"] == 0.8  # 1 - 200/1000


def test_cost_reported_when_pricing_provided():
    class FakeRunner:
        def run(self, question, history=None):
            return AgentResult(
                "答案 [1]",
                [SearchHit("证据", "规则.pdf", 1, 1.0)],
                [],
                True,
                prompt_tokens=500_000,
                completion_tokens=500_000,
            )

    pricing = {"input_per_1m": 1.0, "output_per_1m": 3.0}
    report = evaluate(FakeRunner(), [EvalCase("x", "问题", ["答案"], ["规则.pdf"])], pricing=pricing)
    assert abs(report["summary"]["mean_cost_usd"] - 2.0) < 1e-9  # (0.5*1 + 0.5*3)
# RESUME_ALIGN_ACCEPTANCE


def test_judge_reports_retrieval_hit_and_citation_accuracy():
    case = EvalCase("hit", "问题", ["答案"], ["规则.pdf"], True)
    result = AgentResult(
        answer="这是答案 [1]，附注 [3]",
        hits=[SearchHit("证据", "规则.pdf", 1, 1.0)],
        steps=[],
        grounded=True,
    )

    row = judge(case, result)

    assert row["retrieval_hit"] is True
    assert row["citation_accuracy"] == 0.5


def test_summary_exposes_resume_level_metrics():
    class FakeRunner:
        def run(self, question, history=None):
            return AgentResult("答案 [1]", [SearchHit("证据", "规则.pdf", 1, 1.0)], [], True)

    report = evaluate(FakeRunner(), [EvalCase("x", "问题", ["答案"], ["规则.pdf"])])
    summary = report["summary"]

    assert summary["retrieval_hit_rate"] == 1.0
    assert summary["citation_accuracy"] == 1.0


def test_abstained_case_is_excluded_from_resume_level_metrics():
    row = judge(
        EvalCase("nab", "出国政策？", [], [], False, [], [], True),
        AgentResult("当前知识库中没有找到足以回答该问题的证据。", [], [], False),
    )

    assert row["retrieval_hit"] is None
    assert row["citation_accuracy"] is None
