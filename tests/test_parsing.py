from agentic_rag.parsing import extract_json_object, render_cited_claims


def test_extract_plain_json():
    assert extract_json_object('{"action":"search","query":"奖学金"}')["query"] == "奖学金"


def test_extract_fenced_json():
    result = extract_json_object('说明如下\n```json\n{"action":"list_documents","query":""}\n```')
    assert result["action"] == "list_documents"


def test_invalid_output_returns_empty_dict():
    assert extract_json_object("没有 JSON") == {}


def test_render_cited_claims_rejects_out_of_range_and_evidence_echo():
    assert render_cited_claims({"claims": [{"text": "结论", "citation": 1}]}, 2) == "- 结论 [1]"
    assert render_cited_claims({"claims": [{"text": "结论", "citation": 3}]}, 2) is None
    assert render_cited_claims({"claims": [{"text": "原始证据如下", "citation": 1}]}, 2) is None
