from agentic_rag.tools import ToolRegistry, ToolSpec, ToolValidationError, validate_search


def test_registry_rejects_unknown_tool():
    registry = ToolRegistry(timeout_seconds=1, max_retries=0)
    try:
        registry.execute("delete_files", {})
    except ToolValidationError:
        return
    raise AssertionError("unknown tool should be rejected")


def test_read_only_tool_is_validated_cached_and_truncated():
    calls = []

    def search(query):
        calls.append(query)
        return "abcdefgh"

    registry = ToolRegistry(timeout_seconds=1, max_retries=0, max_output_chars=4)
    registry.register(ToolSpec("knowledge_search", search, validate_search))
    first, first_record = registry.execute("knowledge_search", {"query": "test"})
    second, second_record = registry.execute("knowledge_search", {"query": "test"})
    assert first == second == "abcd"
    assert calls == ["test"]
    assert first_record.attempts == 1
    assert second_record.attempts == 0

    registry.begin_run()
    third, third_record = registry.execute("knowledge_search", {"query": "test"})
    assert third == "abcd"
    assert calls == ["test", "test"]
    assert third_record.attempts == 1
