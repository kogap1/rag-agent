from agentic_rag.context import ContextManager


def test_context_is_bounded_and_reports_dropped_messages():
    history = [{"role": "user", "content": f"message-{index}-" + "x" * 50} for index in range(10)]
    snapshot = ContextManager(max_chars=180, max_messages=4).build(history)
    assert snapshot.chars <= 180
    assert snapshot.kept_messages == 4
    assert snapshot.dropped_messages == 6
    assert "message-9" in snapshot.text


def test_context_reports_raw_chars_before_compression():
    history = [{"role": "user", "content": f"message-{index}-" + "x" * 50} for index in range(10)]
    snapshot = ContextManager(max_chars=180, max_messages=4).build(history)
    assert snapshot.raw_chars > snapshot.chars
    assert snapshot.raw_chars > 0
