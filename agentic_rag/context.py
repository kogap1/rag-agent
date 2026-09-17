from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextSnapshot:
    text: str
    chars: int
    raw_chars: int
    kept_messages: int
    dropped_messages: int


class ContextManager:
    """Bounded conversation memory with deterministic structured compression."""

    def __init__(self, max_chars: int = 6000, max_messages: int = 6):
        self.max_chars = max_chars
        self.max_messages = max_messages

    def build(self, history: list[dict[str, str]]) -> ContextSnapshot:
        normalized = [
            {"role": item.get("role", "user"), "content": str(item.get("content", "")).strip()}
            for item in history
            if item.get("content")
        ]
        recent = normalized[-self.max_messages :]
        older = normalized[: -self.max_messages] if len(normalized) > self.max_messages else []

        parts: list[str] = []
        if older:
            topics = [item["content"].replace("\n", " ")[:80] for item in older if item["role"] == "user"]
            if topics:
                parts.append("更早对话主题（仅供指代消解）：" + "；".join(topics[-4:]))
        for item in recent:
            role = "用户" if item["role"] == "user" else "助手"
            parts.append(f"{role}：{item['content']}")

        text = "\n".join(parts)
        raw_chars = len(text)
        if len(text) > self.max_chars:
            text = text[-self.max_chars :]
            newline = text.find("\n")
            if newline > 0:
                text = text[newline + 1 :]
        return ContextSnapshot(
            text=text,
            chars=len(text),
            raw_chars=raw_chars,
            kept_messages=len(recent),
            dropped_messages=max(0, len(normalized) - len(recent)),
        )
