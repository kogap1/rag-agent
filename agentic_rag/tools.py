from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Callable

from .types import ToolCallRecord


class ToolError(RuntimeError):
    retriable = False


class ToolValidationError(ToolError):
    pass


class ToolTimeoutError(ToolError):
    retriable = True


class ToolExecutionError(ToolError):
    retriable = True


@dataclass(frozen=True)
class ToolSpec:
    name: str
    handler: Callable[..., Any]
    validator: Callable[[dict[str, Any]], dict[str, Any]]
    read_only: bool = True


class ToolRegistry:
    """Allowlisted, validated and observable tool execution boundary."""

    def __init__(self, timeout_seconds: float, max_retries: int, max_output_chars: int = 12000):
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_output_chars = max_output_chars
        self._tools: dict[str, ToolSpec] = {}
        self._cache: dict[str, Any] = {}

    def register(self, spec: ToolSpec) -> None:
        if not spec.read_only:
            raise ValueError("Agent 主链路只允许注册只读工具")
        self._tools[spec.name] = spec

    @property
    def allowed_tools(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def begin_run(self) -> None:
        """Keep idempotency caching inside one Agent run, never across KB versions."""
        self._cache.clear()

    def execute(self, name: str, arguments: dict[str, Any]) -> tuple[Any, ToolCallRecord]:
        if name not in self._tools:
            raise ToolValidationError(f"工具不在白名单中：{name}")
        spec = self._tools[name]
        clean = spec.validator(arguments)
        cache_key = hashlib.sha256(
            json.dumps([name, clean], ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key], ToolCallRecord(name, True, 0.0, 0)

        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"tool-{name}")
            future = executor.submit(spec.handler, **clean)
            try:
                value = future.result(timeout=self.timeout_seconds)
                executor.shutdown(wait=False, cancel_futures=True)
                value = self._truncate(value)
                self._cache[cache_key] = value
                latency = (time.perf_counter() - started) * 1000
                return value, ToolCallRecord(name, True, latency, attempt)
            except FutureTimeout as exc:
                future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                last_error = ToolTimeoutError(f"{name} 超过 {self.timeout_seconds:.0f}s")
            except Exception as exc:  # external model/vector DB errors are recoverable once
                executor.shutdown(wait=False, cancel_futures=True)
                last_error = ToolExecutionError(f"{name} 执行失败：{exc}")
            if attempt <= self.max_retries:
                time.sleep(min(0.5 * attempt, 1.0))
        assert last_error is not None
        latency = (time.perf_counter() - started) * 1000
        setattr(last_error, "record", ToolCallRecord(name, False, latency, self.max_retries + 1, type(last_error).__name__))
        raise last_error

    def _truncate(self, value: Any) -> Any:
        if isinstance(value, str):
            return value[: self.max_output_chars]
        if isinstance(value, list):
            kept: list[Any] = []
            used = 0
            for item in value:
                size = len(getattr(item, "content", str(item)))
                if kept and used + size > self.max_output_chars:
                    break
                kept.append(item)
                used += size
            return kept
        return value


def validate_search(arguments: dict[str, Any]) -> dict[str, Any]:
    query = str(arguments.get("query", "")).strip()
    if not query or len(query) > 500:
        raise ToolValidationError("query 长度必须在 1~500 字符之间")
    if any(ord(char) < 32 and char not in "\n\t" for char in query):
        raise ToolValidationError("query 含非法控制字符")
    return {"query": query}


def validate_no_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    if arguments:
        raise ToolValidationError("该工具不接受参数")
    return {}
