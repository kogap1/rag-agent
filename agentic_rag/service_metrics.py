from __future__ import annotations

import threading
from collections import Counter


class ServiceMetrics:
    """Small dependency-free Prometheus exporter for the HTTP boundary."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests = Counter()
        self._errors = Counter()
        self._latency_sum_ms = 0.0
        self._latency_count = 0
        self._active = 0

    def request_started(self) -> None:
        with self._lock:
            self._active += 1

    def request_finished(self, endpoint: str, status_code: int, latency_ms: float) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)
            self._requests[(endpoint, str(status_code))] += 1
            self._latency_sum_ms += latency_ms
            self._latency_count += 1

    def record_error(self, error_type: str) -> None:
        with self._lock:
            self._errors[error_type] += 1

    def render_prometheus(self) -> str:
        with self._lock:
            request_rows = list(self._requests.items())
            error_rows = list(self._errors.items())
            latency_sum = self._latency_sum_ms / 1000
            latency_count = self._latency_count
            active = self._active

        lines = [
            "# HELP agentic_rag_http_requests_total HTTP requests by endpoint and status.",
            "# TYPE agentic_rag_http_requests_total counter",
        ]
        for (endpoint, status), count in sorted(request_rows):
            lines.append(
                f'agentic_rag_http_requests_total{{endpoint="{endpoint}",status="{status}"}} {count}'
            )
        lines.extend(
            [
                "# HELP agentic_rag_http_request_duration_seconds_sum Total HTTP request duration.",
                "# TYPE agentic_rag_http_request_duration_seconds_sum counter",
                f"agentic_rag_http_request_duration_seconds_sum {latency_sum:.6f}",
                "# HELP agentic_rag_http_request_duration_seconds_count Number of timed HTTP requests.",
                "# TYPE agentic_rag_http_request_duration_seconds_count counter",
                f"agentic_rag_http_request_duration_seconds_count {latency_count}",
                "# HELP agentic_rag_active_requests Current in-flight HTTP requests.",
                "# TYPE agentic_rag_active_requests gauge",
                f"agentic_rag_active_requests {active}",
                "# HELP agentic_rag_errors_total Service errors by exception type.",
                "# TYPE agentic_rag_errors_total counter",
            ]
        )
        for error_type, count in sorted(error_rows):
            lines.append(f'agentic_rag_errors_total{{type="{error_type}"}} {count}')
        return "\n".join(lines) + "\n"

