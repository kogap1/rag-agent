from agentic_rag.service_metrics import ServiceMetrics


def test_prometheus_metrics_include_status_latency_and_errors():
    metrics = ServiceMetrics()
    metrics.request_started()
    metrics.record_error("ValueError")
    metrics.request_finished("/v1/query", 500, 250)

    rendered = metrics.render_prometheus()
    assert 'endpoint="/v1/query",status="500"} 1' in rendered
    assert "agentic_rag_http_request_duration_seconds_sum 0.250000" in rendered
    assert "agentic_rag_active_requests 0" in rendered
    assert 'type="ValueError"} 1' in rendered

