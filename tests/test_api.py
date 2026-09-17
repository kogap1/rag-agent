from fastapi.testclient import TestClient

import api


def test_liveness_returns_request_id():
    response = TestClient(api.app).get("/v1/health/live", headers={"X-Request-ID": "test-request"})
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["X-Request-ID"] == "test-request"


def test_metrics_requires_configured_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "top-secret")
    api.get_settings.cache_clear()
    try:
        client = TestClient(api.app)
        assert client.get("/metrics").status_code == 401
        response = client.get("/metrics", headers={"X-API-Key": "top-secret"})
        assert response.status_code == 200
        assert "agentic_rag_http_requests_total" in response.text
    finally:
        api.get_settings.cache_clear()


def test_mcp_mount_uses_same_api_key_boundary(monkeypatch):
    monkeypatch.setenv("API_KEY", "top-secret")
    api.get_settings.cache_clear()
    try:
        response = TestClient(api.app).post("/mcp/", json={})
        assert response.status_code == 401
        assert response.json()["detail"] == "API key 无效"
    finally:
        api.get_settings.cache_clear()
# RESUME_ALIGN_ACCEPTANCE


def test_version_and_rollback_endpoints_require_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "top-secret")
    api.get_settings.cache_clear()
    try:
        client = TestClient(api.app)
        assert client.get("/v1/documents/versions?source=rules.pdf").status_code == 401
        assert client.post("/v1/documents/rollback", json={"source": "rules.pdf"}).status_code == 401
    finally:
        api.get_settings.cache_clear()


def test_version_list_and_rollback_conflict_are_exposed(monkeypatch, tmp_path):
    monkeypatch.setenv("API_KEY", "top-secret")
    monkeypatch.setenv("METADATA_DB_PATH", str(tmp_path / "catalog.sqlite3"))
    api.get_settings.cache_clear()
    api.get_services.cache_clear()
    try:
        client = TestClient(api.app)
        headers = {"X-API-Key": "top-secret"}

        versions = client.get("/v1/documents/versions?source=rules.pdf", headers=headers)
        assert versions.status_code == 200
        assert versions.json() == {"source": "rules.pdf", "versions": []}

        rollback = client.post(
            "/v1/documents/rollback",
            json={"source": "rules.pdf"},
            headers=headers,
        )
        assert rollback.status_code == 409
    finally:
        api.get_settings.cache_clear()
        api.get_services.cache_clear()
