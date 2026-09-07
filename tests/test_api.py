"""
API integration and unit tests for FastAPI backend endpoints.
"""
import io
import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture
def client():
    # Context manager triggers lifespan setup and teardown
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert "KT Knowledge Transfer Assistant" in data["message"]


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "memory_backend" in data
    assert "vector_backend" in data


def test_list_documents(client):
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert isinstance(data["documents"], list)


def test_ask_validation_empty(client):
    # Empty question should fail validation or return 422
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_session_history_and_clear(client):
    # History for a non-existent/new session should return empty history
    session_id = "test-session-1234"
    resp = client.get(f"/sessions/{session_id}/history")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == session_id
    assert data["turn_count"] == 0

    # Clear session history
    del_resp = client.delete(f"/sessions/{session_id}/history")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "cleared"


def test_upload_unsupported_format(client):
    fake_file = io.BytesIO(b"malicious script content")
    response = client.post(
        "/upload",
        files={"files": ("malicious.exe", fake_file, "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "Unsupported format" in response.json()["detail"]
