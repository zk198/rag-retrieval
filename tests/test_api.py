from fastapi.testclient import TestClient

from rag_retrieval import api


class FakeService:
    def __init__(self):
        self.calls = []

    def search(self, query, tenant_id, user_id, limit):
        self.calls.append((query, tenant_id, user_id, limit))
        return [{"chunk_id": 1}]


def test_health_does_not_initialize_retrieval_service(monkeypatch):
    api._service = None
    client = TestClient(api.app)

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert api._service is None


def test_search_requires_tenant():
    api._service = FakeService()
    client = TestClient(api.app)

    response = client.post("/search", json={"query": "hello"})

    assert response.status_code == 401


def test_search_passes_gateway_context():
    service = FakeService()
    api._service = service
    client = TestClient(api.app)

    response = client.post(
        "/search",
        json={"query": "hello", "limit": 3},
        headers={"X-RAG-Tenant-ID": "t1", "X-RAG-User-ID": "u1"},
    )

    assert response.status_code == 200
    assert response.json() == [{"chunk_id": 1}]
    assert service.calls == [("hello", "t1", "u1", 3)]


def test_search_maps_service_validation_to_400():
    class InvalidService:
        def search(self, *args):
            raise ValueError("bad query")

    api._service = InvalidService()
    client = TestClient(api.app)

    response = client.post(
        "/search",
        json={"query": "hello"},
        headers={"X-RAG-Tenant-ID": "t1"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "bad query"
