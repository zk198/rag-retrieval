from types import SimpleNamespace

import pytest

from rag_retrieval.service import RetrievalService


class FakeEmbedding:
    def __init__(self, vector):
        self.vector = vector
        self.calls = []

    def embed(self, texts):
        self.calls.append(texts)
        return iter([self.vector])


class FakeQdrant:
    def __init__(self):
        self.calls = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        payload = {
            "tenant_id": "t1",
            "user_id": "u1",
            "chunk_id": 7,
            "source_name": "mailbox.pst",
            "parent_kind": "message",
            "parent_id": 11,
            "text": "hello",
        }
        return SimpleNamespace(
            points=[SimpleNamespace(id="t1:7", payload=payload, score=0.9)]
        )


def make_service():
    service = RetrievalService.__new__(RetrievalService)
    service.db = "postgresql://test"
    service.collection = "rag_chunks"
    service.qdrant = FakeQdrant()
    service.dense = FakeEmbedding(SimpleNamespace(tolist=lambda: [0.1, 0.2]))
    service.sparse = FakeEmbedding(
        SimpleNamespace(
            indices=SimpleNamespace(tolist=lambda: [1]),
            values=SimpleNamespace(tolist=lambda: [0.5]),
        )
    )
    return service


def test_filter_always_contains_tenant_and_optional_user():
    service = make_service()

    tenant_only = service._filter("t1", None)
    scoped = service._filter("t1", "u1")

    assert len(tenant_only.must) == 1
    assert len(scoped.must) == 2
    assert scoped.must[0].key == "tenant_id"
    assert scoped.must[1].key == "user_id"


def test_search_uses_named_dense_and_sparse_query_points():
    service = make_service()
    service._hydrate = lambda ids, payloads, scores, tenant, user: ids

    result = service.search("hello", "t1", "u1", 5)

    assert result == ["t1:7"]
    assert len(service.qdrant.calls) == 2
    assert {call["using"] for call in service.qdrant.calls} == {"dense", "sparse"}
    assert all(call["query_filter"].must[0].key == "tenant_id" for call in service.qdrant.calls)
    assert all(call["query_filter"].must[0].match.value == "t1" for call in service.qdrant.calls)
    assert all(call["query_filter"].must[1].match.value == "u1" for call in service.qdrant.calls)


def test_search_rejects_missing_context():
    service = make_service()

    with pytest.raises(ValueError, match="tenant_id"):
        service.search("hello", "", None)

    with pytest.raises(ValueError, match="query"):
        service.search("   ", "t1", None)


def test_hydrate_batches_parent_lookups(monkeypatch):
    service = make_service()
    calls = []

    class Result:
        def fetchall(self):
            return [{"id": 11, "subject": "Test"}]

    class Conn:
        def execute(self, sql, params):
            calls.append((sql, params))
            return Result()

    class Context:
        def __enter__(self):
            return Conn()

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("rag_retrieval.service.psycopg.connect", lambda *a, **k: Context())

    payloads = {
        "t1:7": {
            "tenant_id": "t1",
            "user_id": "u1",
            "chunk_id": 7,
            "source_name": "mailbox.pst",
            "parent_kind": "message",
            "parent_id": 11,
            "text": "hello",
        }
    }
    result = service._hydrate(["t1:7"], payloads, {"t1:7": 0.1}, "t1", "u1")

    assert result[0]["parent"]["subject"] == "Test"
    assert len(calls) == 1
    assert "ANY(%s)" in calls[0][0]


def test_hydrate_rejects_cross_tenant_payload():
    service = make_service()
    payloads = {"other:7": {"tenant_id": "other", "user_id": "u1"}}

    with pytest.raises(ValueError, match="tenant mismatch"):
        service._hydrate(["other:7"], payloads, {"other:7": 0.1}, "t1", "u1")
