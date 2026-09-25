from __future__ import annotations

from collections import defaultdict
import os

import psycopg
from psycopg.rows import dict_row
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, NamedSparseVector, NamedVector, SparseVector


class RetrievalService:
    def __init__(self):
        self.db = os.environ["RAG_POSTGRES_DSN"]
        self.qdrant = QdrantClient(url=os.getenv("RAG_QDRANT_URL", "http://qdrant:6333"))
        self.collection = os.getenv("RAG_QDRANT_COLLECTION", "rag_chunks")
        self.dense = TextEmbedding(os.getenv("RAG_DENSE_MODEL", "BAAI/bge-small-en-v1.5"))
        self.sparse = SparseTextEmbedding(os.getenv("RAG_SPARSE_MODEL", "Qdrant/bm25"))

    def search(self, query: str, tenant_id: str, user_id: str | None, limit: int = 10):
        dense = list(self.dense.embed([query]))[0].tolist()
        sparse = list(self.sparse.embed([query]))[0]
        scope = [FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
        if user_id:
            scope.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))
        filt = Filter(must=scope)
        k = max(limit * 4, 20)
        dense_hits = self.qdrant.search(self.collection, NamedVector(name="dense", vector=dense), query_filter=filt, limit=k, with_payload=True)
        sparse_hits = self.qdrant.search(
            self.collection,
            NamedSparseVector(name="sparse", vector=SparseVector(indices=sparse.indices.tolist(), values=sparse.values.tolist())),
            query_filter=filt,
            limit=k,
            with_payload=True,
        )
        scores = defaultdict(float)
        payloads = {}
        for rank, hit in enumerate(dense_hits, 1):
            scores[str(hit.id)] += 1.0 / (60 + rank)
            payloads[str(hit.id)] = hit.payload or {}
        for rank, hit in enumerate(sparse_hits, 1):
            scores[str(hit.id)] += 1.0 / (60 + rank)
            payloads[str(hit.id)] = hit.payload or {}
        ordered = sorted(scores, key=scores.get, reverse=True)[:limit]
        return self._hydrate(ordered, payloads, scores)

    def _hydrate(self, ids, payloads, scores):
        if not ids:
            return []
        with psycopg.connect(self.db, row_factory=dict_row) as conn:
            out=[]
            for point_id in ids:
                p=payloads[point_id]
                if p.get("parent_kind") == "message":
                    row=conn.execute(
                        "SELECT * FROM messages WHERE tenant_id=%s AND id=%s",
                        (p["tenant_id"], p["parent_id"]),
                    ).fetchone()
                else:
                    row=conn.execute(
                        "SELECT * FROM documents WHERE tenant_id=%s AND id=%s",
                        (p["tenant_id"], p["parent_id"]),
                    ).fetchone()
                out.append({"chunk_id":p.get("chunk_id"), "score":scores[point_id], "text":p.get("text"), "source_name":p.get("source_name"), "parent":dict(row) if row else None})
            return out
