from __future__ import annotations

from collections import defaultdict
import os

import psycopg
from psycopg.rows import dict_row
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    SparseVector,
)


class RetrievalService:
    def __init__(self):
        self.db = os.environ["RAG_POSTGRES_DSN"]
        self.qdrant = QdrantClient(
            url=os.getenv("RAG_QDRANT_URL", "http://qdrant:6333")
        )
        self.collection = os.getenv("RAG_QDRANT_COLLECTION", "rag_chunks")
        self.dense_model = os.getenv(
            "RAG_DENSE_MODEL", "BAAI/bge-small-en-v1.5"
        )
        self.sparse_model = os.getenv("RAG_SPARSE_MODEL", "Qdrant/bm25")
        self.dense = TextEmbedding(model_name=self.dense_model, lazy_load=True)
        self.sparse = SparseTextEmbedding(
            model_name=self.sparse_model, lazy_load=True
        )

    def _filter(self, tenant_id: str, user_id: str | None) -> Filter:
        conditions = [
            FieldCondition(
                key="tenant_id", match=MatchValue(value=tenant_id)
            )
        ]
        if user_id:
            conditions.append(
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            )
        return Filter(must=conditions)

    def search(
        self,
        query: str,
        tenant_id: str,
        user_id: str | None,
        limit: int = 10,
    ):
        if not tenant_id:
            raise ValueError("tenant_id is required")
        if not query or not query.strip():
            raise ValueError("query is required")

        dense = list(self.dense.embed([query]))[0].tolist()
        sparse = list(self.sparse.embed([query]))[0]
        scope = self._filter(tenant_id, user_id)
        candidate_limit = max(limit * 4, 20)

        dense_hits = self.qdrant.query_points(
            collection_name=self.collection,
            query=dense,
            using="dense",
            query_filter=scope,
            limit=candidate_limit,
            with_payload=True,
        ).points
        sparse_hits = self.qdrant.query_points(
            collection_name=self.collection,
            query=SparseVector(
                indices=sparse.indices.tolist(),
                values=sparse.values.tolist(),
            ),
            using="sparse",
            query_filter=scope,
            limit=candidate_limit,
            with_payload=True,
        ).points

        scores = defaultdict(float)
        payloads = {}
        for rank, hit in enumerate(dense_hits, 1):
            point_id = str(hit.id)
            scores[point_id] += 1.0 / (60 + rank)
            payloads[point_id] = hit.payload or {}
        for rank, hit in enumerate(sparse_hits, 1):
            point_id = str(hit.id)
            scores[point_id] += 1.0 / (60 + rank)
            payloads[point_id] = hit.payload or {}

        ordered = sorted(scores, key=scores.get, reverse=True)[:limit]
        return self._hydrate(ordered, payloads, scores, tenant_id, user_id)

    def _hydrate(self, ids, payloads, scores, tenant_id, user_id):
        if not ids:
            return []

        message_ids = []
        document_ids = []
        for point_id in ids:
            payload = payloads[point_id]
            if payload.get("tenant_id") != tenant_id:
                raise ValueError("retrieval payload tenant mismatch")
            if user_id and payload.get("user_id") != user_id:
                raise ValueError("retrieval payload user mismatch")
            if payload.get("parent_kind") == "message":
                message_ids.append(payload["parent_id"])
            else:
                document_ids.append(payload["parent_id"])

        with psycopg.connect(self.db, row_factory=dict_row) as conn:
            messages = {}
            if message_ids:
                rows = conn.execute(
                    "SELECT * FROM messages WHERE tenant_id=%s AND id = ANY(%s)",
                    (tenant_id, message_ids),
                ).fetchall()
                messages = {str(row["id"]): dict(row) for row in rows}

            documents = {}
            if document_ids:
                rows = conn.execute(
                    "SELECT * FROM documents WHERE tenant_id=%s AND id = ANY(%s)",
                    (tenant_id, document_ids),
                ).fetchall()
                documents = {str(row["id"]): dict(row) for row in rows}

        out = []
        for point_id in ids:
            payload = payloads[point_id]
            parent_id = str(payload["parent_id"])
            if payload.get("parent_kind") == "message":
                parent = messages.get(parent_id)
            else:
                parent = documents.get(parent_id)
            out.append(
                {
                    "chunk_id": payload.get("chunk_id"),
                    "parent_kind": payload.get("parent_kind"),
                    "score": scores[point_id],
                    "text": payload.get("text"),
                    "source_name": payload.get("source_name"),
                    "parent": parent,
                }
            )
        return out
