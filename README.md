# RAG Retrieval

Tenant-scoped hybrid retrieval API for the AI Platform. This service retrieves indexed document chunks; it is deliberately not a chat or agent endpoint.

## Responsibilities
- Embed queries using FastEmbed dense embeddings and Qdrant sparse BM25.
- Run tenant/user-filtered dense and sparse searches against Qdrant.
- Fuse ranked results using reciprocal-rank fusion and hydrate source metadata from PostgreSQL.
- Keep PostgreSQL authoritative; Qdrant is a rebuildable derived index.

## API
- `GET /health` — liveness.
- `POST /search` — retrieval-only search; body: `{"query":"...","limit":10}`.
- Requires `X-RAG-Tenant-ID`; optionally accepts `X-RAG-User-ID`. Limit: 1–100.

The authenticated `ai-gateway` is the intended caller and must provide trusted tenant context. Do not expose this service directly to untrusted clients.

## Configuration
Required: `RAG_POSTGRES_DSN`, `RAG_QDRANT_URL`.
Optional: `RAG_QDRANT_COLLECTION` (default `rag_chunks`), `RAG_DENSE_MODEL` (default `BAAI/bge-small-en-v1.5`), `RAG_SPARSE_MODEL` (default `Qdrant/bm25`).

## Development
Python 3.11+; dependencies use uv.
```bash
uv sync --extra test
uv run pytest
```
Run through the platform composition in `ai-infra`. PostgreSQL and Qdrant must be reachable and Qdrant populated by `rag-indexer`.

## Data flow
`rag-ingestion → PostgreSQL chunks/outbox → rag-indexer → Qdrant → rag-retrieval`.
Answer generation and agent orchestration belong to `ai-gateway` / `agent-core`.

## Operations
Keep this API on the internal network; preserve tenant filters on every vector query. Monitor query/embedding latency, PostgreSQL/Qdrant availability and model-cache storage. Deployment readiness should validate downstream dependencies in addition to the liveness endpoint.
