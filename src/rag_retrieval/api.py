from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from .service import RetrievalService

app=FastAPI(title="RAG Retrieval")
service=RetrievalService()

class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=100)

@app.get("/healthz")
def healthz(): return {"status":"ok"}

@app.post("/search")
def search(request: SearchRequest, x_rag_tenant_id: str | None = Header(default=None), x_rag_user_id: str | None = Header(default=None)):
    if not x_rag_tenant_id:
        raise HTTPException(401, "tenant context is required")
    return service.search(request.query, x_rag_tenant_id, x_rag_user_id, request.limit)
