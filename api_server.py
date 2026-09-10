from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from src.rag_pipeline import GraphRAGPipeline


class AskRequest(BaseModel):
    question: str


def create_app() -> FastAPI:
    app = FastAPI(title="UFDC GraphRAG API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    pipeline: GraphRAGPipeline | None = None

    @app.on_event("startup")
    def startup() -> None:
        nonlocal pipeline
        pipeline = GraphRAGPipeline(verbose=False)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "service": "ufdc-rag"}

    @app.post("/api/ask")
    def ask(req: AskRequest) -> dict:
        if not req.question or not req.question.strip():
            raise HTTPException(status_code=400, detail="Pertanyaan tidak boleh kosong.")

        if pipeline is None:
            raise HTTPException(status_code=500, detail="Pipeline belum siap.")

        result = pipeline.answer(req.question.strip())
        return {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "cache_hit": result.get("cache_hit"),
            "structured_facts_used": result.get("structured_facts_used", False),
            "elapsed_seconds": result.get("elapsed_seconds", 0),
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
