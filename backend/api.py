from pathlib import Path
import os
import time
from typing import Any
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agents.graph import build_sales_helper_graph
from agents.knowledge_retrieval_agent import (
    clear_retrieval_cache,
    retrieval_cache_status,
)
from agents.llm import (
    reset_llm_preferences,
    reset_request_started_at,
    set_llm_preferences,
    set_request_started_at,
)
from backend.agent_trace import build_agent_trace
from backend.chat_memory import (
    append_chat_turn,
    build_contextual_query,
    clear_chat_history,
    get_chat_with_messages,
    get_chat_history,
    initialize_chat_store,
    list_chat_sessions,
)
from backend.schemas import QueryRequest


BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


def seeded_internal_context(query: str) -> dict[str, list[dict[str, Any]]]:
    return {
        "internal_context": [],
        "qdrant_sources": [],
    }


def build_workflow_timings(result: dict[str, Any], request_started_at: float) -> list[dict[str, Any]]:
    total_duration_ms = round((time.perf_counter() - request_started_at) * 1000, 2)
    return [
        *result.get("workflow_timings", []),
        {
            "step": "total_request",
            "status": "completed",
            "duration_ms": total_duration_ms,
            "duration_seconds": round(total_duration_ms / 1000, 3),
        },
    ]


def build_chat_turn_metadata(result: dict[str, Any], retrieval_collection: str) -> dict[str, Any]:
    sources = result.get("qdrant_sources", [])
    source_hints = []

    for source in sources[:3]:
        source_hints.append(
            {
                "customer_name": source.get("customer_name", ""),
                "usecase_name": source.get("usecase_name", ""),
                "ppt_name": source.get("ppt_name", ""),
                "slide_number": source.get("slide_number"),
            }
        )

    return {
        "retrieval_collection": retrieval_collection,
        "source_hints": source_hints,
    }


def create_app() -> FastAPI:
    app = FastAPI(title="Predikly Sales Helper")
    frontend_origins = [
        origin.strip()
        for origin in os.getenv("FRONTEND_ORIGINS", "").split(",")
        if origin.strip()
    ]

    if frontend_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=frontend_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    graph = build_sales_helper_graph()
    initialize_chat_store()
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/app.js")
    def frontend_app() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "app.js")

    @app.get("/config.js")
    def frontend_config() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "config.js")

    @app.get("/styles.css")
    def frontend_styles() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "styles.css")

    @app.post("/query")
    def query_agent(request: QueryRequest) -> dict[str, Any]:
        request_started_at = time.perf_counter()
        session_id = request.session_id or f"session_{uuid4()}"
        chat_history = get_chat_history(session_id)
        contextual_query = build_contextual_query(request.query, chat_history)
        request_token = set_request_started_at(request_started_at)
        llm_token = set_llm_preferences(
            use_gemini=request.use_gemini_llm,
            use_local=request.use_local_llm,
        )
        seed = seeded_internal_context(request.query)

        try:
            result = graph.invoke(
                {
                    "user_query": request.query,
                    "contextual_query": contextual_query,
                    "chat_history": chat_history,
                    "use_gemini_llm": request.use_gemini_llm,
                    "use_local_llm": request.use_local_llm,
                    "workflow_started_at": request_started_at,
                    **seed,
                }
            )
        finally:
            reset_request_started_at(request_token)
            reset_llm_preferences(llm_token)

        final_response = result.get("final_response", {})
        workflow_timings = build_workflow_timings(result, request_started_at)
        response_text = str(final_response.get("answer") or final_response.get("message") or "")
        retrieval_collection = result.get("retrieval_collection", "")

        if final_response:
            append_chat_turn(
                session_id,
                request.query,
                response_text,
                build_chat_turn_metadata(result, retrieval_collection),
            )
            return {
                **final_response,
                "session_id": session_id,
                "cache_status": result.get("retrieval_cache_status", "not_used"),
                "retrieval_collection": retrieval_collection,
                "workflow_timings": workflow_timings,
                "agent_trace": build_agent_trace(result, verbose=request.verbose),
            }

        return {
            "message": "No final response was produced.",
            "session_id": session_id,
            "sources": [],
            "evaluation": result.get("evaluations", []),
            "fallback_status": result.get("fallback_status", "unknown"),
            "trace_id": result.get("trace_id"),
            "cache_status": result.get("retrieval_cache_status", "not_used"),
            "retrieval_collection": retrieval_collection,
            "workflow_timings": workflow_timings,
            "agent_trace": build_agent_trace(result, verbose=request.verbose),
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/cache/status")
    def cache_status() -> dict[str, Any]:
        return retrieval_cache_status()

    @app.delete("/cache")
    def clear_cache() -> dict[str, Any]:
        return clear_retrieval_cache()

    @app.delete("/chat/{session_id}")
    def clear_chat(session_id: str) -> dict[str, str]:
        clear_chat_history(session_id)
        return {"status": "cleared", "session_id": session_id}

    @app.get("/chats")
    def list_chats() -> dict[str, Any]:
        return {"sessions": list_chat_sessions()}

    @app.get("/chat/{session_id}")
    def get_chat(session_id: str) -> dict[str, Any]:
        return get_chat_with_messages(session_id)

    return app
