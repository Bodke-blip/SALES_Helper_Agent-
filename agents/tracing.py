import os
import time
from functools import wraps
from typing import Any, Callable
from uuid import uuid4

from dotenv import load_dotenv

from agents.state import SalesHelperState


load_dotenv()

SENSITIVE_KEYS = (
    "api_key",
    "secret",
    "token",
    "password",
    "credential",
)


def _load_langfuse_client():
    try:
        from langfuse import Langfuse
    except ImportError:
        return None

    if os.getenv("ENABLE_LANGFUSE_TRACING", "").lower() not in {"1", "true", "yes"}:
        return None

    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        return None

    return Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
    )


langfuse_client = _load_langfuse_client()


def langfuse_enabled() -> bool:
    return langfuse_client is not None


def sanitize_for_trace(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized = {}

        for key, item in value.items():
            key_text = str(key).lower()

            if any(sensitive_key in key_text for sensitive_key in SENSITIVE_KEYS):
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = sanitize_for_trace(item)

        return sanitized

    if isinstance(value, list):
        return [sanitize_for_trace(item) for item in value[:20]]

    if isinstance(value, tuple):
        return tuple(sanitize_for_trace(item) for item in value[:20])

    if isinstance(value, str) and len(value) > 3000:
        return value[:3000] + "...[TRUNCATED]"

    return value


def ensure_trace_id(state: SalesHelperState) -> SalesHelperState:
    if state.get("trace_id"):
        return state

    if langfuse_enabled():
        trace_id = langfuse_client.create_trace_id(seed=str(uuid4()))
    else:
        trace_id = uuid4().hex

    return {
        **state,
        "trace_id": trace_id,
    }


def start_trace(state: SalesHelperState) -> None:
    if not langfuse_enabled():
        return

    observation = langfuse_client.start_observation(
        trace_context={"trace_id": state.get("trace_id")},
        name="Predikly Sales Helper Agentic AI Solution",
        as_type="agent",
        input=sanitize_for_trace(
            {
                "user_query": state.get("user_query", ""),
            }
        ),
        metadata={
            "system": "predikly_sales_helper",
            "framework": "langgraph",
        },
    )
    observation.update(output={"trace_started": True})
    observation.end()


def trace_event(
    state: SalesHelperState,
    *,
    name: str,
    input_data: Any | None = None,
    output_data: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    if not langfuse_enabled():
        return

    span = langfuse_client.start_observation(
        trace_context={"trace_id": state.get("trace_id")},
        name=name,
        as_type=metadata.get("observation_type", "span") if metadata else "span",
        input=sanitize_for_trace(input_data),
        metadata=sanitize_for_trace(metadata or {}),
    )
    span.update(output=sanitize_for_trace(output_data))
    span.end()


def trace_score(
    state: SalesHelperState,
    *,
    name: str,
    value: float,
    comment: str = "",
) -> None:
    if not langfuse_enabled():
        return

    langfuse_client.create_score(
        trace_id=state.get("trace_id"),
        name=name,
        value=value,
        comment=comment,
    )


def flush_langfuse() -> None:
    if langfuse_enabled():
        langfuse_client.flush()


def build_final_response_trace_output(state: SalesHelperState) -> dict[str, Any]:
    final_response = state.get("final_response", {}) or {}

    return {
        "answer": final_response.get("answer", ""),
        "sources": final_response.get("sources", []),
        "evaluation": final_response.get("evaluation", []),
        "llm_models": final_response.get("llm_models", {}),
        "fallback_status": final_response.get("fallback_status", state.get("fallback_status", "not_used")),
        "trace_id": final_response.get("trace_id", state.get("trace_id")),
        "retrieval_collection": state.get("retrieval_collection", ""),
        "retrieval_cache_status": state.get("retrieval_cache_status", "not_used"),
    }


def traced_node(node_name: str, node_fn: Callable[[SalesHelperState], SalesHelperState]):
    @wraps(node_fn)
    def wrapper(state: SalesHelperState) -> SalesHelperState:
        node_started_at = time.perf_counter()
        input_summary = {
            "intent": state.get("intent"),
            "selected_agents": state.get("selected_agents", []),
            "fallback_status": state.get("fallback_status"),
            "eval_status": state.get("eval_status"),
        }
        try:
            output_state = node_fn(state)
            node_status = "completed"
        except Exception as error:
            elapsed_ms = round((time.perf_counter() - node_started_at) * 1000, 2)
            timing = {
                "step": node_name,
                "status": "error",
                "duration_ms": elapsed_ms,
                "duration_seconds": round(elapsed_ms / 1000, 3),
                "error": str(error),
            }
            state = {
                **state,
                "workflow_timings": [*state.get("workflow_timings", []), timing],
            }
            raise

        elapsed_ms = round((time.perf_counter() - node_started_at) * 1000, 2)
        timing = {
            "step": node_name,
            "status": node_status,
            "duration_ms": elapsed_ms,
            "duration_seconds": round(elapsed_ms / 1000, 3),
        }
        output_state = {
            **output_state,
            "workflow_timings": [
                *state.get("workflow_timings", []),
                timing,
            ],
        }
        output_summary = {
            "intent": output_state.get("intent"),
            "selected_agents": output_state.get("selected_agents", []),
            "fallback_status": output_state.get("fallback_status"),
            "eval_status": output_state.get("eval_status"),
            "evaluations_count": len(output_state.get("evaluations", [])),
            "duration_ms": elapsed_ms,
        }

        if node_name == "output_guardrail":
            output_summary["final_response"] = build_final_response_trace_output(output_state)

        trace_event(
            output_state,
            name=node_name,
            input_data=input_summary,
            output_data=output_summary,
            metadata={"node_type": "langgraph_node"},
        )

        if node_name == "output_guardrail":
            trace_event(
                output_state,
                name="final_response",
                input_data={
                    "user_query": output_state.get("user_query", ""),
                    "intent": output_state.get("intent", ""),
                },
                output_data=build_final_response_trace_output(output_state),
                metadata={
                    "node_type": "final_output",
                    "observation_type": "span",
                },
            )
            flush_langfuse()

        return output_state

    return wrapper


def initialize_observability(state: SalesHelperState) -> SalesHelperState:
    state = ensure_trace_id(state)
    workflow_started_at = state.get("workflow_started_at") or time.perf_counter()
    state = {
        **state,
        "workflow_started_at": workflow_started_at,
        "workflow_timings": state.get("workflow_timings", []),
    }
    start_trace(state)
    trace_event(
        state,
        name="initialize_observability",
        input_data={"user_query_present": bool(state.get("user_query"))},
        output_data={"trace_id": state.get("trace_id"), "langfuse_enabled": langfuse_enabled()},
        metadata={"node_type": "observability"},
    )
    return state
