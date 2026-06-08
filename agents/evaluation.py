from agents.state import AgentEvaluation, SalesHelperState
from agents.tracing import trace_score


CONFIDENCE_THRESHOLD = 0.6


def append_evaluation(
    state: SalesHelperState,
    evaluation: AgentEvaluation,
) -> SalesHelperState:
    confidence = float(evaluation.get("confidence", 0.0))
    status = evaluation.get("status", "failed")
    notes = "; ".join(evaluation.get("notes", []))

    trace_score(
        state,
        name=f"{evaluation.get('agent_name', 'agent')}_confidence",
        value=confidence,
        comment=notes,
    )
    trace_score(
        state,
        name=f"{evaluation.get('agent_name', 'agent')}_passed",
        value=1.0 if status == "passed" else 0.0,
        comment=notes,
    )

    return {
        **state,
        "evaluations": [*state.get("evaluations", []), evaluation],
    }


def evaluate_knowledge_retrieval(state: SalesHelperState) -> SalesHelperState:
    sources = state.get("qdrant_sources", [])
    confidence = 0.9 if sources else 0.0
    status = "passed" if sources else "failed"

    if sources:
        notes = ["Qdrant sources returned."]
    else:
        notes = ["No Qdrant sources returned."]

    return append_evaluation(
        state,
        {
            "agent_name": "knowledge_retrieval",
            "status": status,
            "confidence": confidence,
            "notes": notes,
        },
    )


def evaluate_eval(state: SalesHelperState) -> SalesHelperState:
    status = state.get("eval_status")
    passed = status in {"approved", "needs_revision", "unsafe"}

    return append_evaluation(
        state,
        {
            "agent_name": "eval",
            "status": "passed" if passed else "failed",
            "confidence": 0.9 if passed else 0.0,
            "notes": ["Eval returned a clear status."] if passed else ["Eval did not return a clear status."],
        },
    )


def has_failed_evaluation(state: SalesHelperState) -> bool:
    return any(evaluation.get("status") == "failed" for evaluation in state.get("evaluations", []))
