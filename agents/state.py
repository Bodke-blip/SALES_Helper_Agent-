from typing import Any, Literal, TypedDict


GuardrailStatus = Literal["safe", "needs_clarification", "blocked"]
EvaluationStatus = Literal["passed", "failed"]
FinalStatus = Literal["approved", "needs_revision", "unsafe"]


class AgentEvaluation(TypedDict, total=False):
    agent_name: str
    status: EvaluationStatus
    confidence: float
    notes: list[str]
    required_fix: str


class SalesHelperState(TypedDict, total=False):
    trace_id: str
    workflow_started_at: float
    workflow_timings: list[dict[str, Any]]
    user_query: str
    contextual_query: str
    chat_history: list[dict[str, str]]
    use_gemini_llm: bool
    use_local_llm: bool
    input_guardrail_status: GuardrailStatus
    input_guardrail_reason: str
    intent: str
    selected_agents: list[str]
    orchestrator_tool: str
    orchestrator_tool_input: str
    orchestrator_reason: str
    orchestrator_decision: dict[str, Any]
    llm_provider_status: dict[str, Any]
    orchestrator_llm_model: str
    internal_context: list[dict[str, Any]]
    qdrant_sources: list[dict[str, Any]]
    retrieval_collection: str
    retrieval_customer_filter: str
    retrieval_cache_status: str
    retrieval_error: str
    eval_status: FinalStatus
    eval_notes: str
    eval_llm_model: str
    evaluations: list[AgentEvaluation]
    fallback_status: str
    fallback_reason: str
    final_response: dict[str, Any]
