from typing import Any


def with_verbose_details(item: dict[str, Any], details: dict[str, Any], verbose: bool) -> dict[str, Any]:
    if not verbose:
        return item

    return {
        **item,
        "verbose": details,
    }


def build_agent_trace(result: dict[str, Any], *, verbose: bool = False) -> list[dict[str, Any]]:
    evaluations = {
        evaluation.get("agent_name"): evaluation
        for evaluation in result.get("evaluations", [])
    }
    selected_agents = result.get("selected_agents", [])
    trace_items = [
        with_verbose_details(
            {
                "agent": "Input Guardrail",
                "status": result.get("input_guardrail_status", "unknown"),
                "summary": result.get("input_guardrail_reason", ""),
            },
            {
                "what_it_checked": [
                    "prompt-injection attempts",
                    "system prompt / hidden reasoning requests",
                    "self-harm or harm-inducing intent",
                    "violence or credential extraction",
                    "sales-helper scope",
                ],
                "decision": result.get("input_guardrail_status", "unknown"),
                "safe_reasoning_summary": result.get("input_guardrail_reason", ""),
            },
            verbose,
        ),
        with_verbose_details(
            {
                "agent": "Main Orchestrator Agent",
                "status": "completed",
                "summary": (
                    f"Called tool '{result.get('orchestrator_tool', 'unknown')}' "
                    f"and executed: {', '.join(selected_agents) or 'direct response'}."
                ),
                "model": result.get("orchestrator_llm_model"),
            },
            {
                "thinking_summary": (
                    "The LLM orchestrator selected a callable tool. Retrieval is used for "
                    "data questions, while capabilities questions can be answered directly."
                ),
                "inputs_considered": ["user_query", "chat_history", "contextual_query"],
                "selected_agents": selected_agents,
                "orchestrator_tool": result.get("orchestrator_tool", ""),
                "orchestrator_reason": result.get("orchestrator_reason", ""),
                "tools_used": [
                    "LLM Tool Selection",
                    result.get("orchestrator_tool", "unknown"),
                ],
            },
            verbose,
        ),
    ]

    if "knowledge_retrieval" in selected_agents:
        evaluation = evaluations.get("knowledge_retrieval", {})
        trace_items.append(
            with_verbose_details(
                {
                    "agent": "Knowledge Retrieval Agent",
                    "status": evaluation.get("status", "not_evaluated"),
                    "summary": (
                        f"Searched Qdrant collection '{result.get('retrieval_collection', 'unknown')}' "
                        f"and returned {len(result.get('internal_context', []))} context item(s) "
                        f"with {len(result.get('qdrant_sources', []))} source record(s)."
                    ),
                    "confidence": evaluation.get("confidence"),
                },
                {
                    "thinking_summary": (
                        "The agent searched internal Qdrant knowledge with hybrid dense+sparse "
                        "retrieval, ranked matching chunks, extracted source metadata, and "
                        "prepared grounded context for downstream agents."
                    ),
                    "tools_used": [
                        "Qdrant Hybrid Retrieval Tool",
                        "Metadata Filter Tool",
                        "Context Builder Tool",
                        "Source Ranking Tool",
                        "Source Citation Tool",
                    ],
                    "inputs_considered": [
                        "user_query",
                        "chat_history",
                        "contextual_query",
                        "HYBRID_QDRANT_COLLECTION_NAME",
                        "HYBRID_QDRANT_FALLBACK_COLLECTION_NAME",
                    ],
                    "outputs_created": ["internal_context", "qdrant_sources"],
                    "retrieval_collection": result.get("retrieval_collection", "unknown"),
                    "retrieval_error": result.get("retrieval_error", ""),
                    "retrieval_cache_status": result.get("retrieval_cache_status", "not_used"),
                },
                verbose,
            )
        )

    if "eval" in selected_agents or result.get("eval_status"):
        eval_evaluation = evaluations.get("eval", {})
        trace_items.append(
            with_verbose_details(
                {
                    "agent": "Eval Agent",
                    "status": result.get("eval_status", eval_evaluation.get("status", "unknown")),
                    "summary": result.get("eval_notes", "Checked grounding, safety, and response quality."),
                    "model": result.get("eval_llm_model"),
                    "confidence": eval_evaluation.get("confidence"),
                },
                {
                    "thinking_summary": (
                        "The agent checked whether the response plan is grounded in available sources, "
                        "avoids unsupported claims, stays safe for brand/tone, and has a clear status."
                    ),
                    "tools_used": [
                        "Hallucination Check Tool",
                        "Source Grounding Tool",
                        "Confidentiality Check Tool",
                        "Sales Claim Validation Tool",
                        "Tone and Brand Safety Tool",
                        "Completeness Check Tool",
                        "Final Approval Tool",
                    ],
                    "inputs_considered": ["user_query", "internal_context", "qdrant_sources"],
                    "outputs_created": ["eval_status", "eval_notes"],
                },
                verbose,
            )
        )

    trace_items.append(
        with_verbose_details(
            {
                "agent": "Output Guardrail",
                "status": result.get("final_response", {}).get(
                    "fallback_status",
                    result.get("fallback_status", "not_used"),
                ),
                "summary": "Prepared UI-safe output with sources, evaluation results, fallback status, and trace ID only.",
            },
            {
                "thinking_summary": (
                    "The output guardrail ensures the UI receives only answer text, safe summaries, "
                    "sources, evaluation results, fallback status, model labels, and trace ID."
                ),
                "blocked_from_ui": [
                    "hidden chain-of-thought",
                    "system prompts",
                    "developer instructions",
                    "credentials",
                    "raw unfiltered state",
                ],
            },
            verbose,
        )
    )

    return trace_items
