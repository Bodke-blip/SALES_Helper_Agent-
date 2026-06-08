from langgraph.graph import END, START, StateGraph

from agents.evaluation import (
    evaluate_eval,
    evaluate_knowledge_retrieval,
    has_failed_evaluation,
)
from agents.eval_agent import eval_agent
from agents.fallback import fallback_handler
from agents.guardrails import input_guardrail, output_guardrail
from agents.knowledge_retrieval_agent import knowledge_retrieval_agent
from agents.orchestrator_agent import compose_final_response, main_orchestrator_agent
from agents.state import SalesHelperState
from agents.tracing import initialize_observability, traced_node


def guardrail_route(state: SalesHelperState) -> str:
    if state.get("input_guardrail_status") == "safe":
        return "main_orchestrator"
    return "fallback_handler"


def route_after_orchestrator(state: SalesHelperState) -> str:
    if state.get("orchestrator_tool") == "explain_capabilities":
        return "compose_final_response"

    selected_agents = state.get("selected_agents", [])

    if not selected_agents:
        return "fallback_handler"

    return selected_agents[0]


def route_after_knowledge(state: SalesHelperState) -> str:
    if has_failed_evaluation(state):
        return "fallback_handler"

    return "eval"


def route_after_eval(state: SalesHelperState) -> str:
    if has_failed_evaluation(state) or state.get("eval_status") != "approved":
        return "fallback_handler"

    return "compose_final_response"


def build_sales_helper_graph():
    graph = StateGraph(SalesHelperState)

    graph.add_node("initialize_observability", initialize_observability)
    graph.add_node("input_guardrail", traced_node("input_guardrail", input_guardrail))
    graph.add_node("main_orchestrator", traced_node("main_orchestrator", main_orchestrator_agent))
    graph.add_node("knowledge_retrieval", traced_node("knowledge_retrieval", knowledge_retrieval_agent))
    graph.add_node("evaluate_knowledge_retrieval", traced_node("evaluate_knowledge_retrieval", evaluate_knowledge_retrieval))
    graph.add_node("eval", traced_node("eval", eval_agent))
    graph.add_node("evaluate_eval", traced_node("evaluate_eval", evaluate_eval))
    graph.add_node("fallback_handler", traced_node("fallback_handler", fallback_handler))
    graph.add_node("compose_final_response", traced_node("compose_final_response", compose_final_response))
    graph.add_node("output_guardrail", traced_node("output_guardrail", output_guardrail))

    graph.add_edge(START, "initialize_observability")
    graph.add_edge("initialize_observability", "input_guardrail")
    graph.add_conditional_edges(
        "input_guardrail",
        guardrail_route,
        {
            "main_orchestrator": "main_orchestrator",
            "fallback_handler": "fallback_handler",
        },
    )
    graph.add_conditional_edges(
        "main_orchestrator",
        route_after_orchestrator,
        {
            "knowledge_retrieval": "knowledge_retrieval",
            "compose_final_response": "compose_final_response",
            "fallback_handler": "fallback_handler",
        },
    )

    graph.add_edge("knowledge_retrieval", "evaluate_knowledge_retrieval")
    graph.add_conditional_edges(
        "evaluate_knowledge_retrieval",
        route_after_knowledge,
        {
            "eval": "eval",
            "fallback_handler": "fallback_handler",
        },
    )

    graph.add_edge("eval", "evaluate_eval")
    graph.add_conditional_edges(
        "evaluate_eval",
        route_after_eval,
        {
            "compose_final_response": "compose_final_response",
            "fallback_handler": "fallback_handler",
        },
    )

    graph.add_edge("compose_final_response", "output_guardrail")
    graph.add_edge("fallback_handler", "output_guardrail")
    graph.add_edge("output_guardrail", END)

    return graph.compile()
