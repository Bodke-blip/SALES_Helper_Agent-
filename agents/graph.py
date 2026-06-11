from langgraph.graph import END, START, StateGraph

from agents.fallback import fallback_handler
from agents.guardrails import input_guardrail, output_guardrail
from agents.sales_helper_agent import sales_helper_agent_node
from agents.state import SalesHelperState
from agents.tracing import initialize_observability, traced_node


def guardrail_route(state: SalesHelperState) -> str:
    if state.get("input_guardrail_status") == "safe":
        return "sales_helper_agent"
    return "fallback_handler"


def build_sales_helper_graph():
    graph = StateGraph(SalesHelperState)

    graph.add_node("initialize_observability", initialize_observability)
    graph.add_node("input_guardrail", traced_node("input_guardrail", input_guardrail))
    graph.add_node("sales_helper_agent", traced_node("sales_helper_agent", sales_helper_agent_node))
    graph.add_node("fallback_handler", traced_node("fallback_handler", fallback_handler))
    graph.add_node("output_guardrail", traced_node("output_guardrail", output_guardrail))

    graph.add_edge(START, "initialize_observability")
    graph.add_edge("initialize_observability", "input_guardrail")
    graph.add_conditional_edges(
        "input_guardrail",
        guardrail_route,
        {
            "sales_helper_agent": "sales_helper_agent",
            "fallback_handler": "fallback_handler",
        },
    )

    graph.add_edge("sales_helper_agent", "output_guardrail")
    graph.add_edge("fallback_handler", "output_guardrail")
    graph.add_edge("output_guardrail", END)

    return graph.compile()
