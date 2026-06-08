from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    handler: Callable | None = None


ORCHESTRATOR_TOOLS = [
    ToolSpec("hybrid_retrieval", "Searches Predikly internal Qdrant data with dense+sparse RRF retrieval."),
    ToolSpec("usecase_catalog", "Lists or counts use cases from Qdrant payload metadata, optionally filtered by company, domain, or country."),
    ToolSpec("explain_capabilities", "Explains the assistant's available tools, retrieval behavior, and limits."),
]

KNOWLEDGE_RETRIEVAL_TOOLS = [
    ToolSpec("qdrant_hybrid_retrieval", "Retrieves internal chunks from Qdrant with dense+sparse RRF search."),
    ToolSpec("metadata_filter", "Applies Qdrant metadata filters."),
    ToolSpec("context_builder", "Builds grounded internal context."),
    ToolSpec("source_ranking", "Ranks retrieved sources."),
    ToolSpec("source_citation", "Formats source citations."),
]

EVAL_TOOLS = [
    ToolSpec("hallucination_check", "Checks hallucination risk."),
    ToolSpec("source_grounding", "Checks source grounding."),
    ToolSpec("confidentiality_check", "Checks confidentiality risk."),
    ToolSpec("sales_claim_validation", "Checks sales claims."),
    ToolSpec("tone_brand_safety", "Checks tone and brand safety."),
    ToolSpec("completeness_check", "Checks completeness."),
    ToolSpec("final_approval", "Returns final eval status."),
]
