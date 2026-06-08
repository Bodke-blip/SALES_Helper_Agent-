import os
import json
import re
import time
from copy import deepcopy
from functools import lru_cache
from threading import Lock
from typing import Any

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import FieldCondition, Filter, MatchValue, PayloadSchemaType

from hybrid_retrieval import (
    BM25SparseEncoder,
    DENSE_VECTOR_NAME,
    HYBRID_FALLBACK_COLLECTION_NAME,
    HYBRID_COLLECTION_NAME,
    SPARSE_VECTOR_NAME,
)
from ingestion import (
    EMBEDDING_MODEL,
    QDRANT_API_KEY,
    QDRANT_URL,
)
from agents.state import SalesHelperState
from agents.tools import KNOWLEDGE_RETRIEVAL_TOOLS


AGENT_NAME = "knowledge_retrieval"
TOOLS = KNOWLEDGE_RETRIEVAL_TOOLS
DEFAULT_TOP_K = int(os.getenv("QDRANT_TOP_K", "15"))
HYBRID_PREFETCH_LIMIT = int(os.getenv("QDRANT_HYBRID_PREFETCH_LIMIT", "50"))
DOCUMENT_EXPANSION_SOURCE_LIMIT = int(os.getenv("QDRANT_DOCUMENT_EXPANSION_SOURCE_LIMIT", "3"))
DOCUMENT_EXPANSION_CHUNK_LIMIT = int(os.getenv("QDRANT_DOCUMENT_EXPANSION_CHUNK_LIMIT", "40"))
MAX_CONTEXT_ITEMS = int(os.getenv("QDRANT_MAX_CONTEXT_ITEMS", "30"))
MIN_RETRIEVAL_RESULTS = int(os.getenv("MIN_RETRIEVAL_RESULTS", "1"))
QDRANT_TIMEOUT_SECONDS = float(os.getenv("QDRANT_TIMEOUT_SECONDS", "8"))
ENSURE_PAYLOAD_INDEXES_ON_QUERY = os.getenv(
    "ENSURE_PAYLOAD_INDEXES_ON_QUERY",
    "",
).lower() in {"1", "true", "yes"}
RETRIEVAL_CACHE_TTL_SECONDS = int(os.getenv("RETRIEVAL_CACHE_TTL_SECONDS", "300"))
RETRIEVAL_CACHE_MAX_ENTRIES = int(os.getenv("RETRIEVAL_CACHE_MAX_ENTRIES", "256"))
retrieval_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
retrieval_cache_lock = Lock()

DOMAIN_ALIASES = {
    "healthcare": {"healthcare", "health care", "hospital", "clinical", "clinic"},
    "finance": {"finance", "financial", "banking", "bank", "tax", "accounting", "payroll"},
    "it professional services": {"it professional services", "it services", "technology services", "software services"},
}

COUNTRY_ALIASES = {
    "canada": {"canada", "canadian"},
    "united states": {"united states", "usa", "u s", "u s a", "us"},
    "united kingdom": {"united kingdom", "uk", "u k"},
    "costa rica": {"costa rica", "cost rica"},
    "philippines": {"philippines", "philippine"},
    "trinidad": {"trinidad"},
    "india": {"india"},
}

GENERIC_USECASE_NAMES = {
    "",
    "use case",
    "usecase",
    "workflow",
    "process",
    "solution",
    "overview",
    "agenda",
    "summary",
    "slide",
    "use case category",
}


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be configured.")

    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=QDRANT_TIMEOUT_SECONDS,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_sparse_encoder() -> BM25SparseEncoder:
    return BM25SparseEncoder.load()


@lru_cache(maxsize=8)
def ensure_payload_indexes(collection_name: str = HYBRID_COLLECTION_NAME) -> None:
    client = get_qdrant_client()
    indexes = {
        "document_id": PayloadSchemaType.KEYWORD,
        "drive_id": PayloadSchemaType.KEYWORD,
        "ppt_name": PayloadSchemaType.KEYWORD,
        "customer_name_normalized": PayloadSchemaType.KEYWORD,
        "company_name": PayloadSchemaType.TEXT,
        "use_case_name": PayloadSchemaType.TEXT,
        "source_type": PayloadSchemaType.KEYWORD,
        "is_internal": PayloadSchemaType.BOOL,
        "content_type": PayloadSchemaType.KEYWORD,
    }

    for field_name, field_schema in indexes.items():
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )
        except Exception as error:
            if "already exists" not in str(error).lower():
                print(f"Payload index unavailable for {field_name}: {error}")


def get_fallback_collections() -> list[str]:
    if not HYBRID_FALLBACK_COLLECTION_NAME:
        return []

    return [HYBRID_FALLBACK_COLLECTION_NAME]


def get_retrieval_collections() -> list[str]:
    return list(dict.fromkeys([HYBRID_COLLECTION_NAME, *get_fallback_collections()]))


def is_hybrid_collection(collection_name: str) -> bool:
    return collection_name in {HYBRID_COLLECTION_NAME, HYBRID_FALLBACK_COLLECTION_NAME}


def parse_jsonish_object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)

    text = str(value or "").strip()

    if not text:
        return {}

    try:
        parsed = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)

        if not match:
            return {}

        try:
            parsed = json.loads(match.group(0))
        except Exception:
            return {}

    return parsed if isinstance(parsed, dict) else {}


def retrieve_from_collection(
    query: str,
    *,
    collection_name: str,
    top_k: int = DEFAULT_TOP_K,
    customer_name_normalized: str = "",
) -> list[tuple]:
    client = get_qdrant_client()
    query_vector = get_embeddings().embed_query(query)
    query_filter = None

    if customer_name_normalized:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="customer_name_normalized",
                    match=MatchValue(value=customer_name_normalized),
                )
            ]
        )

    if is_hybrid_collection(collection_name):
        sparse_query_vector = get_sparse_encoder().encode_query(query)

        if sparse_query_vector.indices:
            response = client.query_points(
                collection_name=collection_name,
                prefetch=[
                    models.Prefetch(
                        query=query_vector,
                        using=DENSE_VECTOR_NAME,
                        filter=query_filter,
                        limit=max(HYBRID_PREFETCH_LIMIT, top_k),
                    ),
                    models.Prefetch(
                        query=sparse_query_vector,
                        using=SPARSE_VECTOR_NAME,
                        filter=query_filter,
                        limit=max(HYBRID_PREFETCH_LIMIT, top_k),
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
        else:
            response = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                using=DENSE_VECTOR_NAME,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
    else:
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
    results = []

    for point in response.points:
        payload = get_point_payload(point)
        document = Document(
            page_content=get_payload_text(payload),
            metadata=get_payload_metadata(payload),
        )
        results.append((document, float(point.score or 0.0)))

    return results


def infer_catalog_filters_from_query(query: str) -> dict[str, str]:
    lowered_query = query.lower()
    filters = {
        "action": "count" if re.search(r"\b(count|number|how many)\b", lowered_query) else "list",
        "company": "",
        "domain": "",
        "country": "",
    }

    company_match = re.search(
        r"\b(?:for|with|from|by)\s+(?:company|customer|client)\s+([a-z0-9&.'’\-\s]+?)(?:\s+(?:in|domain|use cases|usecases)|[?.!,]|$)",
        query,
        flags=re.IGNORECASE,
    )
    if not company_match:
        company_match = re.search(
            r"\buse\s*cases?\b.*?\b(?:for|from|with|by)\s+([a-z0-9&.'’\-\s]+?)(?:[?.!,]|$)",
            query,
            flags=re.IGNORECASE,
        )

    if company_match:
        filters["company"] = company_match.group(1).strip()

    domain_match = re.search(
        r"\b(?:in|for|with)\s+(?:the\s+)?([a-z0-9&.'’\-\s]+?)\s+(?:domain|industry|vertical)\b",
        query,
        flags=re.IGNORECASE,
    )
    if domain_match:
        filters["domain"] = domain_match.group(1).strip()

    if not filters["domain"]:
        filters["domain"] = infer_alias_filter(lowered_query, DOMAIN_ALIASES)

    country_match = re.search(
        r"\b(?:in|for|from|with)\s+(?:the\s+)?([a-z0-9&.'’\-\s]+?)\s+(?:country|region|market)\b",
        query,
        flags=re.IGNORECASE,
    )
    if country_match:
        filters["country"] = country_match.group(1).strip()

    if not filters["country"]:
        filters["country"] = infer_alias_filter(lowered_query, COUNTRY_ALIASES)

    return filters


def get_catalog_request(state: SalesHelperState) -> dict[str, str]:
    query = state.get("contextual_query") or state.get("user_query", "")
    request = infer_catalog_filters_from_query(query)
    parsed_input = parse_jsonish_object(state.get("orchestrator_tool_input", ""))

    for key in ("action", "company", "domain", "country"):
        value = str(parsed_input.get(key) or "").strip()

        if value:
            request[key] = value

    request["action"] = "count" if request.get("action", "").lower() == "count" else "list"
    request["domain"] = normalize_alias_value(request.get("domain", ""), DOMAIN_ALIASES)
    request["country"] = normalize_alias_value(request.get("country", ""), COUNTRY_ALIASES)
    return request


def normalized_contains(haystack: str, needle: str) -> bool:
    normalized_haystack = normalize_lookup_value(haystack)
    normalized_needle = normalize_lookup_value(needle)
    return bool(normalized_needle and normalized_needle in normalized_haystack)


def normalize_alias_value(value: str, aliases: dict[str, set[str]]) -> str:
    normalized_value = normalize_lookup_value(value)

    if not normalized_value:
        return ""

    for canonical_value, alias_values in aliases.items():
        if normalized_value == normalize_lookup_value(canonical_value):
            return canonical_value

        if any(normalized_value == normalize_lookup_value(alias) for alias in alias_values):
            return canonical_value

    return value.strip()


def infer_alias_filter(query: str, aliases: dict[str, set[str]]) -> str:
    normalized_query = normalize_lookup_value(query)

    for canonical_value, alias_values in aliases.items():
        candidates = {canonical_value, *alias_values}

        for candidate in candidates:
            normalized_candidate = normalize_lookup_value(candidate)

            if normalized_candidate and re.search(
                rf"\b{re.escape(normalized_candidate)}\b",
                normalized_query,
            ):
                return canonical_value

    return ""


def aliases_match_text(value: str, text: str, aliases: dict[str, set[str]]) -> bool:
    canonical_value = normalize_alias_value(value, aliases)
    normalized_text = normalize_lookup_value(text)
    candidates = aliases.get(canonical_value, {canonical_value})

    return any(
        re.search(rf"\b{re.escape(normalize_lookup_value(candidate))}\b", normalized_text)
        for candidate in candidates
        if normalize_lookup_value(candidate)
    )


def infer_countries_from_text(text: str) -> list[str]:
    return [
        canonical_country
        for canonical_country in COUNTRY_ALIASES
        if aliases_match_text(canonical_country, text, COUNTRY_ALIASES)
    ]


def has_explicit_usecase_name(metadata: dict) -> bool:
    return bool(str(metadata.get("usecase_name") or metadata.get("use_case_name") or "").strip())


def is_multi_usecase_ppt(metadata: dict) -> bool:
    ppt_name = str(metadata.get("ppt_name") or "")
    return bool(re.search(r"\buse\s*cases\b", ppt_name, flags=re.IGNORECASE))


def clean_inferred_usecase_name(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(value or "")).strip()
    cleaned = cleaned.strip(" .:-–—")
    cleaned = re.sub(
        r"^web based automation is used to integrate\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s+was done with the help of.+$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^(?:the\s+)?(?:process\s+of|workflow\s+for)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:automate|automating|automation\s+of)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+(?:process|workflow)$", lambda match: match.group(0), cleaned, flags=re.IGNORECASE)

    if not cleaned:
        return ""

    if len(cleaned) > 90:
        return ""

    if normalize_lookup_value(cleaned) in GENERIC_USECASE_NAMES:
        return ""

    if re.fullmatch(r"[<#›‹\s\d]+", cleaned):
        return ""

    if cleaned.lower().startswith(("the solution", "a process", "a workflow", "a systematic approach")):
        return ""

    if re.search(
        r"\b(does not explicitly state|please switch|used to automate the use case|bot was created to process)\b",
        cleaned,
        flags=re.IGNORECASE,
    ):
        return ""

    return cleaned


def titlecase_catalog_name(value: str) -> str:
    words_to_keep_lower = {"and", "or", "of", "the", "to", "in", "on", "for", "from", "with"}
    cleaned = clean_inferred_usecase_name(value)

    if not cleaned:
        return ""

    if any(character.isupper() for character in cleaned[1:]):
        return cleaned[0].upper() + cleaned[1:]

    words = []

    for index, word in enumerate(cleaned.split()):
        if index > 0 and word.lower() in words_to_keep_lower:
            words.append(word.lower())
        else:
            words.append(word[:1].upper() + word[1:])

    return " ".join(words)


def compact_process_name(value: str) -> str:
    cleaned = titlecase_catalog_name(value)

    replacements = [
        (r"^Managing Candidate Profiles in a Workday System$", "Candidate Profile Management"),
        (r"^Managing Employee Termination Reports$", "Employee Termination Report Management"),
        (r"^Distribute ITR and Merit Letter Documents on Workday Profile$", "ITR and Merit Letter Distribution"),
        (r"^HR Memo File From Stat Tracker$", "HR Memo File Creation from Stat Tracker"),
    ]

    for pattern, replacement in replacements:
        if re.match(pattern, cleaned, flags=re.IGNORECASE):
            return replacement

    return cleaned


def infer_usecase_name_from_text(text: str) -> str:
    normalized_text = re.sub(r"\s+", " ", str(text or "")).strip()

    patterns = [
        r"Solution Proposed:\s*Bot was created to\s+(?:automate\s+)?(.+?)(?:\s+using\b|\.| Workflow/Image Summary:| Tools Used:| Benefits:|$)",
        r"Solution Proposed:.*?\bBot will create\s+(.+?)(?:\s+and will\b|\.| Workflow/Image Summary:| Tools Used:| Benefits:|$)",
        r"Solution Proposed:\s*(?:The solution proposed is\s+)?(?:to\s+)?(?:automate\s+)?(.+?)(?:\s+using\b|\.| Workflow/Image Summary:| Tools Used:| Benefits:|$)",
        r"Use Case:\s*(.+?)(?: Use Case Category:| Solution Proposed:| Workflow/Image Summary:| Tools Used:| Benefits:|$)",
        r"Workflow/Image Summary:\s*(?:The workflow|The flowchart|The diagram|The slide)\s+(?:outlines|depicts|illustrates|shows|describes)\s+(?:a\s+)?(?:process|workflow)?\s*(?:for|to)\s+(.+?)(?:\.| Tools Used:| Benefits:|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized_text, flags=re.IGNORECASE)

        if not match:
            continue

        candidate = compact_process_name(match.group(1))

        if candidate:
            return candidate

    return ""


def meaningful_slide_title(metadata: dict) -> str:
    title = compact_process_name(str(metadata.get("slide_title") or ""))

    if not title:
        return ""

    if normalize_lookup_value(title) in {"number", "page", "slide"}:
        return ""

    return title


def clean_usecase_catalog_name(metadata: dict, text: str = "") -> str:
    usecase_name = (
        metadata.get("usecase_name")
        or metadata.get("use_case_name")
        or ""
    )

    if str(usecase_name).strip():
        return str(usecase_name).strip()

    ppt_name = str(metadata.get("ppt_name") or "").strip()
    ppt_base = re.sub(r"\.pptx$", "", ppt_name, flags=re.IGNORECASE).strip()

    inferred_name = infer_usecase_name_from_text(text)

    if inferred_name:
        return inferred_name

    if is_multi_usecase_ppt(metadata):
        slide_title = meaningful_slide_title(metadata)

        if slide_title:
            return slide_title

    if is_multi_usecase_ppt(metadata) and metadata.get("slide_number"):
        return f"{ppt_base} - Slide {metadata['slide_number']}"

    return ppt_base


def is_fallback_catalog_name(entry: dict[str, Any]) -> bool:
    usecase_name = str(entry.get("usecase_name") or "")
    ppt_base = re.sub(
        r"\.pptx$",
        "",
        str(entry.get("ppt_name") or ""),
        flags=re.IGNORECASE,
    )

    return (
        not usecase_name
        or usecase_name == "Use case not named"
        or usecase_name == ppt_base
        or bool(re.search(r"\bSlide\s+\d+\b", usecase_name, flags=re.IGNORECASE))
    )


def catalog_name_quality(entry: dict[str, Any]) -> int:
    if is_fallback_catalog_name(entry):
        return 0

    name = str(entry.get("usecase_name") or "")

    if re.search(r"\b(process|automation|management|upload|deactivation|reconciliation|chatbot|assistant|bot|generation|distribution|creation)\b", name, flags=re.IGNORECASE):
        return 3

    return 2


def build_usecase_catalog_entry(payload: dict, collection_name: str) -> dict[str, Any]:
    metadata = get_payload_metadata(payload)
    text = get_payload_text(payload)
    usecase_name = clean_usecase_catalog_name(metadata, text)
    company_name = (
        metadata.get("customer_name")
        or metadata.get("company_name")
        or re.sub(r"\.pptx$", "", str(metadata.get("ppt_name") or ""), flags=re.IGNORECASE)
        or "Unknown company"
    )
    search_text = " ".join(
        str(value or "")
        for value in (
            company_name,
            metadata.get("customer_domain"),
            usecase_name,
            metadata.get("use_case_category"),
            metadata.get("ppt_name"),
            text,
        )
    )

    return {
        "collection": collection_name,
        "company_name": str(company_name).strip(),
        "customer_domain": str(metadata.get("customer_domain") or "").strip(),
        "countries": infer_countries_from_text(search_text),
        "usecase_name": usecase_name or "Use case not named",
        "use_case_category": str(metadata.get("use_case_category") or "").strip(),
        "ppt_name": str(metadata.get("ppt_name") or "").strip(),
        "drive_id": str(metadata.get("drive_id") or metadata.get("document_id") or "").strip(),
        "slides": [],
        "search_text": search_text,
        "dedupe_scope": "explicit_usecase"
        if has_explicit_usecase_name(metadata)
        else ("slide" if is_multi_usecase_ppt(metadata) else "ppt"),
    }


def usecase_catalog_key(entry: dict[str, Any]) -> tuple[str, str, str]:
    if entry.get("dedupe_scope") == "slide":
        slide_key = ",".join(str(slide) for slide in entry.get("slides", []))

        return (
            normalize_lookup_value(entry.get("company_name", "")),
            normalize_lookup_value(entry.get("usecase_name", "")),
            f"{entry.get('drive_id') or ''}:{slide_key}",
        )

    if entry.get("dedupe_scope") == "ppt":
        return (
            normalize_lookup_value(entry.get("company_name", "")),
            "ppt",
            str(entry.get("drive_id") or ""),
        )

    return (
        normalize_lookup_value(entry.get("company_name", "")),
        normalize_lookup_value(entry.get("usecase_name", "")),
        str(entry.get("drive_id") or ""),
    )


def scroll_collection_payloads(collection_name: str, *, limit: int = 256) -> list[dict]:
    client = get_qdrant_client()
    payloads = []
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payloads.append(get_point_payload(point))

        if offset is None:
            break

    return payloads


def collect_usecase_catalog(collection_name: str) -> list[dict[str, Any]]:
    entries_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}

    for payload in scroll_collection_payloads(collection_name):
        entry = build_usecase_catalog_entry(payload, collection_name)

        if not entry["usecase_name"] or entry["usecase_name"] == "Use case not named":
            continue

        key = usecase_catalog_key(entry)
        existing = entries_by_key.get(key)

        if not existing:
            existing = entry
            entries_by_key[key] = existing

        elif catalog_name_quality(entry) > catalog_name_quality(existing):
            existing["usecase_name"] = entry["usecase_name"]

        metadata = get_payload_metadata(payload)
        slide_number = metadata.get("slide_number")

        if slide_number and slide_number not in existing["slides"]:
            existing["slides"].append(slide_number)

        for field_name in ("customer_domain", "use_case_category", "ppt_name"):
            if not existing.get(field_name) and entry.get(field_name):
                existing[field_name] = entry[field_name]

        for country in entry.get("countries", []):
            if country not in existing["countries"]:
                existing["countries"].append(country)

        if entry.get("search_text"):
            existing["search_text"] = " ".join(
                [existing.get("search_text", ""), entry["search_text"]]
            ).strip()

    entries = list(entries_by_key.values())

    for entry in entries:
        entry["slides"] = sorted(
            entry["slides"],
            key=lambda value: int(value) if str(value).isdigit() else 9999,
        )

    def catalog_sort_key(item: dict[str, Any]) -> tuple:
        first_slide = item["slides"][0] if item.get("slides") else 9999
        first_slide = int(first_slide) if str(first_slide).isdigit() else 9999

        return (
            normalize_lookup_value(item.get("company_name", "")),
            0 if item.get("dedupe_scope") != "slide" else 1,
            first_slide,
            normalize_lookup_value(item.get("usecase_name", "")),
        )

    return sorted(entries, key=catalog_sort_key)


def filter_usecase_catalog(
    entries: list[dict[str, Any]],
    *,
    company: str = "",
    domain: str = "",
    country: str = "",
) -> list[dict[str, Any]]:
    filtered_entries = []

    for entry in entries:
        entry_search_text = entry.get("search_text", "")

        if company and not (
            normalized_contains(entry.get("company_name", ""), company)
            or normalized_contains(entry.get("ppt_name", ""), company)
        ):
            continue

        if domain:
            entry_domain = entry.get("customer_domain", "")

            if entry_domain:
                if not aliases_match_text(domain, entry_domain, DOMAIN_ALIASES):
                    continue
            elif not aliases_match_text(domain, entry_search_text, DOMAIN_ALIASES):
                continue

        if country and not (
            normalize_alias_value(country, COUNTRY_ALIASES) in entry.get("countries", [])
            or aliases_match_text(country, entry_search_text, COUNTRY_ALIASES)
        ):
            continue

        filtered_entries.append(entry)

    return filtered_entries


def build_catalog_context_text(catalog_result: dict[str, Any]) -> str:
    lines = [
        "Catalog result from Qdrant metadata.",
        f"Collection: {catalog_result['collection']}",
        f"Total matching use cases: {catalog_result['total_matching_use_cases']}",
    ]

    if catalog_result.get("filters"):
        lines.append(f"Filters: {catalog_result['filters']}")

    lines.append("Use cases:")

    for index, usecase in enumerate(catalog_result["use_cases"], start=1):
        company = usecase.get("company_name") or "Unknown company"
        name = usecase.get("usecase_name") or "Use case not named"
        domain = usecase.get("customer_domain") or "Domain not specified"
        countries = ", ".join(usecase.get("countries", [])) or "Country not specified"
        ppt_name = usecase.get("ppt_name") or "Source PPT not specified"
        lines.append(
            f"{index}. {name} | Company: {company} | Domain: {domain} | "
            f"Country: {countries} | Source: {ppt_name}"
        )

    return "\n".join(lines)


def build_catalog_sources(
    entries: list[dict[str, Any]],
    collection_name: str,
    *,
    max_sources: int = 20,
) -> list[dict[str, Any]]:
    sources = []

    for entry in entries[:max_sources]:
        sources.append(
            {
                "collection": collection_name,
                "customer_name": entry.get("company_name", ""),
                "usecase_name": entry.get("usecase_name", ""),
                "customer_domain": entry.get("customer_domain", ""),
                "countries": entry.get("countries", []),
                "drive_id": entry.get("drive_id", ""),
                "match_key": "",
                "ppt_name": entry.get("ppt_name", ""),
                "slide_number": entry["slides"][0] if entry.get("slides") else None,
                "chunk_type": "usecase_catalog",
                "content_type": "metadata_catalog",
                "score": 1.0,
            }
        )

    return sources


def public_catalog_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "collection": entry.get("collection", ""),
        "company_name": entry.get("company_name", ""),
        "customer_domain": entry.get("customer_domain", ""),
        "countries": entry.get("countries", []),
        "usecase_name": entry.get("usecase_name", ""),
        "use_case_category": entry.get("use_case_category", ""),
        "ppt_name": entry.get("ppt_name", ""),
        "drive_id": entry.get("drive_id", ""),
        "slides": entry.get("slides", []),
    }


def retrieve_usecase_catalog(state: SalesHelperState) -> dict[str, Any]:
    catalog_request = get_catalog_request(state)
    retrieval_error = ""

    for collection_name in get_retrieval_collections():
        try:
            entries = collect_usecase_catalog(collection_name)
            filtered_entries = filter_usecase_catalog(
                entries,
                company=catalog_request.get("company", ""),
                domain=catalog_request.get("domain", ""),
                country=catalog_request.get("country", ""),
            )
        except Exception as error:
            retrieval_error = str(error)
            continue

        if filtered_entries:
            public_entries = [public_catalog_entry(entry) for entry in filtered_entries]
            catalog_result = {
                "type": "catalog_result",
                "action": catalog_request["action"],
                "collection": collection_name,
                "filters": {
                    key: value
                    for key, value in {
                        "company": catalog_request.get("company", ""),
                        "domain": catalog_request.get("domain", ""),
                        "country": catalog_request.get("country", ""),
                    }.items()
                    if value
                },
                "total_matching_use_cases": len(public_entries),
                "use_cases": public_entries,
            }
            return {
                "retrieved_context": [
                    {
                        "rank": 1,
                        "score": 1.0,
                        "text": build_catalog_context_text(catalog_result),
                        "catalog_result": catalog_result,
                        "collection": collection_name,
                    }
                ],
                "retrieved_sources": build_catalog_sources(public_entries, collection_name),
                "used_collection": collection_name,
                "retrieval_error": "",
                "requested_customer_name": catalog_request.get("company", ""),
            }

    return {
        "retrieved_context": [],
        "retrieved_sources": [],
        "used_collection": HYBRID_COLLECTION_NAME,
        "retrieval_error": retrieval_error,
        "requested_customer_name": catalog_request.get("company", ""),
    }


def normalize_lookup_value(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def normalize_query_for_retrieval_cache(query: str) -> str:
    return " ".join(query.lower().strip().split())


def build_retrieval_cache_key(
    *,
    query: str,
    collections: list[str],
) -> tuple[Any, ...]:
    return (
        normalize_query_for_retrieval_cache(query),
        tuple(collections),
        HYBRID_COLLECTION_NAME,
        HYBRID_FALLBACK_COLLECTION_NAME,
        DEFAULT_TOP_K,
        HYBRID_PREFETCH_LIMIT,
        DOCUMENT_EXPANSION_SOURCE_LIMIT,
        DOCUMENT_EXPANSION_CHUNK_LIMIT,
        MAX_CONTEXT_ITEMS,
        MIN_RETRIEVAL_RESULTS,
    )


def get_cached_retrieval(cache_key: tuple[Any, ...]) -> dict[str, Any] | None:
    if RETRIEVAL_CACHE_TTL_SECONDS <= 0:
        return None

    now = time.monotonic()

    with retrieval_cache_lock:
        cached_item = retrieval_cache.get(cache_key)

        if not cached_item:
            return None

        if now - cached_item["created_at"] > RETRIEVAL_CACHE_TTL_SECONDS:
            retrieval_cache.pop(cache_key, None)
            return None

        return deepcopy(cached_item["payload"])


def set_cached_retrieval(cache_key: tuple[Any, ...], payload: dict[str, Any]) -> None:
    if RETRIEVAL_CACHE_TTL_SECONDS <= 0:
        return

    with retrieval_cache_lock:
        if len(retrieval_cache) >= RETRIEVAL_CACHE_MAX_ENTRIES:
            oldest_key = min(
                retrieval_cache,
                key=lambda key: retrieval_cache[key]["created_at"],
            )
            retrieval_cache.pop(oldest_key, None)

        retrieval_cache[cache_key] = {
            "created_at": time.monotonic(),
            "payload": deepcopy(payload),
        }


def retrieval_cache_status() -> dict[str, Any]:
    with retrieval_cache_lock:
        return {
            "enabled": RETRIEVAL_CACHE_TTL_SECONDS > 0,
            "entries": len(retrieval_cache),
            "ttl_seconds": RETRIEVAL_CACHE_TTL_SECONDS,
            "max_entries": RETRIEVAL_CACHE_MAX_ENTRIES,
            "scope": "knowledge_retrieval",
        }


def clear_retrieval_cache() -> dict[str, Any]:
    with retrieval_cache_lock:
        cleared_entries = len(retrieval_cache)
        retrieval_cache.clear()

    return {
        "status": "cleared",
        "entries_cleared": cleared_entries,
        "scope": "knowledge_retrieval",
    }


def warm_retrieval_models() -> None:
    get_embeddings()
    get_sparse_encoder()


def get_point_payload(point: Any) -> dict:
    return dict(point.payload or {})


def get_payload_metadata(payload: dict) -> dict:
    metadata = payload.get("metadata")

    if isinstance(metadata, dict):
        metadata = dict(metadata)
    else:
        metadata = {
            key: payload.get(key, "")
            for key in (
                "vector_point_id",
                "document_id",
                "ppt_name",
                "slide_number",
                "slide_title",
                "company_name",
                "customer_name_normalized",
                "customer_domain",
                "use_case_name",
                "use_case_category",
                "solution_proposed",
                "workflow_image_summary",
                "tools_used",
                "benefits",
                "chunk_type",
                "content_type",
                "source_type",
                "is_internal",
                "usecase_name",
                "customer_name",
                "drive_id",
                "match_key",
            )
        }

    company_name = metadata.get("company_name") or metadata.get("customer_name") or ""
    use_case_name = metadata.get("use_case_name") or metadata.get("usecase_name") or ""
    document_id = metadata.get("document_id") or metadata.get("drive_id") or ""

    metadata["company_name"] = metadata.get("company_name") or company_name
    metadata["customer_name"] = metadata.get("customer_name") or company_name
    metadata["customer_name_normalized"] = (
        metadata.get("customer_name_normalized")
        or normalize_lookup_value(str(company_name))
    )
    metadata["use_case_name"] = metadata.get("use_case_name") or use_case_name
    metadata["usecase_name"] = metadata.get("usecase_name") or use_case_name
    metadata["document_id"] = metadata.get("document_id") or document_id
    metadata["drive_id"] = metadata.get("drive_id") or document_id

    return metadata


def get_payload_text(payload: dict) -> str:
    return str(
        payload.get("page_content")
        or payload.get("content")
        or payload.get("text")
        or ""
    )


def build_context_and_sources(results: list[tuple], collection_name: str) -> tuple[list[dict], list[dict]]:
    internal_context = []
    qdrant_sources = []
    seen_sources = set()

    for rank, (document, score) in enumerate(results, start=1):
        metadata = dict(document.metadata or {})
        context_item = {
            "rank": rank,
            "score": float(score),
            "text": document.page_content,
            **metadata,
        }
        internal_context.append(context_item)

        source_key = (
            metadata.get("drive_id") or metadata.get("document_id"),
            metadata.get("customer_name") or metadata.get("company_name"),
            metadata.get("usecase_name") or metadata.get("use_case_name"),
        )

        if source_key in seen_sources:
            continue

        seen_sources.add(source_key)
        qdrant_sources.append(
            {
                "collection": collection_name,
                "customer_name": metadata.get("customer_name") or metadata.get("company_name", ""),
                "usecase_name": metadata.get("usecase_name") or metadata.get("use_case_name", ""),
                "customer_domain": metadata.get("customer_domain", ""),
                "drive_id": metadata.get("drive_id") or metadata.get("document_id", ""),
                "match_key": metadata.get("match_key", ""),
                "ppt_name": metadata.get("ppt_name", ""),
                "slide_number": metadata.get("slide_number"),
                "chunk_type": metadata.get("chunk_type", ""),
                "content_type": metadata.get("content_type", ""),
                "score": float(score),
            }
        )

    return internal_context, qdrant_sources


def result_key(document: Document) -> tuple:
    metadata = dict(document.metadata or {})
    return (
        metadata.get("vector_point_id", ""),
        metadata.get("document_id") or metadata.get("drive_id", ""),
        metadata.get("slide_number", ""),
        metadata.get("chunk_type", ""),
        document.page_content[:120],
    )


def sort_results_for_context(results: list[tuple]) -> list[tuple]:
    def sort_key(item: tuple) -> tuple:
        document, score = item
        metadata = dict(document.metadata or {})
        slide_number = metadata.get("slide_number")

        if isinstance(slide_number, str) and slide_number.isdigit():
            slide_number = int(slide_number)

        return (
            -(float(score or 0.0)),
            str(metadata.get("document_id") or metadata.get("drive_id") or ""),
            slide_number if isinstance(slide_number, int) else 9999,
            str(metadata.get("chunk_type") or ""),
        )

    return sorted(results, key=sort_key)


def scroll_related_document_chunks(
    *,
    collection_name: str,
    field_name: str,
    field_value: str,
    seed_score: float,
) -> list[tuple]:
    if not field_value:
        return []

    client = get_qdrant_client()
    try:
        points, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key=field_name,
                        match=MatchValue(value=field_value),
                    )
                ]
            ),
            limit=DOCUMENT_EXPANSION_CHUNK_LIMIT,
            with_payload=True,
            with_vectors=False,
        )
    except Exception:
        return []
    related_results = []

    for point in points:
        payload = get_point_payload(point)
        related_results.append(
            (
                Document(
                    page_content=get_payload_text(payload),
                    metadata=get_payload_metadata(payload),
                ),
                max(float(seed_score or 0.0) * 0.95, 0.0001),
            )
        )

    return related_results


def expand_results_with_document_chunks(results: list[tuple], collection_name: str) -> list[tuple]:
    if not results or DOCUMENT_EXPANSION_SOURCE_LIMIT <= 0:
        return results

    expanded_results = list(results)
    seen_keys = {result_key(document) for document, _ in expanded_results}
    expanded_source_ids = set()

    for document, score in results[:DOCUMENT_EXPANSION_SOURCE_LIMIT]:
        metadata = dict(document.metadata or {})
        source_id = str(metadata.get("document_id") or metadata.get("drive_id") or "").strip()

        if not source_id or source_id in expanded_source_ids:
            continue

        expanded_source_ids.add(source_id)
        related_results = scroll_related_document_chunks(
            collection_name=collection_name,
            field_name="document_id",
            field_value=source_id,
            seed_score=score,
        )

        if not related_results and metadata.get("drive_id"):
            related_results = scroll_related_document_chunks(
                collection_name=collection_name,
                field_name="drive_id",
                field_value=str(metadata.get("drive_id")),
                seed_score=score,
            )

        for related_document, related_score in related_results:
            key = result_key(related_document)

            if key in seen_keys:
                continue

            seen_keys.add(key)
            expanded_results.append((related_document, related_score))

    return sort_results_for_context(expanded_results)[:MAX_CONTEXT_ITEMS]


def retrieve_context_from_single_collection(query: str, collection_name: str) -> dict[str, Any]:
    if ENSURE_PAYLOAD_INDEXES_ON_QUERY or DOCUMENT_EXPANSION_SOURCE_LIMIT > 0:
        ensure_payload_indexes(collection_name)
    results = retrieve_from_collection(query, collection_name=collection_name)
    results = expand_results_with_document_chunks(results, collection_name)
    retrieved_context = []
    retrieved_sources = []

    if len(results) >= MIN_RETRIEVAL_RESULTS:
        retrieved_context, retrieved_sources = build_context_and_sources(results, collection_name)

    return {
        "results": results,
        "retrieved_context": retrieved_context,
        "retrieved_sources": retrieved_sources,
        "requested_customer_name": "",
    }


def knowledge_retrieval_agent(state: SalesHelperState) -> SalesHelperState:
    query = state.get("contextual_query") or state.get("user_query", "")

    if state.get("orchestrator_tool") == "usecase_catalog":
        catalog_retrieval = retrieve_usecase_catalog(state)

        return {
            **state,
            "internal_context": [
                *state.get("internal_context", []),
                *catalog_retrieval.get("retrieved_context", []),
            ],
            "qdrant_sources": [
                *state.get("qdrant_sources", []),
                *catalog_retrieval.get("retrieved_sources", []),
            ],
            "retrieval_collection": catalog_retrieval.get("used_collection", HYBRID_COLLECTION_NAME),
            "retrieval_error": catalog_retrieval.get("retrieval_error", ""),
            "retrieval_customer_filter": catalog_retrieval.get("requested_customer_name", ""),
            "retrieval_cache_status": "catalog_scan",
        }

    collections = get_retrieval_collections()
    cache_key = build_retrieval_cache_key(query=query, collections=collections)
    cached_retrieval = get_cached_retrieval(cache_key)

    if cached_retrieval:
        return {
            **state,
            "internal_context": [
                *state.get("internal_context", []),
                *cached_retrieval.get("retrieved_context", []),
            ],
            "qdrant_sources": [
                *state.get("qdrant_sources", []),
                *cached_retrieval.get("retrieved_sources", []),
            ],
            "retrieval_collection": cached_retrieval.get("used_collection", HYBRID_COLLECTION_NAME),
            "retrieval_error": cached_retrieval.get("retrieval_error", ""),
            "retrieval_customer_filter": cached_retrieval.get("requested_customer_name", ""),
            "retrieval_cache_status": "hit",
        }

    retrieval_error = ""
    results = []
    used_collection = HYBRID_COLLECTION_NAME
    retrieved_context = []
    retrieved_sources = []
    requested_customer_name = ""

    for collection_name in collections:
        try:
            collection_result = retrieve_context_from_single_collection(query, collection_name)
            results = collection_result["results"]
            retrieved_context = collection_result["retrieved_context"]
            retrieved_sources = collection_result["retrieved_sources"]
            requested_customer_name = collection_result["requested_customer_name"]

            if (
                len(retrieved_context) >= MIN_RETRIEVAL_RESULTS
                or len(retrieved_sources) >= MIN_RETRIEVAL_RESULTS
            ):
                used_collection = collection_name
                break
        except Exception as error:
            retrieval_error = str(error)
            results = []
            continue

    if not retrieved_context and results:
        retrieved_context, retrieved_sources = build_context_and_sources(results, used_collection)

    internal_context = [*state.get("internal_context", []), *retrieved_context]
    qdrant_sources = [*state.get("qdrant_sources", []), *retrieved_sources]

    cache_payload = {
        "retrieved_context": retrieved_context,
        "retrieved_sources": retrieved_sources,
        "used_collection": used_collection,
        "retrieval_error": retrieval_error,
        "requested_customer_name": requested_customer_name,
    }

    if retrieved_context or retrieved_sources:
        set_cached_retrieval(cache_key, cache_payload)

    return {
        **state,
        "internal_context": internal_context,
        "qdrant_sources": qdrant_sources,
        "retrieval_collection": used_collection,
        "retrieval_error": retrieval_error,
        "retrieval_customer_filter": requested_customer_name,
        "retrieval_cache_status": "miss",
    }
