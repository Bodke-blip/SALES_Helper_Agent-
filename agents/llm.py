import os
from contextvars import ContextVar
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

PRIMARY_LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
SECONDARY_LLM_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite")
PRIMARY_LLM_TIMEOUT_SECONDS = float(os.getenv("PRIMARY_LLM_TIMEOUT_SECONDS", "45"))
_use_gemini_llm: ContextVar[bool] = ContextVar("use_gemini_llm", default=True)


class LLMGatewayError(RuntimeError):
    pass


def set_request_started_at(started_at: float | None):
    return None


def reset_request_started_at(token) -> None:
    return None


def set_llm_preferences(*, use_gemini: bool, use_local: bool):
    return _use_gemini_llm.set(use_gemini)


def reset_llm_preferences(tokens) -> None:
    _use_gemini_llm.reset(tokens)


def gemini_enabled() -> bool:
    return _use_gemini_llm.get()


@lru_cache(maxsize=1)
def get_primary_llm():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None

    return ChatGoogleGenerativeAI(
        model=PRIMARY_LLM_MODEL,
        google_api_key=api_key,
        temperature=0.2,
        request_timeout=PRIMARY_LLM_TIMEOUT_SECONDS,
        retries=1,
    )


@lru_cache(maxsize=1)
def get_secondary_llm():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or SECONDARY_LLM_MODEL == PRIMARY_LLM_MODEL:
        return None

    return ChatGoogleGenerativeAI(
        model=SECONDARY_LLM_MODEL,
        google_api_key=api_key,
        temperature=0.2,
        request_timeout=PRIMARY_LLM_TIMEOUT_SECONDS,
        retries=1,
    )


def get_llm_provider_status() -> dict[str, Any]:
    return {
        "primary_model": PRIMARY_LLM_MODEL,
        "secondary_model": SECONDARY_LLM_MODEL,
        "primary_provider": "gemini",
        "primary_enabled": gemini_enabled(),
        "primary_available": gemini_enabled() and get_primary_llm() is not None,
        "secondary_available": gemini_enabled() and get_secondary_llm() is not None,
        "primary_timeout_seconds": PRIMARY_LLM_TIMEOUT_SECONDS,
    }


def invoke_llm(
    *,
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, str]:
    primary = get_primary_llm() if gemini_enabled() else None

    if primary is None:
        raise LLMGatewayError("Gemini is not available. Configure GEMINI_API_KEY.")

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    errors = []

    for model, model_name in (
        (primary, PRIMARY_LLM_MODEL),
        (get_secondary_llm() if gemini_enabled() else None, SECONDARY_LLM_MODEL),
    ):
        if model is None:
            continue

        try:
            response = model.invoke(messages)
            return str(response.content), model_name
        except Exception as error:
            errors.append(f"{model_name}: {error}")

    raise LLMGatewayError(f"Gemini request failed: {' | '.join(errors)}")
