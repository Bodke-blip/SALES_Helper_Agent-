import os
import requests
import time
from contextvars import ContextVar
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

PRIMARY_LLM_MODEL = "gemini-2.5-flash"
FALLBACK_LLM_MODEL = "llama3.2:3b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "10"))
REQUEST_FALLBACK_AFTER_SECONDS = float(os.getenv("REQUEST_FALLBACK_AFTER_SECONDS", "999"))
PRIMARY_LLM_TIMEOUT_SECONDS = float(os.getenv("PRIMARY_LLM_TIMEOUT_SECONDS", "12"))
_request_started_at: ContextVar[float | None] = ContextVar(
    "request_started_at",
    default=None,
)
_use_gemini_llm: ContextVar[bool] = ContextVar("use_gemini_llm", default=True)
_use_local_llm: ContextVar[bool] = ContextVar("use_local_llm", default=True)


class LLMGatewayError(RuntimeError):
    pass


def set_request_started_at(started_at: float | None):
    return _request_started_at.set(started_at)


def reset_request_started_at(token) -> None:
    _request_started_at.reset(token)


def set_llm_preferences(*, use_gemini: bool, use_local: bool):
    gemini_token = _use_gemini_llm.set(use_gemini)
    local_token = _use_local_llm.set(use_local)
    return gemini_token, local_token


def reset_llm_preferences(tokens) -> None:
    gemini_token, local_token = tokens
    _use_gemini_llm.reset(gemini_token)
    _use_local_llm.reset(local_token)


def request_elapsed_seconds() -> float | None:
    started_at = _request_started_at.get()

    if started_at is None:
        return None

    return time.perf_counter() - started_at


def should_skip_primary_llm() -> bool:
    elapsed = request_elapsed_seconds()
    return elapsed is not None and elapsed >= REQUEST_FALLBACK_AFTER_SECONDS


def gemini_enabled() -> bool:
    return _use_gemini_llm.get()


def local_llm_enabled() -> bool:
    return _use_local_llm.get()


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
def get_fallback_llm_available() -> bool:
    try:
        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=3,
        )
        response.raise_for_status()
    except requests.RequestException:
        return False

    models = response.json().get("models", [])
    return any(model.get("name") == FALLBACK_LLM_MODEL for model in models)


def invoke_ollama(system_prompt: str, user_prompt: str) -> str:
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": FALLBACK_LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 900,
                },
            },
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise LLMGatewayError(f"Ollama fallback failed: {error}") from error

    content = response.json().get("message", {}).get("content", "")

    if not content:
        raise LLMGatewayError("Ollama fallback returned an empty response.")

    return str(content)


def get_llm_provider_status() -> dict[str, Any]:
    return {
        "primary_model": PRIMARY_LLM_MODEL,
        "primary_enabled": gemini_enabled(),
        "primary_available": gemini_enabled() and get_primary_llm() is not None,
        "fallback_model": FALLBACK_LLM_MODEL,
        "fallback_provider": "ollama",
        "fallback_enabled": local_llm_enabled(),
        "fallback_available": local_llm_enabled() and get_fallback_llm_available(),
        "fallback_after_seconds": REQUEST_FALLBACK_AFTER_SECONDS,
        "primary_timeout_seconds": PRIMARY_LLM_TIMEOUT_SECONDS,
    }


def invoke_llm(
    *,
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, str]:
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    primary = None if should_skip_primary_llm() or not gemini_enabled() else get_primary_llm()

    if primary is not None:
        try:
            response = primary.invoke(messages)
            return str(response.content), PRIMARY_LLM_MODEL
        except Exception:
            pass

    if local_llm_enabled() and get_fallback_llm_available():
        return invoke_ollama(system_prompt, user_prompt), FALLBACK_LLM_MODEL

    raise LLMGatewayError(
        "No enabled LLM provider is available. Enable Gemini or local Ollama."
    )
