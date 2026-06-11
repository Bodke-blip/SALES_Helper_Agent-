import os
import re
from datetime import UTC, datetime
from threading import Lock
from typing import Any


MAX_CHAT_HISTORY_TURNS = int(os.getenv("MAX_CHAT_HISTORY_TURNS", "6"))
CHAT_HISTORY_LIMIT = int(os.getenv("CHAT_HISTORY_LIMIT", "40"))
CHAT_DB_URL = os.getenv("CHAT_DB_URL", "")

chat_sessions: dict[str, dict[str, Any]] = {}
chat_messages: dict[str, list[dict[str, Any]]] = {}
chat_sessions_lock = Lock()


def postgres_enabled() -> bool:
    return bool(CHAT_DB_URL)


def get_postgres_connection():
    import psycopg
    from psycopg.rows import dict_row

    return psycopg.connect(CHAT_DB_URL, row_factory=dict_row)


def initialize_chat_store() -> None:
    if not postgres_enabled():
        return

    with get_postgres_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                create table if not exists chat_sessions (
                    id text primary key,
                    title text not null,
                    created_at timestamptz not null default now(),
                    updated_at timestamptz not null default now()
                )
                """
            )
            cursor.execute(
                """
                create table if not exists chat_messages (
                    id bigserial primary key,
                    session_id text not null references chat_sessions(id) on delete cascade,
                    role text not null check (role in ('user', 'assistant')),
                    content text not null,
                    metadata jsonb not null default '{}',
                    created_at timestamptz not null default now()
                )
                """
            )
            cursor.execute(
                """
                create index if not exists chat_messages_session_created_idx
                on chat_messages (session_id, created_at)
                """
            )
            cursor.execute(
                """
                create index if not exists chat_sessions_updated_idx
                on chat_sessions (updated_at desc)
                """
            )
        conn.commit()


def build_chat_title(user_query: str) -> str:
    title = " ".join(user_query.strip().split())

    if not title:
        return "New chat"

    return title[:57] + "..." if len(title) > 60 else title


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def normalize_message(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": str(row.get("role", "")),
        "content": str(row.get("content", "")),
        "metadata": row.get("metadata") or {},
    }


def list_chat_sessions(limit: int = 50) -> list[dict[str, Any]]:
    if postgres_enabled():
        initialize_chat_store()
        from psycopg.types.json import Jsonb

        with get_postgres_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    select id, title, created_at, updated_at
                    from chat_sessions
                    order by updated_at desc
                    limit %s
                    """,
                    (limit,),
                )
                return [dict(row) for row in cursor.fetchall()]

    with chat_sessions_lock:
        sessions = sorted(
            chat_sessions.values(),
            key=lambda item: item.get("updated_at", ""),
            reverse=True,
        )
        return [dict(item) for item in sessions[:limit]]


def get_chat_history(session_id: str, limit: int | None = None) -> list[dict[str, str]]:
    message_limit = limit or CHAT_HISTORY_LIMIT

    if postgres_enabled():
        initialize_chat_store()

        with get_postgres_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    select role, content, metadata
                    from (
                        select role, content, metadata, created_at, id
                        from chat_messages
                        where session_id = %s
                        order by created_at desc, id desc
                        limit %s
                    ) recent_messages
                    order by created_at asc, id asc
                    """,
                    (session_id, message_limit),
                )
                return [normalize_message(dict(row)) for row in cursor.fetchall()]

    with chat_sessions_lock:
        return [
            normalize_message(item)
            for item in chat_messages.get(session_id, [])[-message_limit:]
        ]


def get_chat_session(session_id: str) -> dict[str, Any]:
    if postgres_enabled():
        initialize_chat_store()

        with get_postgres_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    select id, title, created_at, updated_at
                    from chat_sessions
                    where id = %s
                    """,
                    (session_id,),
                )
                session = cursor.fetchone()

        return dict(session) if session else {}

    with chat_sessions_lock:
        return dict(chat_sessions.get(session_id, {}))


def get_chat_with_messages(session_id: str) -> dict[str, Any]:
    session = get_chat_session(session_id)
    messages = get_chat_history(session_id, limit=200)

    return {
        "session": session,
        "messages": messages,
    }


def append_chat_turn(
    session_id: str,
    user_query: str,
    assistant_answer: str,
    assistant_metadata: dict[str, Any] | None = None,
) -> None:
    title = build_chat_title(user_query)
    metadata = assistant_metadata or {}

    if postgres_enabled():
        initialize_chat_store()

        with get_postgres_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    insert into chat_sessions (id, title)
                    values (%s, %s)
                    on conflict (id) do nothing
                    """,
                    (session_id, title),
                )
                cursor.execute(
                    """
                    insert into chat_messages (session_id, role, content, metadata)
                    values
                        (%s, 'user', %s, '{}'),
                        (%s, 'assistant', %s, %s)
                    """,
                    (session_id, user_query, session_id, assistant_answer, Jsonb(metadata)),
                )
                cursor.execute(
                    """
                    update chat_sessions
                    set updated_at = now()
                    where id = %s
                    """,
                    (session_id,),
                )
            conn.commit()
        return

    timestamp = now_iso()

    with chat_sessions_lock:
        chat_sessions.setdefault(
            session_id,
            {
                "id": session_id,
                "title": title,
                "created_at": timestamp,
                "updated_at": timestamp,
            },
        )
        chat_sessions[session_id]["updated_at"] = timestamp
        messages = chat_messages.setdefault(session_id, [])
        messages.extend(
            [
                {"role": "user", "content": user_query, "created_at": timestamp},
                {
                    "role": "assistant",
                    "content": assistant_answer,
                    "metadata": metadata,
                    "created_at": timestamp,
                },
            ]
        )


def clear_chat_history(session_id: str) -> None:
    if postgres_enabled():
        initialize_chat_store()

        with get_postgres_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("delete from chat_sessions where id = %s", (session_id,))
            conn.commit()
        return

    with chat_sessions_lock:
        chat_sessions.pop(session_id, None)
        chat_messages.pop(session_id, None)


def is_follow_up_query(query: str) -> bool:
    normalized = " ".join(query.lower().strip().split())

    if not normalized:
        return False

    words = normalized.split()
    pronoun_reference = bool(
        re.search(
            r"\b(it|its|that|this|these|those|they|them|their|same|above|previous|earlier)\b",
            normalized,
        )
    )
    follow_up_phrase = normalized.startswith(("and ", "also ", "then ")) or any(
        phrase in normalized
        for phrase in (
            "tell me more",
            "explain more",
            "more detail",
            "go deeper",
            "what else",
            "what are the benefits",
            "what benefits",
            "tools used",
            "which tools",
            "how does it",
            "how did it",
        )
    )

    return len(words) <= 14 and (pronoun_reference or follow_up_phrase)


def last_user_query(chat_history: list[dict[str, str]]) -> str:
    for item in reversed(chat_history):
        if item.get("role") == "user" and item.get("content"):
            return str(item.get("content", "")).strip()

    return ""


def last_assistant_source_hint(chat_history: list[dict[str, Any]]) -> str:
    for item in reversed(chat_history):
        if item.get("role") != "assistant":
            continue

        metadata = item.get("metadata") or {}
        source_hints = metadata.get("source_hints") or []
        hint_parts = []

        for source in source_hints[:3]:
            if not isinstance(source, dict):
                continue

            values = [
                source.get("customer_name", ""),
                source.get("usecase_name", ""),
                source.get("ppt_name", ""),
            ]
            label = " | ".join(str(value).strip() for value in values if str(value).strip())

            if label and label not in hint_parts:
                hint_parts.append(label)

        if hint_parts:
            return "; ".join(hint_parts)

    return ""


def build_contextual_query(query: str, chat_history: list[dict[str, str]]) -> str:
    max_messages = max(0, MAX_CHAT_HISTORY_TURNS * 2)

    if not chat_history or max_messages == 0 or not is_follow_up_query(query):
        return query

    recent_history = chat_history[-max_messages:]
    previous_query = last_user_query(recent_history)
    source_hint = last_assistant_source_hint(recent_history)

    if not previous_query:
        return query

    source_context = f"\nPrevious retrieved source/project hints: {source_hint}" if source_hint else ""

    return (
        "Resolve this follow-up using the immediately previous user question and retrieved source hints.\n"
        f"Previous user question: {previous_query}\n"
        f"{source_context}\n"
        f"Current follow-up question: {query}"
    )
