"""Conversation memory + Context Resolver agent.

Keeps the last few turns per (session, role). Follow-ups such as "what about netbanking?" or
"now break that down by channel" are rewritten into standalone questions before the rest of the
graph runs, so the access guard, intent parser and SQL agent always see a complete question.
"""
from __future__ import annotations

import re
import threading
from collections import deque

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from core import llm
from core.semantic import METRICS

MAX_TURNS = 6
_store: dict[str, deque] = {}
_lock = threading.Lock()

FOLLOWUP = re.compile(
    r"^\s*(and|but|also|now|then|ok|okay|so|what about|how about|same|instead|only|just|break|drill|split|compare|vs\.?|versus|"
    r"by |per |for |what if|show me|and for|now for|how does that)\b"
    r"|\b(it|that|this|those|these|them|same|there|previous|above|earlier|instead)\b", re.I)


def _key(session_id: str, role_key: str) -> str:
    return f"{session_id}:{role_key}"


def history(session_id: str | None, role_key: str) -> list[dict]:
    if not session_id:
        return []
    with _lock:
        return list(_store.get(_key(session_id, role_key), []))


def remember(session_id: str | None, role_key: str, res) -> None:
    if not session_id or res.status == "denied" or res.intent is None:
        return
    it = res.intent
    turn = {
        "question": res.original_question or res.question,
        "standalone": res.question,
        "status": res.status,
        "intent": it.intent,
        "metric": it.metric,
        "windows": [w.to_dict() for w in it.windows],
        "filters": it.filters,
        "group_by": it.group_by,
        "headline": (res.narrative or {}).get("headline") if res.status == "answered" else res.message,
    }
    with _lock:
        _store.setdefault(_key(session_id, role_key), deque(maxlen=MAX_TURNS)).append(turn)


def forget(session_id: str) -> None:
    with _lock:
        for k in [k for k in _store if k.startswith(f"{session_id}:")]:
            del _store[k]


# ------------------------------------------------------------------ resolver
RESOLVER_SYSTEM = """You resolve follow-up questions in a payments analytics chat.
Given the recent conversation and a new message, rewrite the new message as a complete standalone question.
- Carry over the metric, time period, filters and question type ("why ..." / trend / breakdown) from the previous turn unless the new message changes them.
- If the new message answers a clarification (e.g. just "success rate" or "last week"), merge it into the clarified question.
- Write time periods as explicit ISO dates from the history (e.g. "from 2026-11-09 to 2026-11-15") so nothing is lost.
- If the new message is already standalone, return it unchanged.
Return ONLY JSON: {{"standalone_question": "...", "is_followup": true/false, "carried_over": ["metric", "time period", ...]}}"""

RESOLVER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RESOLVER_SYSTEM),
    ("human", "Conversation so far (oldest first):\n{history}\n\nNew message: {question}"),
])


class ResolverOut(BaseModel):
    standalone_question: str
    is_followup: bool = False
    carried_over: list[str] = Field(default_factory=list)


def _history_text(turns: list[dict]) -> str:
    out = []
    for t in turns[-3:]:
        w = "; ".join(f"{x['start']} to {x['end']}" for x in t["windows"]) or "none"
        out.append(f"- Q: {t['standalone']}\n  (type={t['intent']}, metric={t['metric']}, period={w}, filters={t['filters'] or 'none'}, "
                   f"group_by={t['group_by'] or 'none'}, status={t['status']})\n  A: {t['headline']}")
    return "\n".join(out)


def looks_like_followup(question: str) -> bool:
    return bool(FOLLOWUP.search(question)) or len(question.split()) <= 4


def resolve(question: str, turns: list[dict], stats: llm.LLMStats | None = None) -> dict:
    """Return {"question", "used_context", "carried_over", "source"}."""
    if not turns or not looks_like_followup(question):
        return {"question": question, "used_context": False, "carried_over": [], "source": "none"}
    if llm.available():
        out = llm.run_chain(RESOLVER_PROMPT, {"history": _history_text(turns), "question": question}, ResolverOut,
                            tier="fast", stats=stats, max_tokens=800)
        if out and out.standalone_question.strip():
            changed = out.standalone_question.strip().rstrip("?").lower() != question.strip().rstrip("?").lower()
            return {"question": out.standalone_question.strip(), "used_context": changed,
                    "carried_over": out.carried_over if changed else [], "source": "llm"}
    return _rule_resolve(question, turns[-1])


def _rule_resolve(question: str, prev: dict) -> dict:
    """Deterministic fallback: append the previous turn's missing metric / period / filters."""
    from core.agents import _rule_filters, _rule_metric
    from core.timeparse import parse_windows

    parts, carried = [question.rstrip("?")], []
    if not _rule_metric(question) and prev.get("metric") in METRICS:
        parts.append(METRICS[prev["metric"]].label.lower()); carried.append("metric")
    new_filters = _rule_filters(question)
    from core import catalog
    readable = {k.upper(): v.lower() for k, v in catalog.methods().items()}   # id -> display name, from the data
    for k, v in (prev.get("filters") or {}).items():
        if k not in new_filters:
            v = readable.get(str(v).upper(), v) if k == "payment_method" else (f"{v} channel" if k == "channel" else v)
            parts.append(f"for {v}"); carried.append(f"filter {k}")
    if not parse_windows(question) and prev.get("windows"):
        w = prev["windows"][0]
        parts.append(f"from {w['start']} to {w['end']}"); carried.append("time period")
    if prev.get("intent") == "explain_change" and not re.search(r"\bwhy\b", question, re.I) and re.search(r"\bwhat about|how about\b", question, re.I):
        parts[0] = "why did success rate drop - " + parts[0]; carried.append("question type")
    q = " ".join(parts)
    return {"question": q, "used_context": bool(carried), "carried_over": carried, "source": "rules"}
