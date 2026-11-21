"""SQL Critic agent: reviews SQL for *semantic* correctness before it runs.

The SQL guard already guarantees the query is safe; the critic checks it actually answers the question
(right window, filters, metric denominator, grouping). Deterministic checks run first (cheap, reliable),
then an LLM reviewer looks for subtler mistakes. A "revise" verdict sends the SQL back to the SQL agent.
"""
from __future__ import annotations

import json
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from core import llm
from core import catalog
from core.agents import Intent, _sql_task
from core.semantic import metrics_prompt

CRITIC_SYSTEM = """You are a meticulous SQL reviewer for a payments analytics team.
The SQL has ALREADY passed a safety validator. Judge ONLY semantic correctness: does it correctly do the TASK given to the SQL author?
The task is authoritative: e.g. if it asks for a DAILY series over a date span, daily grouping is correct (comparisons and driver
analysis are done by other agents) - do not ask for period-level aggregation or comparisons the task did not request.
Check: (1) metric formula and denominator match the definitions, (2) the exact resolved date window(s) are used,
(3) every requested filter is applied and no unrequested filter is added, (4) grouping matches the question,
(5) joins use the right keys and do not multiply rows before aggregation.
Do NOT flag style, aliases, ordering, LIMIT, or extra helpful columns. Use "revise" only for errors that change the numbers or the meaning.
Return ONLY JSON: {{"verdict": "approve" or "revise", "issues": ["specific problem -> specific fix", ...]}}"""

CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CRITIC_SYSTEM),
    ("human", "TASK given to the SQL author:\n{task}\n\nResolved intent: {intent}\n\nMetric definitions:\n{metrics}\n\nSQL to review:\n{sql}"),
])


class CriticOut(BaseModel):
    verdict: Literal["approve", "revise"] = "approve"
    issues: list[str] = Field(default_factory=list)


def _expected_dates(it: Intent) -> list[str]:
    if not it.windows:
        return []
    if it.intent == "explain_change":
        f = it.windows[0]
        b = it.windows[1] if len(it.windows) > 1 else f.previous()
        return [str(min(b.start, f.start)), str(max(b.end, f.end))]
    return [str(d) for w in it.windows for d in (w.start, w.end)]


def deterministic_checks(sql: str, it: Intent) -> list[str]:
    s, issues = sql.upper(), []
    for d in _expected_dates(it):
        if d not in sql:
            issues.append(f"The resolved date {d} does not appear in the SQL - use the exact window(s) from the intent.")
    f = it.filters
    if "payment_method" in f:
        code = catalog.method_words().get(f["payment_method"].lower(), f["payment_method"].upper())
        if f"'{code}'" not in s and f["payment_method"].upper() not in s:
            issues.append(f"Payment method filter ({code}) is missing.")
    if "bank" in f:
        name = f["bank"]
        bank_id = catalog.bank_id(name)
        if name.split()[0].upper() not in s and (not bank_id or bank_id not in s):
            issues.append(f"Bank filter ({name}) is missing.")
    for k in ("region", "channel", "merchant_category"):
        if k in f and str(f[k]).upper() not in s:
            issues.append(f"{k.replace('_', ' ').title()} filter ({f[k]}) is missing.")
    if it.metric == "success_rate" and "SUCCESS" in s and "'FAILED'" not in s:
        issues.append("Success rate denominator must be SUCCESS + FAILED (exclude PENDING); the SQL does not reference FAILED.")
    return issues


def review(sql: str, it: Intent, source: str, stats: llm.LLMStats | None = None) -> dict:
    """Return {"verdict", "issues", "reviewer"}."""
    if source == "template":
        return {"verdict": "approve", "issues": [], "reviewer": "skipped (template SQL is built from the semantic layer)"}
    issues = deterministic_checks(sql, it)
    if issues:
        return {"verdict": "revise", "issues": issues, "reviewer": "rule checks"}
    if llm.available():
        out = llm.run_chain(CRITIC_PROMPT, {"task": _sql_task(it), "intent": json.dumps(it.to_dict(), default=str),
                                            "metrics": metrics_prompt(), "sql": sql},
                            CriticOut, tier="fast", stats=stats, max_tokens=1200)
        if out:
            return {"verdict": out.verdict, "issues": out.issues if out.verdict == "revise" else [], "reviewer": "rule checks + LLM"}
    return {"verdict": "approve", "issues": [], "reviewer": "rule checks"}
