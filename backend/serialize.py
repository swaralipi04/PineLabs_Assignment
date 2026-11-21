"""Convert pipeline objects (dataclasses, DataFrames, Plotly figures) into JSON-safe dicts for the API."""
from __future__ import annotations

import json
import math
from datetime import date, datetime

import numpy as np
import pandas as pd

from core.orchestrator import AnalysisResult, Step
from core.sql_guard import pretty


def _clean(v):
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(v, datetime):  # includes pd.Timestamp
        return v.date().isoformat() if (v.hour, v.minute, v.second) == (0, 0, 0) else v.isoformat()
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def table(df: pd.DataFrame | None, limit: int = 500) -> dict | None:
    if df is None:
        return None
    d = df.head(limit)
    return {
        "columns": [str(c) for c in d.columns],
        "rows": [[_clean(v) for v in row] for row in d.itertuples(index=False, name=None)],
        "total_rows": int(len(df)),
    }


def figure(fig) -> dict | None:
    return None if fig is None else json.loads(fig.to_json())


def step(s: Step) -> dict:
    return {"agent": s.agent, "status": s.status, "detail": s.detail, "ms": round(s.ms, 1)}


def result(res: AnalysisResult) -> dict:
    it = res.intent
    out = {
        "question": res.question,
        "original_question": res.original_question or res.question,
        "context": res.context,
        "role": {"key": res.role.key, "label": res.role.label},
        "status": res.status,
        "message": res.message,
        "intent": it.to_dict() if it else None,
        "trace": [step(s) for s in res.trace],
        "llm": {"calls": res.llm_stats.calls, "prompt_tokens": res.llm_stats.prompt_tokens,
                "completion_tokens": res.llm_stats.completion_tokens, "models": res.llm_stats.models,
                "errors": res.llm_stats.errors[-5:]},
    }
    if res.status != "answered":
        return out
    d = res.drivers
    out.update({
        "sql": pretty(res.sql) if res.sql else "",
        "sql_explanation": res.sql_explanation,
        "sql_source": res.sql_source,
        "sql_attempts": [{**a, "sql": pretty(a["sql"])} for a in res.sql_attempts],
        "result": table(res.result),
        "charts": [{"name": n, "figure": figure(f)} for n, f in res.charts if f is not None],
        "narrative": res.narrative,
        "narrative_source": res.narrative_source,
        "fact_checks": res.fact_checks,
        "caveats": res.caveats,
        "dq_issues": [i.to_dict() for i in res.dq_issues],
        "confidence": res.confidence,
        "citations": res.citations,
        "evidence": [{"title": e.title, "sql": pretty(e.sql), "table": table(e.df), "elapsed_ms": round(e.elapsed_ms, 1), "cached": e.cached}
                     for e in res.evidence],
        "critic_reviews": [{**r, "sql": pretty(r["sql"])} for r in res.critic_reviews],
        "impact": res.impact,
        "hypothesis_tests": [{"hypothesis": t["hypothesis"], "test_plan": t.get("test_plan", ""), "support_if": t.get("support_if", ""),
                              "sql": pretty(t["sql"]) if t.get("sql") else "", "table": table(t.get("df"), 50), "error": t.get("error"),
                              "verdict": t["verdict"], "reasoning": t.get("reasoning", ""), "source": t.get("source", "")}
                             for t in res.hypothesis_tests],
        "drivers": None if d is None else {
            "classification": d.classification,
            "delta_pts": round(d.delta * 100, 2),
            "z": round(d.z, 2),
            "base_window": d.base.to_dict(), "focus_window": d.focus.to_dict(),
            "primary_dimension": d.primary.label if d.primary else None,
            "drill_filter": d.drill_filter,
            "decomposition": table(d.primary.table[["segment", "base_n", "focus_n", "base_share", "focus_share", "base_rate",
                                                    "focus_rate", "rate_effect", "mix_effect", "contribution"]]) if d.primary else None,
        },
    })
    return out
