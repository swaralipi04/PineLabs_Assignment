"""The LLM-backed agents. Each has a deterministic fallback so the system works offline / when rate limited.

  IntentAgent    - what is being asked: intent, metric, time windows, filters, ambiguity
  SQLAgent       - text-to-SQL grounded on the semantic layer, with validator feedback for repair
  NarratorAgent  - business answer split into facts / hypotheses / recommendations
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date

from typing import Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import ANCHOR_DATE, DATA_END, DATA_START
from core import llm
from core import catalog
from core.rbac import Role
from core.semantic import BUSINESS_RULES, DIMENSIONS, METRICS, metrics_prompt, schema_prompt
from core.timeparse import Window, parse_windows

# --------------------------------------------------------------------------- vocabulary for fallbacks
# filter vocabulary (regions, banks, methods, categories, channels) is read from the database: see core/catalog.py

METRIC_PATTERNS = [
    ("success_rate_all_attempts", r"success rate.*(all attempts|including pending)"),
    ("success_rate", r"success[- ]?rate|\bsr\b|approval rate|conversion"),
    ("failure_rate", r"failure[- ]?rate|decline rate|failures?"),
    ("gmv", r"\bgmv\b|revenue|value processed|payment value|amount processed|tpv"),
    ("avg_ticket", r"average ticket|avg ticket|average (transaction|order) (value|size)|aov"),
    ("avg_latency", r"latency|response time"),
    ("successful_txns", r"successful (transactions|txns|payments)"),
    ("attempts", r"\bvolume\b|how many (transactions|payments|attempts)|number of (transactions|payments|attempts)|transaction count|attempts"),
]
WHY_WORDS = re.compile(r"\b(why|reason|cause|caus\w+|driv\w+|explain|what happened|root cause|contribut\w+|behind)\b", re.I)
CHANGE_WORDS = re.compile(r"\b(drop\w*|declin\w*|fell|fall\w*|decreas\w*|dip\w*|down|worse\w*|increas\w*|rose|ris\w+|improv\w*|chang\w*|spike\w*|jump\w*)\b", re.I)
VAGUE_METRIC = re.compile(r"\b(how (did|are|is|were) (we|things|it|business)|performance|perform\w*|doing|how's it going)\b", re.I)
ALL_TIME = re.compile(r"\b(all[- ]time|overall|so far|entire|whole period|all data|to date)\b", re.I)
SETTLEMENT_WORDS = re.compile(r"\b(settle\w*|payouts?|paid out|pay-?outs?|net amount|fees?|mdr)\b", re.I)
SETTLEMENT_RULE = ("Payouts = settlements.net_amount_inr (gross settled amount minus MDR fees), dated by the business date "
                   "settlements.txn_date. Settlements is a separate source system from transactions.")


@dataclass
class Intent:
    question: str
    intent: str = "query"              # query | explain_change | compare
    metric: str | None = None
    windows: list[Window] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    filters: dict[str, str] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    clarification: dict | None = None  # {"question": str, "options": [str]}
    source: str = "rules"
    subject: str = "transactions"      # transactions | settlements (which source answers the question)

    def to_dict(self):
        return {"intent": self.intent, "metric": self.metric, "subject": self.subject, "windows": [w.to_dict() for w in self.windows],
                "group_by": self.group_by, "filters": self.filters, "assumptions": self.assumptions,
                "clarification": self.clarification, "source": self.source}


# =========================================================================== Intent agent
INTENT_SYSTEM = """You are the intent-parsing agent of a payments analytics assistant.
Return ONLY JSON with keys:
  intent: "explain_change" (why did a metric move / what drove it) | "compare" (A vs B) | "query" (fetch a number, list, trend)
  metric: one of {metrics} or null if the question does not say which metric
  group_by: list from {dims} plus "day" or "week" for trends (may be empty)
  filters: object, e.g. {{"payment_method":"<method>","bank":"<bank name>","region":"<region>","merchant_category":"<category>","channel":"<channel>"}}
  time_phrase_start / time_phrase_end: ISO dates ONLY if the question names a period that you can resolve; else null
Today is {today}. Data covers {start} to {end}. Do not invent filters that are not in the question."""


class IntentOut(BaseModel):
    intent: Literal["query", "explain_change", "compare"] = "query"
    metric: Optional[str] = None
    group_by: list[str] = Field(default_factory=list)
    filters: dict[str, str] = Field(default_factory=dict)
    time_phrase_start: Optional[str] = None
    time_phrase_end: Optional[str] = None


INTENT_PROMPT = ChatPromptTemplate.from_messages([("system", INTENT_SYSTEM), ("human", "Question: {question}")])


def _rule_metric(q: str) -> str | None:
    for key, pat in METRIC_PATTERNS:
        if re.search(pat, q, re.I):
            return key
    return None


def _rule_filters(q: str) -> dict[str, str]:
    ql, f = q.lower(), {}
    for w, v in catalog.method_words().items():
        if re.search(rf"\b{w}\b", ql):
            f["payment_method"] = v; break
    for b in catalog.bank_names():
        if re.search(rf"\b{b.split()[0].lower()}\b", ql):
            f["bank"] = b; break
    for r in catalog.regions():
        if re.search(rf"\b{r.lower()}\b", ql):
            f["region"] = r; break
    for c in catalog.categories():
        if re.search(rf"\b{re.escape(c.lower())}\b", ql):
            f["merchant_category"] = c; break
    for w, v in catalog.channel_words().items():
        if re.search(rf"\b(on|via|through) (the )?{w}\b|\b{w} channel\b", ql):
            f["channel"] = v; break
    return f


def _rule_group_by(q: str) -> list[str]:
    ql, g = q.lower(), []
    pats = {"payment_method": r"(by|per|across|each) (payment )?method|method[- ]wise", "bank": r"(by|per|across|each) (issuer )?bank|bank[- ]wise",
            "channel": r"(by|per|across) channel", "region": r"(by|per|across|each|for all|all) regions?\b|region[- ]wise",
            "merchant_category": r"(by|per|across) (merchant )?categor", "merchant_tier": r"(by|per|across) (merchant )?tier",
            "card_network": r"(by|per|across) (card )?network", "failure_category": r"(by|per) failure (category|type)"}
    for k, p in pats.items():
        if re.search(p, ql):
            g.append(k)
    if re.search(r"\b(daily|by day|per day|day[- ]wise|trend|over time|each day)\b", ql):
        g.append("day")
    elif re.search(r"\b(weekly|by week|per week|week[- ]on[- ]week|wow)\b", ql):
        g.append("week")
    return g


def understand(question: str, role: Role, stats: llm.LLMStats | None = None) -> Intent:
    it = Intent(question=question)
    it.windows = parse_windows(question)
    it.metric = _rule_metric(question)
    it.filters = _rule_filters(question)
    it.group_by = _rule_group_by(question)
    if WHY_WORDS.search(question) and (CHANGE_WORDS.search(question) or it.metric):
        it.intent = "explain_change"
    elif re.search(r"\b(vs\.?|versus|compared? (to|with)|against)\b", question, re.I) or len(it.windows) >= 2:
        it.intent = "compare"

    # LLM refinement (fast model): fills gaps the rules could not resolve
    if llm.available():
        out = llm.run_chain(INTENT_PROMPT, {"metrics": list(METRICS), "dims": list(DIMENSIONS), "today": ANCHOR_DATE,
                                            "start": DATA_START, "end": DATA_END, "question": question},
                            IntentOut, tier="fast", stats=stats, max_tokens=800)
        data = out.model_dump() if out else None
        if data:
            it.source = "rules+llm"
            if data.get("intent") in ("query", "explain_change", "compare") and it.intent == "query":
                it.intent = data["intent"]
            if not it.metric and data.get("metric") in METRICS and not VAGUE_METRIC.search(question):
                it.metric = data["metric"]
            for g in data.get("group_by") or []:
                if (g in DIMENSIONS or g in ("day", "week")) and g not in it.group_by:
                    it.group_by.append(g)
            for k, v in (data.get("filters") or {}).items():
                if k in DIMENSIONS and k not in it.filters and isinstance(v, str) and v.lower() in question.lower():
                    it.filters[k] = v
            if not it.windows:
                try:
                    s, e = date.fromisoformat(data["time_phrase_start"]), date.fromisoformat(data["time_phrase_end"])
                    if s <= e:
                        it.windows = [Window(s, e, f"{s} to {e}")]
                        it.assumptions.append(f"Time period interpreted by the intent agent as {s} to {e}.")
                except (KeyError, TypeError, ValueError):
                    pass

    # ---- settlement questions are answered from the settlements source, not transaction metrics
    if SETTLEMENT_WORDS.search(question) and _rule_metric(question) in (None, "gmv") and it.intent != "explain_change":
        it.subject, it.metric = "settlements", None
        it.assumptions.append(SETTLEMENT_RULE)

    # ---- ambiguity handling (deterministic, so behaviour is predictable)
    if it.metric is None and it.subject == "transactions" and VAGUE_METRIC.search(question):
        it.clarification = {
            "question": "Which metric do you mean by performance?",
            "options": ["Success rate", "Transaction volume (attempts)", "GMV (successful value)", "Failure rate"],
        }
    elif not it.windows and not ALL_TIME.search(question):
        it.clarification = {
            "question": "Which time period should I use?",
            "options": ["Last week", "Last month", "Last 4 weeks", f"All available data ({DATA_START} to {DATA_END})"],
        }
    if not it.windows and ALL_TIME.search(question):
        it.windows = [Window(DATA_START, DATA_END, "all available data")]
    if it.intent == "explain_change" and it.metric is None:
        it.metric = "success_rate"
        it.assumptions.append("No metric named; assumed success rate.")
    if it.metric == "success_rate":
        it.assumptions.append("Success rate = SUCCESS / (SUCCESS + FAILED); PENDING attempts excluded from the denominator. "
                              "Ask for 'success rate including pending' to use all attempts.")
    if it.intent == "explain_change" and len(it.windows) == 1:
        it.assumptions.append(f"Compared against the preceding period of equal length ({it.windows[0].previous().start} to {it.windows[0].previous().end}).")
    if role.region_restricted:
        it.assumptions.append(f"Results limited to your access scope: {', '.join(role.regions)} region.")
    return it


# =========================================================================== SQL agent
SQL_SYSTEM = """You are a senior analytics engineer writing DuckDB SQL for a payments warehouse.
Rules:
- Return ONLY JSON: {{"sql": "...", "explanation": "one sentence"}}
- One read-only SELECT (CTEs allowed). Never modify data. Use only the tables/columns listed.
- Alias transactions as t. Use the metric SQL exactly as defined when computing a metric.
- Use the exact date window(s) given; filter with t.txn_date BETWEEN DATE 'YYYY-MM-DD' AND DATE 'YYYY-MM-DD'.
- Aggregate - never return raw transaction rows unless explicitly asked (then LIMIT 100).
- Give columns readable snake_case aliases. Order trends by date; order rankings by the metric.
{rules}

Schema:
{schema}

Metrics:
{metrics}

Example:
Q: daily success rate for UPI from 2026-10-01 to 2026-10-07
{{"sql": "SELECT t.txn_date, SUM(CASE WHEN t.status='SUCCESS' THEN 1 ELSE 0 END)::DOUBLE / NULLIF(SUM(CASE WHEN t.status IN ('SUCCESS','FAILED') THEN 1 ELSE 0 END),0) AS success_rate, COUNT(*) AS attempts FROM transactions t WHERE t.payment_method_id = 'UPI' AND t.txn_date BETWEEN DATE '2026-10-01' AND DATE '2026-10-07' GROUP BY 1 ORDER BY 1", "explanation": "Daily UPI success rate."}}
"""


class SQLOut(BaseModel):
    sql: str
    explanation: str = ""


SQL_PROMPT = ChatPromptTemplate.from_messages([("system", SQL_SYSTEM), ("human", "{task}")])


def _sql_task(it: Intent) -> str:
    lines = [f"Question: {it.question}"]
    if it.windows:
        lines.append("Resolved time windows (use these exact dates): " + "; ".join(f"{w.label}: {w.start} to {w.end}" for w in it.windows))
    if it.intent == "explain_change" and it.windows:
        f, b = it.windows[0], (it.windows[1] if len(it.windows) > 1 else it.windows[0].previous())
        lines.append(f"Task: return the DAILY {it.metric} (plus attempts) from {min(b.start, f.start)} to {max(b.end, f.end)} "
                     f"so the change between {b.start}..{b.end} and {f.start}..{f.end} can be charted. Driver analysis is done separately.")
    if it.subject == "settlements":
        lines.append("Source: answer from the SETTLEMENTS table (alias s), not transactions. " + SETTLEMENT_RULE +
                     " Filter time on s.txn_date. For region or category, JOIN merchants m ON m.merchant_id = s.merchant_id. "
                     "Return net_amount_inr totals (plus gross_amount_inr, fees_inr and txn_count where useful).")
    if it.metric:
        lines.append(f"Metric: {it.metric}")
    if it.filters:
        lines.append(f"Filters requested: {json.dumps(it.filters)}")
    if it.group_by:
        lines.append(f"Group by: {it.group_by}")
    return "\n".join(lines)


def generate_sql(it: Intent, role: Role, feedback: str | None = None, previous_sql: str | None = None,
                 stats: llm.LLMStats | None = None) -> tuple[str, str, str]:
    """Return (sql, explanation, source)."""
    if llm.available():
        task = _sql_task(it)
        if feedback:
            task += f"\n\nYour previous SQL was rejected.\nPrevious SQL:\n{previous_sql}\nValidator/database error: {feedback}\nFix it."
        out = llm.run_chain(SQL_PROMPT, {"rules": "\n".join(f"- {r}" for r in BUSINESS_RULES), "schema": schema_prompt(role),
                                         "metrics": metrics_prompt(), "task": task},
                            SQLOut, tier="strong", stats=stats, max_tokens=2000)
        if out and out.sql.strip():
            return out.sql.strip(), out.explanation, "llm"
    sql, expl = template_sql(it)
    return sql, expl, "template"


# ------------------------------------------------------------------ deterministic SQL builder (fallback + eval reference)
def filter_clauses(filters: dict[str, str]) -> tuple[list[str], list[str]]:
    where, joins = [], []
    for k, v in filters.items():
        v = str(v).replace("'", "''")
        if k == "payment_method":
            code = catalog.method_words().get(v.lower(), v.upper())
            where.append(f"t.payment_method_id = '{code}'")
        elif k == "bank":
            joins.append(DIMENSIONS["bank"][2]); where.append(f"b.bank_name = '{v}'")
        elif k == "region":
            where.append(f"t.region = '{v}'")
        elif k == "channel":
            where.append(f"t.channel = '{v.lower()}'")
        elif k in ("merchant_category", "merchant_tier"):
            joins.append(DIMENSIONS[k][2]); where.append(f"{DIMENSIONS[k][1]} = '{v}'")
    return where, list(dict.fromkeys(joins))


def _settlement_template(it: Intent) -> tuple[str, str]:
    w = (it.windows or [Window(DATA_START, DATA_END, "all data")])[0]
    dims = {"region": "m.region", "merchant_category": "m.category", "merchant_tier": "m.tier", "day": "s.txn_date"}
    group = [g for g in it.group_by if g in dims]
    where = [f"s.txn_date BETWEEN DATE '{w.start}' AND DATE '{w.end}'"]
    for k in ("region", "merchant_category"):
        if k in it.filters:
            where.append(f"{dims[k]} = '{str(it.filters[k]).replace(chr(39), chr(39) * 2)}'")
    sel = [f"{dims[g]} AS {g}" for g in group]
    sel += ["SUM(s.net_amount_inr) AS net_payout_inr", "SUM(s.gross_amount_inr) AS gross_settled_inr",
            "SUM(s.fees_inr) AS fees_inr", "SUM(s.txn_count) AS settled_txns"]
    sql = f"SELECT {', '.join(sel)}\nFROM settlements s LEFT JOIN merchants m ON m.merchant_id = s.merchant_id\nWHERE {' AND '.join(where)}"
    if group:
        sql += f"\nGROUP BY {', '.join(dims[g] for g in group)}\nORDER BY " + ("1" if group == ["day"] else "net_payout_inr DESC")
    return sql, "Net payouts from the settlements system (gross settled minus MDR fees)."


def template_sql(it: Intent) -> tuple[str, str]:
    if it.subject == "settlements":
        return _settlement_template(it)
    metric = METRICS[it.metric or "success_rate"]
    windows = it.windows or [Window(DATA_START, DATA_END, "all data")]
    group = list(it.group_by)
    if it.intent == "explain_change":
        f = windows[0]; b = windows[1] if len(windows) > 1 else f.previous()
        windows = [Window(min(b.start, f.start), max(b.end, f.end), "comparison span")]
        group = ["day"]
    where, joins = filter_clauses(it.filters)
    sel, gb = [], []
    if it.intent == "compare" and len(windows) >= 2:
        case = " ".join(f"WHEN t.txn_date BETWEEN DATE '{w.start}' AND DATE '{w.end}' THEN '{w.label}'" for w in windows)
        sel.append(f"CASE {case} END AS period"); gb.append("period")
        where.append("(" + " OR ".join(f"t.txn_date BETWEEN DATE '{w.start}' AND DATE '{w.end}'" for w in windows) + ")")
    else:
        w = windows[0]
        where.append(f"t.txn_date BETWEEN DATE '{w.start}' AND DATE '{w.end}'")
    for g in group:
        if g == "day":
            sel.append("t.txn_date"); gb.append("t.txn_date")
        elif g == "week":
            sel.append("DATE_TRUNC('week', t.txn_date) AS week_start"); gb.append("week_start")
        elif g in DIMENSIONS:
            label, expr, join = DIMENSIONS[g]
            if join:
                joins.append(join)
            sel.append(f"{expr} AS {g}"); gb.append(g)
    sel.append(f"{metric.sql} AS {metric.key}")
    if metric.key not in ("attempts",):
        sel.append("COUNT(*) AS attempts")
    sql = f"SELECT {', '.join(sel)}\nFROM transactions t {' '.join(dict.fromkeys(joins))}\nWHERE {' AND '.join(where)}"
    if gb:
        sql += f"\nGROUP BY {', '.join(gb)}\nORDER BY " + ("1" if group and group[0] in ("day", "week") or gb == ["period"] else f"{metric.key} {'ASC' if metric.higher_is_better else 'DESC'}")
    return sql, f"{metric.label} computed from the semantic-layer definition."


# =========================================================================== Narrator agent
NARRATOR_SYSTEM = """You are a payments business analyst writing for a business leader.
You receive a question and PRE-COMPUTED evidence (query results, driver decomposition, data-quality checks).
Return ONLY JSON:
{{
 "headline": "one sentence direct answer with the key number",
 "facts": ["observed facts - every number MUST appear in the evidence; cite the window"],
 "hypotheses": ["plausible explanations that the data suggests but does not prove - phrase as hypotheses"],
 "recommendations": ["concrete next actions for the business / ops team"]
}}
Rules: never invent numbers; rates are given in percent already (e.g. 92.73 means 92.73%); 'pts' means percentage points.
Write for a business reader: never mention JSON keys, field names or code identifiers (e.g. do not write "drivers.change_pts").
All money is Indian Rupees - write ₹, never $. For amounts, copy the pre-formatted *_display strings from the evidence
(e.g. "₹19.66 lakh") exactly; never convert units yourself (1 lakh = 100,000; 1 crore = 10,000,000).
If classification is mix_shift, say clearly that segment performance did NOT deteriorate and the drop comes from traffic mix.
Keep each item to one or two sentences. 3-5 facts, 1-3 hypotheses, 2-4 recommendations."""


class NarrativeOut(BaseModel):
    headline: str
    facts: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


NARRATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", NARRATOR_SYSTEM),
    ("human", "Question: {question}\n\nEvidence (JSON):\n{evidence}"),
])


def narrate(question: str, evidence: dict, stats: llm.LLMStats | None = None) -> tuple[dict, str]:
    if llm.available():
        out = llm.run_chain(NARRATOR_PROMPT, {"question": question, "evidence": json.dumps(evidence, default=str)[:12000]},
                            NarrativeOut, tier="strong", stats=stats, max_tokens=2500, temperature=0.1)
        if out and out.headline.strip():
            return out.model_dump(), "llm"
    return template_narrative(evidence), "template"


def template_narrative(ev: dict) -> dict:
    d = ev.get("drivers")
    facts, hyps, recs = [], [], []
    if d:
        fw, bw = d["focus_window"], d["base_window"]
        direction = "fell" if d["change_pts"] < 0 else "rose"
        headline = (f"Success rate {direction} {abs(d['change_pts']):.2f} pts, from {d['base_success_rate']:.2f}% ({bw['start']}..{bw['end']}) "
                    f"to {d['focus_success_rate']:.2f}% ({fw['start']}..{fw['end']}).")
        facts.append(headline)
        if d["classification"] == "mix_shift":
            top = d["primary_dimension"][0]
            headline += f" This is a traffic-mix effect: {top['segment']} share moved {top['base_share']:.1f}% -> {top['focus_share']:.1f}%."
            facts.append(f"{d['primary_dimension_name']} mix effect explains {next(x['mix_effect_pts'] for x in d['dimension_summary'] if x['dimension'] == d['primary_dimension_name']):.2f} pts of the change; within-segment rates were broadly stable.")
            for r in d["primary_dimension"][:3]:
                facts.append(f"{r['segment']}: success rate {r['base_rate']:.2f}% -> {r['focus_rate']:.2f}%, share {r['base_share']:.1f}% -> {r['focus_share']:.1f}%.")
            hyps.append(f"A campaign or routing change pushed more traffic to {top['segment']}, which structurally converts lower.")
            recs += [f"Confirm with marketing/product whether a promotion drove {top['segment']} traffic in the window.",
                     "Report segment-level success rates alongside the blended rate to avoid false alarms."]
        elif d["classification"] == "rate_driven":
            top = d["primary_dimension"][0]
            facts.append(f"{top['segment']} ({d['primary_dimension_name']}) success rate went {top['base_rate']:.2f}% -> {top['focus_rate']:.2f}%, "
                         f"contributing {top['rate_effect_pts'] + top['mix_effect_pts']:.2f} pts.")
            for name, rows in (d.get("drilldown") or {}).items():
                if rows:
                    r = rows[0]
                    facts.append(f"Within {d['drilldown_within']}, {name} '{r['segment']}' went {r['base_rate']:.2f}% -> {r['focus_rate']:.2f}%.")
                    break
            fc = d.get("failure_codes_in_focus_segment") or []
            if fc:
                facts.append(f"Top rising failure reason: {fc[0]['failure_code']} ({int(fc[0]['base_failures'])} -> {int(fc[0]['focus_failures'])}).")
                hyps.append(f"An issuer-side technical problem ({fc[0]['failure_code']}) is the likely cause.")
            recs += ["Raise an incident with the affected bank/partner and check their status page / SLA reports.",
                     "Consider smart-routing or retry on an alternate path for the affected segment until resolved."]
        else:
            headline = f"No material change: success rate moved {d['change_pts']:+.2f} pts, which is within normal variation (z={d['z_score']})."
            facts = [headline]
            recs.append("No action needed; continue monitoring.")
    else:
        s = ev.get("result_summary") or {}
        headline = s.get("headline", "Query executed - see the table for results.")
        facts = s.get("facts") or [headline]
        recs.append("Drill into the largest or most unusual segment in the table for more detail.")
        if s.get("spread_pts", 0) > 3:
            hyps.append("The gap between best and worst segments may reflect structural differences (e.g. payment method mix) rather than incidents.")
    return {"headline": headline, "facts": facts, "hypotheses": hyps, "recommendations": recs}


def summarize_result(df, metric_key: str | None) -> dict:
    """Deterministic plain-English summary of a result table (used by the offline narrator and as evidence)."""
    import pandas as pd
    from core.semantic import fmt_value
    if df is None or df.empty:
        return {"headline": "The query returned no rows for this scope and window.", "facts": []}
    mcol = metric_key if metric_key in df.columns else next((c for c in df.columns if c in METRICS), None)
    fmt = METRICS[mcol].fmt if mcol else None
    label = METRICS[mcol].label if mcol else "value"
    if not mcol:  # e.g. settlement payouts: rank by the main amount column
        nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        mcol = next((c for c in nums if "net" in c), None) or next((c for c in nums if "amount" in c or "inr" in c), None) or (nums[0] if nums else None)
        if mcol:
            fmt = "inr" if ("inr" in mcol or "amount" in mcol) else "int"
            label = mcol.replace("_inr", "").replace("_", " ").capitalize()
    f = lambda v: fmt_value(v, fmt) if fmt else str(v)
    if len(df) == 1:
        row = df.iloc[0]
        parts = [f"{c.replace('_', ' ')}: {fmt_value(row[c], METRICS[c].fmt) if c in METRICS else (f'{row[c]:,.0f}' if isinstance(row[c], (int, float)) else row[c])}" for c in df.columns]
        return {"headline": "; ".join(parts) + ".", "facts": parts}
    if not mcol:
        return {"headline": f"{len(df):,} rows returned - see table.", "facts": []}
    dcol = next((c for c in df.columns if "date" in c or c in ("week_start", "period")), None)
    if dcol:
        d = df.sort_values(dcol)
        lo, hi = d.loc[d[mcol].idxmin()], d.loc[d[mcol].idxmax()]
        first, last = d.iloc[0], d.iloc[-1]
        return {"headline": f"{label} ranged from {f(lo[mcol])} ({pd.Timestamp(lo[dcol]).date() if dcol != 'period' else lo[dcol]}) to {f(hi[mcol])} "
                            f"({pd.Timestamp(hi[dcol]).date() if dcol != 'period' else hi[dcol]}) over {len(d)} periods.",
                "facts": [f"First period {f(first[mcol])}, last period {f(last[mcol])}.",
                          f"Lowest {f(lo[mcol])}, highest {f(hi[mcol])}."],
                "spread_pts": float(hi[mcol] - lo[mcol]) * (100 if fmt == "pct" else 0)}
    cat = next((c for c in df.columns if c != mcol and not pd.api.types.is_numeric_dtype(df[c])), None)
    if cat:
        d = df.dropna(subset=[mcol]).sort_values(mcol, ascending=False)
        best, worst = d.iloc[0], d.iloc[-1]
        return {"headline": f"{label} is highest for {best[cat]} ({f(best[mcol])}) and lowest for {worst[cat]} ({f(worst[mcol])}) across {len(d)} {cat.replace('_', ' ')} values.",
                "facts": [f"{r[cat]}: {f(r[mcol])}" + (f" on {int(r['attempts']):,} attempts" if "attempts" in d.columns and mcol != "attempts" else "") for _, r in d.head(6).iterrows()],
                "spread_pts": float(best[mcol] - worst[mcol]) * (100 if fmt == "pct" else 0)}
    return {"headline": f"{len(df):,} rows returned - see table.", "facts": []}
