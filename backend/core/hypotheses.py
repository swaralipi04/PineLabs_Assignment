"""Hypothesis Tester agent: turns the narrator's hypotheses into tested claims.

1. Design  (LLM, strong): for each hypothesis, a falsifiable test = SQL + what result would support it.
2. Execute (code)       : every test SQL goes through the same guard/executor (read-only, role-scoped).
3. Judge   (LLM, fast)  : verdict per test - supported / refuted / inconclusive - citing the numbers.
Fallback (no LLM): built-in tests derived from the driver analysis.
"""
from __future__ import annotations

import json
from typing import Literal

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from core import executor, llm
from core.analysis import DriverReport, merge_joins
from core.rbac import Role
from core.semantic import BUSINESS_RULES, DIMENSIONS, metrics_prompt, schema_prompt
from core.sql_guard import GuardError

MAX_TESTS = 3

DESIGN_SYSTEM = """You design falsifiable data tests for business hypotheses about payment performance.
For each hypothesis write ONE DuckDB SELECT that would confirm or refute it, comparing the BASELINE window with the FOCUS window.
Rules:
- Read-only SELECT, aggregated, at most ~20 result rows. Alias transactions as t. Only use the schema below.
- Return separate columns for baseline and focus (e.g. using SUM(CASE WHEN t.txn_date BETWEEN ... THEN ... END)), or a 'period' column.
- Compare against a control group where possible (e.g. affected segment vs everything else) so the test can actually fail.
- Rates must be fractions with SUCCESS/(SUCCESS+FAILED) denominators. Use the exact dates given.
{rules}

Schema:
{schema}

Metrics:
{metrics}

Return ONLY JSON: {{"tests": [{{"hypothesis": "...", "test_plan": "one sentence", "sql": "...", "support_if": "what result supports the hypothesis"}}]}}"""

DESIGN_PROMPT = ChatPromptTemplate.from_messages([("system", DESIGN_SYSTEM), ("human", "{context}")])

JUDGE_SYSTEM = """You are a strict analyst judging hypothesis tests against query results.
For each test decide: "supported" (result clearly matches support_if), "refuted" (result clearly contradicts it),
or "inconclusive" (query failed, result ambiguous, or effect too small to tell). Cite the specific numbers in your reasoning.
Rates in results are fractions (0.9047 = 90.47%).
Return ONLY JSON: {{"verdicts": [{{"id": 0, "verdict": "supported|refuted|inconclusive", "reasoning": "one or two sentences with numbers"}}]}}"""

JUDGE_PROMPT = ChatPromptTemplate.from_messages([("system", JUDGE_SYSTEM), ("human", "{tests}")])


class TestDesign(BaseModel):
    hypothesis: str
    test_plan: str = ""
    sql: str
    support_if: str = ""


class DesignOut(BaseModel):
    tests: list[TestDesign] = Field(default_factory=list)


class Verdict(BaseModel):
    id: int
    verdict: Literal["supported", "refuted", "inconclusive"] = "inconclusive"
    reasoning: str = ""


class JudgeOut(BaseModel):
    verdicts: list[Verdict] = Field(default_factory=list)


def _context(question: str, hypotheses: list[str], rep: DriverReport, filters: dict) -> str:
    lines = [f"Question: {question}",
             f"BASELINE window: {rep.base.start} to {rep.base.end}", f"FOCUS window: {rep.focus.start} to {rep.focus.end}",
             f"Filters already applied to the question: {json.dumps(filters)}",
             f"Driver analysis: {rep.classification}, change {rep.delta * 100:+.2f} pts"]
    if rep.primary:
        lines.append(f"Primary driver dimension: {rep.primary.label}, top segment: {rep.primary.top_segment}")
    if rep.drill_filter and rep.drilldown:
        lines.append(f"Drill-down within {rep.drill_filter}: top segment by {rep.drilldown[0].label} = {rep.drilldown[0].top_segment}")
    lines.append("Hypotheses to test:")
    lines += [f"{i}. {h}" for i, h in enumerate(hypotheses)]
    return "\n".join(lines)


def _run(sql: str, role: Role) -> tuple[pd.DataFrame | None, str | None]:
    try:
        return executor.run(sql, role).df, None
    except GuardError as e:
        return None, f"{e.kind}: {e}"


def _rows(df: pd.DataFrame | None, n: int = 20) -> list[dict]:
    if df is None:
        return []
    d = df.head(n).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(4)
    return d.astype(object).where(d.notna(), None).to_dict("records")


def run_tests(question: str, hypotheses: list[str], rep: DriverReport, role: Role, filters: dict,
              stats: llm.LLMStats | None = None) -> list[dict]:
    hyps = [h for h in hypotheses if h and h.strip()][:MAX_TESTS]
    if llm.available() and hyps:
        design = llm.run_chain(DESIGN_PROMPT, {"rules": "\n".join(f"- {r}" for r in BUSINESS_RULES), "schema": schema_prompt(role),
                                               "metrics": metrics_prompt(), "context": _context(question, hyps, rep, filters)},
                               DesignOut, tier="strong", stats=stats, max_tokens=3000)
        if design and design.tests:
            tests = []
            for d in design.tests[:MAX_TESTS]:
                df, err = _run(d.sql, role)
                tests.append({"hypothesis": d.hypothesis, "test_plan": d.test_plan, "support_if": d.support_if,
                              "sql": d.sql, "df": df, "error": err, "source": "llm"})
            payload = [{"id": i, "hypothesis": t["hypothesis"], "support_if": t["support_if"],
                        "result_rows": _rows(t["df"]), "query_error": t["error"]} for i, t in enumerate(tests)]
            judged = llm.run_chain(JUDGE_PROMPT, {"tests": json.dumps(payload, default=str)}, JudgeOut, tier="fast",
                                   stats=stats, max_tokens=1500)
            by_id = {v.id: v for v in (judged.verdicts if judged else [])}
            for i, t in enumerate(tests):
                v = by_id.get(i)
                if t["error"]:
                    t["verdict"], t["reasoning"] = "inconclusive", f"Test query could not run ({t['error']})."
                elif v:
                    t["verdict"], t["reasoning"] = v.verdict, v.reasoning
                else:
                    t["verdict"], t["reasoning"] = "inconclusive", "The judge did not return a verdict."
            return tests
    return builtin_tests(rep, role)


# ------------------------------------------------------------------ deterministic fallback tests
def _rate_cols(base, focus, cond="TRUE"):
    def agg(w, succ):
        c = "t.status = 'SUCCESS'" if succ else "t.status IN ('SUCCESS','FAILED')"
        return f"SUM(CASE WHEN t.txn_date BETWEEN DATE '{w.start}' AND DATE '{w.end}' AND {c} AND {cond} THEN 1 ELSE 0 END)"
    return (f"{agg(base, True)}::DOUBLE / NULLIF({agg(base, False)}, 0) AS base_rate, "
            f"{agg(focus, True)}::DOUBLE / NULLIF({agg(focus, False)}, 0) AS focus_rate")


def builtin_tests(rep: DriverReport, role: Role) -> list[dict]:
    tests, b, f = [], rep.base, rep.focus
    lo, hi = min(b.start, f.start), max(b.end, f.end)
    if rep.classification == "rate_driven" and rep.primary:
        label, expr, join = DIMENSIONS[rep.primary.dimension]
        seg = rep.primary.top_segment.replace("'", "''")
        sql = f"""SELECT CASE WHEN {expr} = '{seg}' THEN 'affected: {seg}' ELSE 'everything else' END AS grp, {_rate_cols(b, f)}
FROM transactions t {join} WHERE t.txn_date BETWEEN DATE '{lo}' AND DATE '{hi}' GROUP BY 1"""
        df, err = _run(sql, role)
        verdict, why = "inconclusive", err or ""
        if df is not None and len(df) == 2:
            d = df.set_index("grp")
            seg_chg = (d.loc[f"affected: {rep.primary.top_segment}", "focus_rate"] - d.loc[f"affected: {rep.primary.top_segment}", "base_rate"]) * 100
            rest_chg = (d.loc["everything else", "focus_rate"] - d.loc["everything else", "base_rate"]) * 100
            verdict = "supported" if seg_chg <= -1.5 and abs(rest_chg) < 0.5 else "refuted" if abs(rest_chg) >= abs(seg_chg) / 2 else "inconclusive"
            why = f"{rep.primary.top_segment} changed {seg_chg:+.2f} pts while everything else changed {rest_chg:+.2f} pts."
        tests.append({"hypothesis": f"The decline is isolated to {rep.primary.top_segment} ({label.lower()}).",
                      "test_plan": "Compare the affected segment against all other traffic (control group).",
                      "support_if": "Affected segment drops sharply while the rest stays flat.",
                      "sql": sql, "df": df, "error": err, "verdict": verdict, "reasoning": why, "source": "built-in"})

        if rep.drilldown:
            dd = rep.drilldown[0]
            _, dexpr, djoin = DIMENSIONS[dd.dimension]
            dseg = dd.top_segment.replace("'", "''")
            joins = merge_joins(join, djoin, "LEFT JOIN failure_codes fc ON fc.failure_code = t.failure_code")
            cond = f"{expr} = '{seg}' AND {dexpr} = '{dseg}'"
            sql2 = f"""SELECT CASE WHEN t.txn_date BETWEEN DATE '{f.start}' AND DATE '{f.end}' THEN 'focus' ELSE 'baseline' END AS period,
       AVG(t.response_ms) AS avg_latency_ms,
       SUM(CASE WHEN fc.failure_category = 'TECHNICAL' THEN 1 ELSE 0 END)::DOUBLE / NULLIF(SUM(CASE WHEN t.status = 'FAILED' THEN 1 ELSE 0 END), 0) AS technical_share_of_failures
FROM transactions t {joins}
WHERE t.txn_date BETWEEN DATE '{lo}' AND DATE '{hi}' AND {cond} GROUP BY 1"""
            df2, err2 = _run(sql2, role)
            verdict, why = "inconclusive", err2 or ""
            if df2 is not None and len(df2) == 2:
                d = df2.set_index("period")
                lat = d.loc["focus", "avg_latency_ms"] / d.loc["baseline", "avg_latency_ms"]
                tech_b, tech_f = d.loc["baseline", "technical_share_of_failures"], d.loc["focus", "technical_share_of_failures"]
                verdict = "supported" if lat >= 1.3 and tech_f - tech_b >= 0.15 else "refuted" if lat < 1.1 and tech_f - tech_b < 0.05 else "inconclusive"
                why = (f"Latency x{lat:.1f} ({d.loc['baseline', 'avg_latency_ms']:.0f} -> {d.loc['focus', 'avg_latency_ms']:.0f} ms); "
                       f"technical share of failures {tech_b * 100:.0f}% -> {tech_f * 100:.0f}%.")
            tests.append({"hypothesis": f"A technical (issuer-side) problem at {dd.top_segment} within {rep.primary.top_segment} caused the drop.",
                          "test_plan": "Check latency and the share of TECHNICAL failure codes in the affected segment, before vs during.",
                          "support_if": "Latency rises >=30% and technical failures become a much larger share.",
                          "sql": sql2, "df": df2, "error": err2, "verdict": verdict, "reasoning": why, "source": "built-in"})

    elif rep.classification == "mix_shift" and rep.primary:
        t = rep.primary.table
        changes = (t["focus_rate"] - t["base_rate"]) * 100
        worst = changes.min()
        share = rep.primary.table.set_index("segment")
        seg = rep.primary.top_segment
        share_up = (share.loc[seg, "focus_share"] - share.loc[seg, "base_share"]) * 100
        verdict = "supported" if worst > -0.6 and share_up >= 5 else "refuted" if worst <= -1.5 else "inconclusive"
        tests.append({"hypothesis": f"No segment got worse; the drop comes from more traffic in {seg}.",
                      "test_plan": f"Compare each {rep.primary.label.lower()}'s own success rate and its traffic share, before vs during.",
                      "support_if": "No segment's rate falls by more than 0.6 pts while the low-converting segment's share rises by 5+ pts.",
                      "sql": rep.evidence[0].sql if rep.evidence else "", "df": rep.primary.table[["segment", "base_rate", "focus_rate", "base_share", "focus_share"]],
                      "error": None, "verdict": verdict,
                      "reasoning": f"Worst within-segment change {worst:+.2f} pts (largest move {changes.abs().max():.2f} pts); {seg} share "
                                   f"{share.loc[seg, 'base_share'] * 100:.1f}% -> {share.loc[seg, 'focus_share'] * 100:.1f}% ({share_up:+.1f} pts).",
                      "source": "built-in"})
    return tests
