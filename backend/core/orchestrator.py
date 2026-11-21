"""Orchestrator: the multi-agent pipeline as a LangGraph StateGraph.

  START -> context (conversation memory) -> access_guard -(denied)-> END
        -> intent -(ambiguous)-> END (clarification)
        -> sql_agent -> sql_critic -(revise)-> sql_agent                       [semantic review loop]
                             \-> guard_execute -(rejected & repairable)-> sql_agent   [safety repair loop]
                                         \-(access denied / unrecoverable)-> END
                                         \-(why question)-> driver_analysis -> data_quality
                                         \-> data_quality -> impact -> visualise -> narrator
        -> hypothesis_tester -> fact_check (+ confidence) -> END
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from config import SQL_REPAIR_ATTEMPTS
from core import agents, charts, critic, dq, executor, hypotheses, impact, llm, memory
from core.analysis import DriverReport, Evidence, explain_change
from core.rbac import ROLES, Role, precheck_question
from core.semantic import METRICS, fmt_value
from core.sql_guard import GuardError, pretty


@dataclass
class Step:
    agent: str
    status: str      # ok | warn | fail | skip
    detail: str
    ms: float = 0.0


@dataclass
class AnalysisResult:
    question: str
    role: Role
    status: str = "answered"                 # answered | clarify | denied | error
    message: str = ""
    intent: agents.Intent | None = None
    trace: list[Step] = field(default_factory=list)
    sql: str = ""
    sql_explanation: str = ""
    sql_source: str = ""
    sql_attempts: list[dict] = field(default_factory=list)
    result: pd.DataFrame | None = None
    drivers: DriverReport | None = None
    evidence: list[Evidence] = field(default_factory=list)
    dq_issues: list[dq.DQIssue] = field(default_factory=list)
    charts: list[tuple[str, object]] = field(default_factory=list)
    narrative: dict = field(default_factory=dict)
    narrative_source: str = ""
    fact_checks: list[dict] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    confidence: dict = field(default_factory=dict)
    citations: dict = field(default_factory=dict)
    llm_stats: llm.LLMStats = field(default_factory=llm.LLMStats)
    original_question: str = ""                      # what the user typed (if a follow-up was rewritten)
    context: dict = field(default_factory=dict)      # conversation-memory resolution details
    critic_reviews: list[dict] = field(default_factory=list)
    impact: dict | None = None
    hypothesis_tests: list[dict] = field(default_factory=list)


class _Timer:
    def __init__(self, res: AnalysisResult, agent: str):
        self.res, self.agent = res, agent

    def __enter__(self):
        self.t0 = time.perf_counter(); return self

    def log(self, status, detail):
        self.res.trace.append(Step(self.agent, status, detail, (time.perf_counter() - self.t0) * 1000))

    def __exit__(self, *a):
        return False


# =============================================================================== LangGraph pipeline
class GraphState(TypedDict, total=False):
    res: AnalysisResult
    attempt: int
    feedback: Optional[str]
    prev_sql: Optional[str]
    candidate: Optional[dict]      # {"sql", "explanation", "source"}
    exec_outcome: str              # ok | repair | denied | failed
    critic_outcome: str            # approve | revise
    history: list                  # previous turns of this conversation


def node_context(state: GraphState) -> dict:
    res = state["res"]
    turns = state.get("history") or []
    with _Timer(res, "Conversation Memory") as t:
        r = memory.resolve(res.question, turns, res.llm_stats)
        res.context = {**r, "turns_available": len(turns)}
        if r["used_context"]:
            res.original_question, res.question = res.question, r["question"]
            t.log("ok", f"Follow-up rewritten via {r['source']}: \"{r['question']}\" (carried over: {', '.join(r['carried_over']) or '-'})")
        else:
            t.log("skip" if not turns else "ok", "No previous turns" if not turns else "Standalone question; no context needed")
    return {}


def node_access_guard(state: GraphState) -> dict:
    res = state["res"]
    with _Timer(res, "Access Guard") as t:
        reason = precheck_question(res.question, res.role)
        if reason:
            t.log("fail", reason)
            res.status, res.message = "denied", reason
        else:
            t.log("ok", f"Role {res.role.label}: tables {sorted(res.role.tables)}; regions {', '.join(res.role.regions)}")
    return {}


def node_intent(state: GraphState) -> dict:
    res = state["res"]
    with _Timer(res, "Intent Agent") as t:
        it = agents.understand(res.question, res.role, res.llm_stats)
        res.intent = it
        t.log("ok", f"intent={it.intent}, metric={it.metric}, windows={[w.label for w in it.windows]}, "
                    f"filters={it.filters or '{}'}, group_by={it.group_by or '[]'} (via {it.source})")
    if it.clarification:
        res.trace.append(Step("Clarifier", "warn", it.clarification["question"]))
        res.status, res.message = "clarify", it.clarification["question"]
    return {}


def node_sql_agent(state: GraphState) -> dict:
    res, attempt = state["res"], state.get("attempt", 0)
    with _Timer(res, "SQL Agent") as t:
        sql, expl, src = agents.generate_sql(res.intent, res.role, state.get("feedback"), state.get("prev_sql"), res.llm_stats)
        t.log("ok", f"attempt {attempt + 1}: generated by {src}" + (" (repairing after feedback)" if state.get("feedback") else ""))
    return {"candidate": {"sql": sql, "explanation": expl, "source": src}}


def node_sql_critic(state: GraphState) -> dict:
    res, cand, attempt = state["res"], state["candidate"], state.get("attempt", 0)
    with _Timer(res, "SQL Critic") as t:
        rev = critic.review(cand["sql"], res.intent, cand["source"], res.llm_stats)
        res.critic_reviews.append({"attempt": attempt + 1, "sql": cand["sql"], **rev})
        if rev["verdict"] == "revise" and attempt < SQL_REPAIR_ATTEMPTS:
            t.log("warn", f"revise ({rev['reviewer']}): {'; '.join(rev['issues'])[:220]}")
            return {"critic_outcome": "revise", "attempt": attempt + 1, "prev_sql": cand["sql"],
                    "feedback": "A reviewer found semantic problems: " + " ".join(rev["issues"])}
        if rev["verdict"] == "revise":
            t.log("warn", f"issues remain after {attempt + 1} attempt(s); proceeding with a caveat: {'; '.join(rev['issues'])[:200]}")
            res.caveats.append("SQL reviewer concerns were not fully resolved: " + "; ".join(rev["issues"]))
        else:
            t.log("ok", f"approved ({rev['reviewer']})")
    return {"critic_outcome": "approve"}


def node_guard_execute(state: GraphState) -> dict:
    res, cand, attempt = state["res"], state["candidate"], state.get("attempt", 0)
    with _Timer(res, "SQL Guard + Executor") as t:
        try:
            qr = executor.run(cand["sql"], res.role)
        except GuardError as e:
            res.sql_attempts.append({"sql": cand["sql"], "status": f"rejected ({e.kind})", "error": str(e), "source": cand["source"]})
            t.log("fail", f"{e.kind}: {e}")
            if e.kind == "access_denied":
                res.status, res.message = "denied", str(e)
                return {"exec_outcome": "denied"}
            can_repair = cand["source"] == "llm" and attempt < SQL_REPAIR_ATTEMPTS
            if not can_repair:
                res.status = "error"
                res.message = f"Could not produce a valid, safe query for this question. {e}"
                return {"exec_outcome": "failed"}
            return {"exec_outcome": "repair", "attempt": attempt + 1, "feedback": str(e), "prev_sql": cand["sql"]}
        res.sql_attempts.append({"sql": cand["sql"], "status": "ok", "source": cand["source"]})
        t.log("ok", f"{len(qr.df):,} rows in {qr.elapsed_ms:.0f} ms"
                    + (f", est. {qr.est_rows:,} rows scanned" if qr.est_rows else "") + (" (cache hit)" if qr.cached else ""))
        res.sql, res.sql_explanation, res.sql_source = cand["sql"], cand["explanation"], cand["source"]
        res.result = qr.df
        res.caveats += qr.notes
    return {"exec_outcome": "ok"}


def node_driver_analysis(state: GraphState) -> dict:
    res = state["res"]
    it = res.intent
    with _Timer(res, "Driver Analysis Agent") as t:
        if it.metric not in ("success_rate", "failure_rate", "success_rate_all_attempts"):
            t.log("skip", "Driver decomposition is implemented for rate metrics only.")
            return {}
        try:
            where, joins = agents.filter_clauses(it.filters)
            focus = it.windows[0]
            base = it.windows[1] if len(it.windows) > 1 else None
            rep = explain_change(res.role, focus, base, " AND ".join(where), " ".join(joins))
            res.drivers, res.evidence = rep, rep.evidence
            t.log("ok", f"{rep.classification}: {rep.delta * 100:+.2f} pts (z={rep.z:.1f}); primary driver "
                        f"{rep.primary.label if rep.primary else '-'} -> {rep.primary.top_segment if rep.primary else '-'}; "
                        f"{len(rep.evidence)} supporting queries")
        except GuardError as e:
            t.log("fail", str(e))
    return {}


def node_data_quality(state: GraphState) -> dict:
    res = state["res"]
    it = res.intent
    with _Timer(res, "Data Quality Agent") as t:
        wins = list(it.windows) + ([res.drivers.base] if res.drivers else [])
        tables = set()
        for a in res.sql_attempts:
            tables |= {m.lower() for m in re.findall(r"\b(transactions|settlements|merchants|banks)\b", a["sql"], re.I)}
        reconcile = "settlements" in tables or bool(re.search(r"settle|reconcil|payout|match", res.question, re.I))
        res.dq_issues = dq.run_checks(res.role, wins, tables, reconcile=reconcile)
        worst = max((i.severity for i in res.dq_issues), key=["info", "warning", "critical"].index, default=None)
        t.log("ok" if worst in (None, "info") else "warn", f"{len(res.dq_issues)} issue(s)" + (f", worst={worst}" if worst else ""))
    res.citations = {
        "datasets": sorted(tables) or ["transactions"],
        "role_view": f"{res.role.schema}.* ({res.role.label})",
        "time_windows": [w.to_dict() for w in wins],
        "filters": it.filters or {},
        "metric_definition": METRICS[it.metric].definition if it.metric in METRICS else None,
    }
    return {}


def node_visualise(state: GraphState) -> dict:
    res = state["res"]
    with _Timer(res, "Visualisation Agent") as t:
        shade = (res.drivers.focus.start, res.drivers.focus.end) if res.drivers else None
        fig = charts.auto_chart(res.result, _chart_title(res.intent), shade)
        if fig:
            res.charts.append(("Query result", fig))
        if res.drivers and res.drivers.classification != "no_material_change":
            res.charts.append(("Driver decomposition", charts.driver_chart(res.drivers)))
            if res.drivers.drilldown:
                dd = res.drivers.drilldown[0]
                res.charts.append(("Drill-down", charts.driver_chart(
                    res.drivers, dd, f"Drill-down within {res.drivers.drill_filter}: by {dd.label.lower()}")))
            res.charts.append(("Segment comparison", charts.segment_compare_chart(res.drivers)))
        t.log("ok", f"{len(res.charts)} chart(s)")
    return {}


def node_impact(state: GraphState) -> dict:
    res = state["res"]
    with _Timer(res, "Business Impact Agent") as t:
        if not res.drivers:
            t.log("skip", "Impact sizing runs on 'why did it change' questions.")
            return {}
        if res.drivers.classification == "no_material_change":
            t.log("skip", "No material change to size.")
            return {}
        try:
            where, joins = agents.filter_clauses(res.intent.filters)
            res.impact, ev = impact.compute(res.role, res.drivers, where, joins)
            res.evidence += ev
            t.log("ok", res.impact["headline"] if res.impact else "nothing to size")
        except GuardError as e:
            t.log("fail", str(e))
    return {}


def node_narrator(state: GraphState) -> dict:
    res = state["res"]
    with _Timer(res, "Narrator Agent") as t:
        res.narrative, res.narrative_source = agents.narrate(res.question, _evidence_for_narrator(res), res.llm_stats)
        t.log("ok", f"via {res.narrative_source}")
    return {}


def node_hypothesis_tester(state: GraphState) -> dict:
    res = state["res"]
    with _Timer(res, "Hypothesis Tester") as t:
        if not res.drivers or res.drivers.classification == "no_material_change":
            t.log("skip", "Hypothesis testing runs when a 'why' question finds a material change.")
            return {}
        res.hypothesis_tests = hypotheses.run_tests(res.question, res.narrative.get("hypotheses") or [], res.drivers, res.role,
                                                    res.intent.filters, res.llm_stats)
        counts = {v: sum(1 for x in res.hypothesis_tests if x["verdict"] == v) for v in ("supported", "refuted", "inconclusive")}
        t.log("ok" if counts["supported"] else "warn",
              f"{len(res.hypothesis_tests)} test(s): {counts['supported']} supported, {counts['refuted']} refuted, {counts['inconclusive']} inconclusive")
    return {}


def node_fact_check(state: GraphState) -> dict:
    res = state["res"]
    evidence = _evidence_for_narrator(res)
    with _Timer(res, "Fact Checker") as t:
        res.fact_checks = fact_check(res.narrative, evidence)
        bad = [f for f in res.fact_checks if not f["verified"]]
        t.log("ok" if not bad else "warn", f"{len(res.fact_checks) - len(bad)}/{len(res.fact_checks)} statements numerically verified against evidence")
    res.caveats = res.intent.assumptions + [f"{i.message} {i.impact}".strip() for i in res.dq_issues if i.severity != "info"] + res.caveats
    if res.sql_source == "template":
        res.caveats.append("SQL was produced by the deterministic template builder (LLM unavailable or failed).")
    res.confidence = confidence(res)
    return {}


# ---- conditional edges
def route_after_guard(state: GraphState) -> str:
    return "end" if state["res"].status == "denied" else "intent"


def route_after_intent(state: GraphState) -> str:
    return "end" if state["res"].status == "clarify" else "sql_agent"


def route_after_critic(state: GraphState) -> str:
    return "sql_agent" if state.get("critic_outcome") == "revise" else "guard_execute"


def route_after_execute(state: GraphState) -> str:
    outcome = state["exec_outcome"]
    if outcome == "repair":
        return "sql_agent"
    if outcome != "ok":
        return "end"
    it = state["res"].intent
    return "driver_analysis" if it.intent == "explain_change" and it.windows else "data_quality"


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("context", node_context)
    g.add_node("access_guard", node_access_guard)
    g.add_node("intent", node_intent)
    g.add_node("sql_agent", node_sql_agent)
    g.add_node("sql_critic", node_sql_critic)
    g.add_node("guard_execute", node_guard_execute)
    g.add_node("driver_analysis", node_driver_analysis)
    g.add_node("data_quality", node_data_quality)
    g.add_node("impact", node_impact)
    g.add_node("visualise", node_visualise)
    g.add_node("narrator", node_narrator)
    g.add_node("hypothesis_tester", node_hypothesis_tester)
    g.add_node("fact_check", node_fact_check)

    g.add_edge(START, "context")
    g.add_edge("context", "access_guard")
    g.add_conditional_edges("access_guard", route_after_guard, {"intent": "intent", "end": END})
    g.add_conditional_edges("intent", route_after_intent, {"sql_agent": "sql_agent", "end": END})
    g.add_edge("sql_agent", "sql_critic")
    g.add_conditional_edges("sql_critic", route_after_critic, {"sql_agent": "sql_agent", "guard_execute": "guard_execute"})
    g.add_conditional_edges("guard_execute", route_after_execute,
                            {"sql_agent": "sql_agent", "driver_analysis": "driver_analysis", "data_quality": "data_quality", "end": END})
    g.add_edge("driver_analysis", "data_quality")
    g.add_edge("data_quality", "impact")
    g.add_edge("impact", "visualise")
    g.add_edge("visualise", "narrator")
    g.add_edge("narrator", "hypothesis_tester")
    g.add_edge("hypothesis_tester", "fact_check")
    g.add_edge("fact_check", END)
    return g.compile()


GRAPH = build_graph()


def graph_mermaid() -> str:
    return GRAPH.get_graph().draw_mermaid()


def answer(question: str, role_key: str = "business_leader", history: list | None = None) -> AnalysisResult:
    res = AnalysisResult(question=question, role=ROLES[role_key])
    GRAPH.invoke({"res": res, "attempt": 0, "history": history or []}, {"recursion_limit": 40})
    return res


def stream_answer(question: str, role_key: str = "business_leader", history: list | None = None):
    """Yield ("node", name, new_trace_steps) as each LangGraph node finishes, then ("result", res)."""
    res = AnalysisResult(question=question, role=ROLES[role_key])
    seen = 0
    for update in GRAPH.stream({"res": res, "attempt": 0, "history": history or []}, {"recursion_limit": 40}, stream_mode="updates"):
        for node in update:
            yield "node", node, res.trace[seen:]
            seen = len(res.trace)
    yield "result", res


def _chart_title(it: agents.Intent) -> str:
    m = METRICS.get(it.metric or "", None)
    return f"{m.label if m else 'Result'}" + (f" - {it.windows[0].label}" if it.windows else "")


def _rows_for_llm(df: pd.DataFrame, n=40) -> list[dict]:
    d = df.head(n).copy()
    for c in d.columns:
        if pd.api.types.is_numeric_dtype(d[c]) and any(k in c.lower() for k in ("rate", "share")):
            d[c] = (d[c].astype(float) * 100).round(2)
            d = d.rename(columns={c: f"{c}_pct"})
        elif pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(2)
    return d.astype(object).where(d.notna(), None).to_dict("records")


def _evidence_for_narrator(res: AnalysisResult) -> dict:
    it = res.intent
    ev = {
        "metric": it.metric, "metric_definition": METRICS[it.metric].definition if it.metric in METRICS else None,
        "windows": [w.to_dict() for w in it.windows], "filters": it.filters, "access_scope": list(res.role.regions),
        "query_result_rows": _rows_for_llm(res.result) if res.result is not None else [],
        "query_row_count": 0 if res.result is None else len(res.result),
        "data_quality_issues": [i.to_dict() for i in res.dq_issues],
        "assumptions": it.assumptions,
    }
    if res.drivers:
        ev["drivers"] = res.drivers.summary()
    else:
        ev["result_summary"] = agents.summarize_result(res.result, it.metric)
    if res.impact:
        ev["business_impact"] = {k: v for k, v in res.impact.items() if k != "assumptions"}
    return ev


# ------------------------------------------------------------------------- fact checking
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b|\b20\d{2}\b", re.I)
_NUM_RE = re.compile(r"(?<![\w.])[-+]?\d[\d,]*\.?\d*")


def _collect_numbers(obj, out: list[float]):
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, out)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        for m in _NUM_RE.findall(_DATE_RE.sub(" ", obj)):
            try:
                out.append(float(m.replace(",", "")))
            except ValueError:
                pass


_UNIT_SCALE = {"crore": 1e7, "cr": 1e7, "lakh": 1e5, "lac": 1e5, "lakhs": 1e5, "million": 1e6, "mn": 1e6, "m": 1e6,
               "thousand": 1e3, "k": 1e3, "billion": 1e9, "bn": 1e9}
_NUM_UNIT_RE = re.compile(r"(?<![\w.])([-+]?\d[\d,]*\.?\d*)\s*(crore|cr|lakhs?|lac|million|mn|m|thousand|k|billion|bn)?\b", re.I)


def _matches(x: float, pool: list[float], unit: str | None = None) -> bool:
    """A quoted number matches evidence if it equals a value (or its percent form). With a unit word
    ("₹19.66 lakh"), it must match value / that unit's scale - so a lakh figure can't pass as crore."""
    scale = _UNIT_SCALE.get((unit or "").lower())
    for p in pool:
        cands = (p / scale, abs(p) / scale) if scale else (p, p * 100, abs(p), abs(p) * 100)
        for cand in cands:
            if abs(cand - x) <= max(0.051, abs(cand) * 0.01):
                return True
    return False


def fact_check(narrative: dict, evidence: dict) -> list[dict]:
    pool: list[float] = []
    _collect_numbers(evidence, pool)
    # derived values the narrator may legitimately quote: differences and ratios between evidence numbers
    d = evidence.get("drivers") or {}
    drill_rows = [r for rows in (d.get("drilldown") or {}).values() for r in rows]
    for seg in (d.get("primary_dimension") or []) + drill_rows:
        if seg.get("base_rate") is not None and seg.get("focus_rate") is not None:
            pool.append(seg["focus_rate"] - seg["base_rate"])
        if seg.get("base_share") is not None and seg.get("focus_share") is not None:
            pool.append(seg["focus_share"] - seg["base_share"])
    for fc in (d.get("failure_codes_in_focus_segment") or []):
        if fc.get("base_failures"):
            pool.append(fc["focus_failures"] / fc["base_failures"])
            pool.append(fc["focus_failures"] - fc["base_failures"])
    checks = []
    for kind in ("headline", "facts"):
        items = narrative.get(kind) or []
        for s in ([items] if isinstance(items, str) else items):
            found = []
            for n, unit in _NUM_UNIT_RE.findall(_DATE_RE.sub(" ", str(s))):
                if not n.replace(",", "").replace(".", "").strip("+-"):
                    continue
                v = float(n.replace(",", ""))
                if abs(v) < 10 and v.is_integer() and not unit:  # ignore small counts like "3 banks"
                    continue
                found.append((v, unit or None))
            nums = [v for v, _ in found]
            missing = [f"{v:g} {u}" if u else v for v, u in found if not _matches(v, pool, u)]
            checks.append({"statement": s, "kind": kind, "numbers": nums, "unverified": missing, "verified": not missing})
    return checks


def confidence(res: AnalysisResult) -> dict:
    score, reasons = 90, []
    for i in res.dq_issues:
        if i.severity == "critical":
            score = min(score, 20); reasons.append(f"Critical data issue: {i.check}")
        elif i.severity == "warning":
            score -= 7; reasons.append(f"Data-quality warning: {i.check}")
    repairs = sum(1 for a in res.sql_attempts if a["status"] != "ok")
    if repairs:
        score -= 5 * repairs; reasons.append(f"SQL needed {repairs} repair attempt(s)")
    if res.drivers:
        d = res.drivers
        if d.classification == "no_material_change":
            reasons.append(f"Change of {d.delta * 100:+.2f} pts is not statistically significant (z={d.z:.1f})")
        elif d.primary and d.primary.concentration < 0.5 and d.classification == "rate_driven":
            score -= 15; reasons.append("Change is spread across many segments; no single dominant driver")
        else:
            reasons.append(f"Change is significant (z={d.z:.1f}) and concentrated in {d.primary.top_segment if d.primary else 'n/a'}")
        if d.focus_n < 2000:
            score -= 15; reasons.append(f"Small sample ({d.focus_n:,} resolved attempts)")
    elif res.result is not None and res.result.empty:
        score -= 30; reasons.append("Query returned no rows")
    unverified = [f for f in res.fact_checks if not f["verified"]]
    if unverified:
        score -= min(20, 10 * len(unverified)); reasons.append(f"{len(unverified)} narrative statement(s) contain numbers not found in evidence")
    if any(r["verdict"] == "revise" for r in res.critic_reviews[-1:]):
        score -= 10; reasons.append("SQL reviewer still had concerns about the final query")
    tests = res.hypothesis_tests
    if tests:
        sup = sum(1 for x in tests if x["verdict"] == "supported")
        ref = sum(1 for x in tests if x["verdict"] == "refuted")
        if sup:
            score += 5; reasons.append(f"{sup} of {len(tests)} hypotheses supported by data tests (see Hypothesis tests tab)")
        if ref == len(tests):
            score -= 10; reasons.append("Every proposed explanation was refuted by its data test - root cause still unknown")
        elif ref:
            reasons.append(f"{ref} hypothesis(es) refuted by data tests and should be discounted")
    reasons.append("Data tests show consistency with a cause, not proof; confirm with an external source (e.g. bank incident logs).")
    score = max(5, min(score, 95))
    label = "High" if score >= 75 else "Medium" if score >= 50 else "Low"
    return {"score": score, "label": label, "reasons": reasons}
