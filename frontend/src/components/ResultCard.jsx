import { useState } from "react";
import Chart from "./Chart.jsx";
import DataTable from "./DataTable.jsx";

const CLARIFY_SUFFIX = {
  "Success rate": "success rate",
  "Transaction volume (attempts)": "transaction volume",
  "GMV (successful value)": "GMV",
  "Failure rate": "failure rate",
  "Last week": "last week",
  "Last month": "last month",
  "Last 4 weeks": "last 4 weeks",
};
const STEP_ICON = { ok: "✅", warn: "⚠️", fail: "⛔", skip: "⏭️" };

function Trace({ res }) {
  const llm = res.llm || {};
  return (
    <div>
      {res.trace.map((s, i) => (
        <div className="trace-row" key={i}>
          <span>{STEP_ICON[s.status]}</span>
          <b>{s.agent}</b>
          <span className="detail">{s.detail}</span>
          <span className="ms">{Math.round(s.ms)} ms</span>
        </div>
      ))}
      <p className="muted small" style={{ marginTop: 10 }}>
        LLM calls: {llm.calls} · tokens in/out: {(llm.prompt_tokens || 0).toLocaleString()}/{(llm.completion_tokens || 0).toLocaleString()}
        {llm.models?.length ? ` · models: ${[...new Set(llm.models)].join(", ")}` : ""}
        {llm.errors?.length ? ` · notes: ${llm.errors.join("; ")}` : ""}
      </p>
    </div>
  );
}

function Head({ res, children }) {
  const ctx = res.context || {};
  return (
    <div className="card-head">
      <div>
        <p className="q">{res.original_question || res.question}</p>
        {ctx.used_context && (
          <p className="interp">🧠 Follow-up understood as: <b>{res.question}</b>
            {ctx.carried_over?.length ? <span className="muted"> · carried over {ctx.carried_over.join(", ")}</span> : null}</p>
        )}
        {children}
      </div>
      <span className="meta">asked as <b>{res.role.label}</b></span>
    </div>
  );
}

const inr = (v) => `₹${Math.round(v || 0).toLocaleString("en-IN")}`;
const VERDICT = { supported: "✅ Supported", refuted: "❌ Refuted", inconclusive: "❓ Inconclusive" };

function Impact({ impact }) {
  if (!impact) return null;
  const lost = impact.direction === "lost";
  return (
    <div className="impact">
      <div className="impact-head">💰 Business impact <span className="muted small">· {impact.window.start} → {impact.window.end}</span></div>
      <div className="kpis">
        <div className="kpi"><span className="kv">{(lost ? impact.extra_failed_payments : impact.extra_successful_payments).toLocaleString("en-IN")}</span>
          <span className="kl">{lost ? "extra failed payments" : "extra successful payments"}</span></div>
        <div className="kpi"><span className="kv">{inr(impact.gmv_impact_inr)}</span><span className="kl">GMV {lost ? "at risk (upper bound)" : "gained"}</span></div>
        <div className="kpi"><span className="kv">{inr(impact.monthly_run_rate_inr)}</span><span className="kl">30-day run-rate if it persists</span></div>
        {impact.gmv_impact_pct_of_window_gmv != null && (
          <div className="kpi"><span className="kv">{impact.gmv_impact_pct_of_window_gmv}%</span><span className="kl">of the window's GMV</span></div>
        )}
      </div>
      {impact.scenarios?.length > 0 && (
        <div className="scenarios">
          {impact.scenarios.map((s, i) => (
            <div className="scenario" key={i}>
              <div><b>If we:</b> {s.action}</div>
              <div className="rec">→ recover ~{s.recovered_payments.toLocaleString("en-IN")} payments · {inr(s.recovered_gmv_inr)}</div>
              {s.note && <div className="muted small">{s.note}</div>}
            </div>
          ))}
        </div>
      )}
      <details className="small muted" style={{ marginTop: 6 }}>
        <summary>Assumptions</summary>
        <ul style={{ margin: "4px 0" }}>{impact.assumptions.map((a, i) => <li key={i}>{a}</li>)}</ul>
      </details>
    </div>
  );
}

function HypothesisTests({ tests }) {
  return (
    <>
      <p className="muted small">Each hypothesis was turned into a falsifiable data test, executed through the SQL guard, and judged against its result.</p>
      {tests.map((t, i) => (
        <div key={i} className={`htest ${t.verdict}`}>
          <div className="htest-top">
            <span className={`verdict ${t.verdict}`}>{VERDICT[t.verdict]}</span>
            <b>{t.hypothesis}</b>
          </div>
          <div className="small"><b>Test:</b> {t.test_plan} {t.support_if && <span className="muted">· supports if: {t.support_if}</span>}</div>
          <div className="small" style={{ marginTop: 4 }}><b>Finding:</b> {t.reasoning}</div>
          <details className="ev" style={{ marginTop: 8 }}>
            <summary>Test query & result <span className="muted small">· {t.source}</span></summary>
            <div className="inner">
              {t.sql && <pre className="sql">{t.sql}</pre>}
              {t.error ? <div className="alert warning small">{t.error}</div> : <DataTable table={t.table} maxHeight={220} />}
            </div>
          </details>
        </div>
      ))}
    </>
  );
}

export default function ResultCard({ res, onAsk, busy, currentRole }) {
  const otherRole = currentRole && res.role?.key && res.role.key !== currentRole.key;
  const [tab, setTab] = useState("answer");

  if (res.status === "denied" || res.status === "clarify" || res.status === "error") {
    const opts = res.intent?.clarification?.options || [];
    return (
      <div className="card">
        <Head res={res} />
        <div className="card-body">
          {res.status === "denied" && (
            <div className="alert denied"><b>🔒 Access denied for {res.role.label}</b>{res.message}
              <div className="impact">Blocked by role-based access control before any data was returned.</div>
              {otherRole && (
                <div className="clar-options">
                  <span className="small">You are now signed in as <b>{currentRole.label}</b>.</span>
                  <button className="chip" disabled={busy} onClick={() => onAsk(res.original_question || res.question)}>
                    Run again as {currentRole.label} →
                  </button>
                </div>
              )}
            </div>
          )}
          {res.status === "error" && <div className="alert critical"><b>Could not answer</b>{res.message}</div>}
          {res.status === "clarify" && (
            <div className="alert clarify-box">
              <b>❓ I need a clarification before running a query</b>{res.message}
              <div className="clar-options">
                {opts.map((o) => (
                  <button key={o} className="chip" disabled={busy}
                    onClick={() => onAsk(`${res.question.replace(/\?$/, "")} - ${CLARIFY_SUFFIX[o] || "all time"}`)}>{o}</button>
                ))}
              </div>
            </div>
          )}
          <details className="ev"><summary>Agent trace</summary><div className="inner"><Trace res={res} /></div></details>
        </div>
      </div>
    );
  }

  const n = res.narrative || {};
  const conf = res.confidence || {};
  const checks = Object.fromEntries((res.fact_checks || []).map((c) => [c.statement, c]));
  const cit = res.citations || {};
  const warnCount = (res.dq_issues || []).filter((i) => i.severity !== "info").length;
  const tests = res.hypothesis_tests || [];
  const tabs = [
    ["answer", "Answer"],
    ...(tests.length ? [["tests", "Hypothesis tests", tests.length]] : []),
    ["charts", "Charts", res.charts?.length],
    ["evidence", "Evidence & SQL", 1 + (res.evidence?.length || 0)],
    ["dq", "Data quality & caveats", warnCount || null],
    ["trace", "Agent trace", res.trace.length],
  ];

  return (
    <div className="card">
      <Head res={res} />
      <div className="card-body">
        <div className="headline">{n.headline}</div>
        <div className="badges">
          <span className={`badge ${conf.label}`}>Confidence: {conf.label} ({conf.score}/100)</span>
          <span className="badge">SQL: {res.sql_source}</span>
          <span className="badge">Narrative: {res.narrative_source}</span>
          {res.drivers && <span className="badge">Driver: {res.drivers.classification.replaceAll("_", " ")}</span>}
          {warnCount > 0 && <span className="badge Medium">{warnCount} data warning{warnCount > 1 ? "s" : ""}</span>}
        </div>

        <div className="tabs">
          {tabs.map(([k, label, count]) => (
            <button key={k} className={`tab ${tab === k ? "active" : ""}`} onClick={() => setTab(k)}>
              {label}{count ? <span className="count">{count}</span> : null}
            </button>
          ))}
        </div>

        {tab === "tests" && <HypothesisTests tests={tests} />}

        {tab === "answer" && (
          <>
            <Impact impact={res.impact} />
            <div className="cols3">
              <div className="col-box">
                <h4>✅ Observed facts</h4>
                <ul>
                  {(n.facts || []).map((f, i) => {
                    const c = checks[f];
                    return (
                      <li key={i}>{f}
                        {c && c.verified && c.numbers?.length > 0 && <span className="verified-mark" title="Numbers verified against query results">✔ verified</span>}
                        {c && !c.verified && <span className="unverified">⚠ unverified number(s): {c.unverified.join(", ")}</span>}
                      </li>
                    );
                  })}
                </ul>
              </div>
              <div className="col-box">
                {tests.length ? (
                  <>
                    <h4>🧪 Hypotheses — tested against data</h4>
                    <ul className="hyp-list">
                      {tests.map((t, i) => (
                        <li key={i}><span className={`verdict ${t.verdict}`}>{VERDICT[t.verdict]}</span> {t.hypothesis}
                          <span className="muted small" style={{ display: "block" }}>{t.reasoning}</span></li>
                      ))}
                    </ul>
                    <button className="btn-link" onClick={() => setTab("tests")}>See test queries →</button>
                  </>
                ) : (
                  <>
                    <h4>💡 Hypotheses (not proven)</h4>
                    <ul>{(n.hypotheses?.length ? n.hypotheses : ["None — the data does not point to a specific cause."]).map((h, i) => <li key={i}>{h}</li>)}</ul>
                  </>
                )}
              </div>
              <div className="col-box">
                <h4>🎯 Recommended next steps</h4>
                <ul>{(n.recommendations || []).map((r, i) => <li key={i}>{r}</li>)}</ul>
              </div>
            </div>
            <div className="sources">
              <div><b>Datasets:</b> {(cit.datasets || []).join(", ")} · <b>Access view:</b> <code>{cit.role_view}</code></div>
              <div><b>Time window(s):</b> {(cit.time_windows || []).map((w) => `${w.label} (${w.start} → ${w.end})`).join("; ")}</div>
              <div><b>Filters:</b> {cit.filters && Object.keys(cit.filters).length ? JSON.stringify(cit.filters) : "none"}</div>
              {cit.metric_definition && <div><b>Metric definition:</b> {cit.metric_definition}</div>}
            </div>
            <details className="ev" style={{ marginTop: 10 }}>
              <summary>Why this confidence level?</summary>
              <div className="inner"><ul style={{ margin: 0 }}>{(conf.reasons || []).map((r, i) => <li key={i}>{r}</li>)}</ul></div>
            </details>
          </>
        )}

        {tab === "charts" && (
          <>
            {res.charts?.length ? (
              <div className="chart-grid">
                {res.charts.map((c, i) => <div className="chart-box" key={i}><Chart figure={c.figure} /></div>)}
              </div>
            ) : <p className="muted">No chart for this result shape.</p>}
            <DataTable table={res.result} />
          </>
        )}

        {tab === "evidence" && (
          <>
            <b>Primary query</b> <span className="muted small">({res.sql_source}) {res.sql_explanation}</span>
            <pre className="sql">{res.sql}</pre>
            {res.critic_reviews?.length > 0 && (
              <div className="critic">
                <b>🔍 SQL Critic reviews</b>
                {res.critic_reviews.map((r, i) => (
                  <div key={i} className={`alert ${r.verdict === "approve" ? "info" : "warning"} small`}>
                    <b>Attempt {r.attempt}: {r.verdict === "approve" ? "approved" : "sent back for revision"}</b>
                    <span className="muted">Reviewer: {r.reviewer}</span>
                    {r.issues?.length > 0 && <ul style={{ margin: "4px 0 0" }}>{r.issues.map((x, k) => <li key={k}>{x}</li>)}</ul>}
                    {r.verdict === "revise" && <details><summary>Rejected SQL</summary><pre className="sql">{r.sql}</pre></details>}
                  </div>
                ))}
              </div>
            )}
            {res.sql_attempts?.filter((a) => a.status !== "ok").map((a, i) => (
              <div key={i}>
                <b>Rejected attempt → sent back to SQL agent</b>
                <pre className="sql">{a.sql}</pre>
                <div className="alert warning small">{a.status}: {a.error}</div>
              </div>
            ))}
            {res.drivers?.decomposition && (
              <>
                <b>Decomposition — {res.drivers.primary_dimension}</b>
                <p className="muted small">Change of {res.drivers.delta_pts} pts (z = {res.drivers.z}) split into rate effect (segment got better/worse) and mix effect (traffic moved).</p>
                <DataTable table={res.drivers.decomposition} />
                <div style={{ height: 14 }} />
              </>
            )}
            {res.evidence?.length > 0 && <p><b>Supporting driver queries ({res.evidence.length})</b> <span className="muted small">templated from the semantic layer, run through the same guard</span></p>}
            {res.evidence?.map((e, i) => (
              <details className="ev" key={i}>
                <summary>{e.title} <span className="muted small">· {e.cached ? "cached" : `${e.elapsed_ms} ms`}</span></summary>
                <div className="inner"><pre className="sql">{e.sql}</pre><DataTable table={e.table} maxHeight={240} /></div>
              </details>
            ))}
          </>
        )}

        {tab === "dq" && (
          <>
            {!res.dq_issues?.length && <div className="alert info">No data-quality issues detected for this scope and window.</div>}
            {res.dq_issues?.map((i, k) => (
              <div className={`alert ${i.severity}`} key={k}>
                <b>{i.check.replaceAll("_", " ")} · {i.severity}</b>{i.message}
                {i.impact && <div className="impact">Impact: {i.impact}</div>}
              </div>
            ))}
            <h4 style={{ marginBottom: 6 }}>Assumptions & caveats</h4>
            <ul>{(res.caveats || []).map((c, i) => <li key={i}>{c}</li>)}</ul>
          </>
        )}

        {tab === "trace" && <Trace res={res} />}
      </div>
    </div>
  );
}
