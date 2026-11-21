import { useEffect, useRef, useState } from "react";
import { askStream, clearCache, getMeta, resetSession } from "./api.js";
import AnomalyWatch from "./components/AnomalyWatch.jsx";
import Pipeline, { NODES } from "./components/Pipeline.jsx";
import ResultCard from "./components/ResultCard.jsx";
import SqlConsole from "./components/SqlConsole.jsx";

const ORDER = NODES.map((n) => n.id);
const STATUS_FROM_STEP = { ok: "done", warn: "warn", fail: "fail", skip: "skipped" };
const freshPipeline = () => Object.fromEntries(ORDER.map((id) => [id, { status: "pending", visits: 0 }]));
const newSessionId = () => (crypto.randomUUID ? crypto.randomUUID() : `s-${Date.now()}-${Math.random().toString(36).slice(2)}`);

export default function App() {
  const [meta, setMeta] = useState(null);
  const [metaError, setMetaError] = useState(null);
  const [role, setRole] = useState("business_leader");
  const [history, setHistory] = useState([]);
  const [busy, setBusy] = useState(false);
  const [pending, setPending] = useState(null);
  const [pipe, setPipe] = useState(freshPipeline);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState(newSessionId);
  const feedRef = useRef(null);

  function newConversation() {
    resetSession(sessionId);
    setSessionId(newSessionId());
    setHistory([]);
    setPipe(freshPipeline());
  }

  useEffect(() => {
    getMeta().then(setMeta).catch((e) => setMetaError(e.message));
  }, []);

  // keep the newest answer in view
  useEffect(() => {
    feedRef.current?.scrollTo({ top: 0, behavior: "smooth" });
  }, [history.length]);

  async function ask(question, roleOverride) {
    const q = question.trim();
    if (!q || busy) return;
    const r = roleOverride || role;
    if (roleOverride) setRole(roleOverride);
    setBusy(true);
    setInput("");
    setPending({ question: q, role: r });
    setPipe({ ...freshPipeline(), context: { status: "running", visits: 0 } });
    feedRef.current?.scrollTo({ top: 0, behavior: "smooth" });

    try {
      const res = await askStream(q, r, ({ node, steps }) => {
        setPipe((p) => {
          const next = { ...p };
          const worst = steps.some((s) => s.status === "fail") ? "fail"
            : steps.some((s) => s.status === "warn") ? "warn"
            : steps.some((s) => s.status === "skip") ? "skipped" : "done";
          const last = steps[steps.length - 1];
          next[node] = { status: STATUS_FROM_STEP[last?.status] ? worst : "done", detail: last?.detail, visits: (p[node]?.visits || 0) + 1 };
          // guess the next node to show as running (both loops go back to the SQL agent)
          const repair = (node === "guard_execute" && worst === "fail")
            || (node === "sql_critic" && steps.some((s) => s.detail?.startsWith("revise")));
          const nxt = repair ? "sql_agent" : ORDER[ORDER.indexOf(node) + 1];
          if (nxt && (repair || next[nxt]?.status === "pending")) next[nxt] = { ...next[nxt], status: "running" };
          return next;
        });
      }, sessionId);
      setPipe((p) => Object.fromEntries(Object.entries(p).map(([k, v]) => [k, v.status === "pending" || v.status === "running" ? { ...v, status: "skipped" } : v])));
      setHistory((h) => [{ id: Date.now(), res }, ...h]);
      getMeta().then(setMeta).catch(() => {});  // refresh LLM quota status
    } catch (e) {
      setHistory((h) => [{ id: Date.now(), res: { question: q, role: { label: r }, status: "error", message: e.message, trace: [] } }, ...h]);
      setPipe(freshPipeline());
    } finally {
      setPending(null);
      setBusy(false);
    }
  }

  const roleObj = meta?.roles.find((x) => x.key === role);

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-word">pine labs</span>
          <span className="brand-ai">.AI</span>
          <p className="brand-sub">AI Business Analyst · Payments</p>
        </div>

        <div>
          <p className="section-title">Signed in as (simulated)</p>
          <select className="select" value={role} onChange={(e) => { setRole(e.target.value); setSessionId(newSessionId()); }} disabled={busy}>
            {meta?.roles.map((r) => <option key={r.key} value={r.key}>{r.label}</option>)}
          </select>
          {roleObj && <p className="hint">{roleObj.description}</p>}
        </div>

        {metaError && <div className="status-pill warn"><span className="dot warn" />Backend unreachable: {metaError}. Start it with <code>uvicorn backend.main:app --port 8000</code>.</div>}

        <div>
          <p className="section-title">Agent pipeline (LangGraph) {busy && "· live"}</p>
          <Pipeline state={pipe} />
        </div>

        <div>
          <p className="section-title">Demo questions</p>
          <div className="demo-list">
            {meta?.demo_questions.map((d) => (
              <button key={d.label} className="demo-btn" disabled={busy} onClick={() => ask(d.question)} title={d.question}>
                <span className="lbl">{d.label}</span>
                <span className="q">{d.question}</span>
                {d.role !== "business_leader" && (
                  <span className="role-tag">
                    {d.role === role ? "🔒 will be blocked for your role" : `🔒 blocked only for ${meta.roles.find((r) => r.key === d.role)?.label} — switch role to see it`}
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>

        <details>
          <summary className="section-title" style={{ cursor: "pointer" }}>🛡️ SQL guard console</summary>
          <p className="hint" style={{ marginBottom: 8 }}>Paste any SQL to see the validator accept or block it for the current role.</p>
          <SqlConsole role={role} />
        </details>

        <button className="btn-ghost" onClick={() => { newConversation(); clearCache(); }} disabled={busy}>
          Clear history & cache
        </button>
      </aside>

      <main className="main">
        <div className="feed" ref={feedRef}>
          <div className="feed-inner">
            <AnomalyWatch role={role} busy={busy} onExplain={(q) => ask(q)} />
            {pending && (
              <div className="card running-card">
                <div className="card-head"><p className="q">{pending.question}</p><span className="meta">{meta?.roles.find((r) => r.key === pending.role)?.label}</span></div>
                <div className="card-body"><div className="spinner" /> Agents working — follow progress in the pipeline panel…</div>
              </div>
            )}
            {!pending && history.length === 0 && (
              <div className="empty">
                <h2>Ask a question about payment performance</h2>
                <p>Every answer shows its SQL, segment drivers, charts, data-quality checks and a confidence score.</p>
                <div className="chips">
                  {meta?.demo_questions.slice(0, 4).map((d) => (
                    <button key={d.label} className="chip" onClick={() => ask(d.question)}>{d.question}</button>
                  ))}
                </div>
              </div>
            )}
            {history.map(({ id, res }) => (
              <ResultCard key={id} res={res} onAsk={(q) => ask(q)} busy={busy} currentRole={roleObj} />
            ))}
          </div>
        </div>
        <div className="composer">
          <form onSubmit={(e) => { e.preventDefault(); ask(input); }}>
            {history.length > 0 && (
              <button type="button" className="btn-ghost" onClick={newConversation} disabled={busy} title="Forget previous turns (follow-ups use them)">
                ＋ New conversation
              </button>
            )}
            <input value={input} onChange={(e) => setInput(e.target.value)}
              placeholder={history.length ? "Ask a follow-up, e.g. “what about netbanking?” or “break that down by channel”" : "e.g. Why did success rate drop last week?"}
              disabled={busy} />
            <button className="btn-primary" type="submit" disabled={busy || !input.trim()}>{busy ? "Working…" : "Ask"}</button>
          </form>
        </div>
      </main>
    </div>
  );
}
