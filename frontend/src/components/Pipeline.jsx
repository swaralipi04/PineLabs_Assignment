// Live view of the LangGraph agent pipeline. Node states come from the SSE stream.

export const NODES = [
  { id: "context", name: "Conversation Memory", sub: "rewrites follow-ups into full questions" },
  { id: "access_guard", name: "Access Guard", sub: "RBAC pre-check on the question" },
  { id: "intent", name: "Intent Agent", sub: "LLM + rules · clarifies ambiguity" },
  { id: "sql_agent", name: "SQL Agent", sub: "LLM text-to-SQL on semantic layer" },
  { id: "sql_critic", name: "SQL Critic", sub: "reviews meaning before execution" },
  { id: "guard_execute", name: "SQL Guard + Executor", sub: "sqlglot checks · DuckDB · cost limit" },
  { id: "driver_analysis", name: "Driver Analysis", sub: "rate vs mix decomposition · drill-down" },
  { id: "data_quality", name: "Data Quality", sub: "dupes · nulls · gaps · reconciliation" },
  { id: "impact", name: "Business Impact", sub: "failed payments → ₹ · recovery sizing" },
  { id: "visualise", name: "Visualisation", sub: "auto charts" },
  { id: "narrator", name: "Narrator", sub: "facts · hypotheses · actions" },
  { id: "hypothesis_tester", name: "Hypothesis Tester", sub: "runs a data test per hypothesis" },
  { id: "fact_check", name: "Fact-check + Confidence", sub: "every number verified" },
];

const ICON = { done: "✓", warn: "!", fail: "✕", running: "•", skipped: "–", pending: "" };

export default function Pipeline({ state }) {
  return (
    <div className="pipeline">
      {NODES.map((n) => {
        const s = state[n.id] || { status: "pending" };
        return (
          <div key={n.id} className={`pnode ${s.status}`} title={s.detail || n.sub}>
            <div className="bullet">{ICON[s.status]}</div>
            <div style={{ minWidth: 0 }}>
              <div className="name">{n.name}</div>
              <div className="sub">{n.sub}</div>
            </div>
            {s.visits > 1 && <span className="loop">↻ ×{s.visits}</span>}
          </div>
        );
      })}
    </div>
  );
}
