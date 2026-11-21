import { useState } from "react";
import { validateSql } from "../api.js";

const EXAMPLES = [
  "SELECT * FROM transactions a, transactions b",
  "DROP TABLE transactions",
  "SELECT * FROM read_csv('/etc/passwd')",
  "SELECT region, COUNT(*) FROM transactions GROUP BY 1",
];

export default function SqlConsole({ role }) {
  const [sql, setSql] = useState(EXAMPLES[0]);
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);

  async function run() {
    setBusy(true);
    try { setRes(await validateSql(sql, role)); } catch (e) { setRes({ allowed: false, kind: "error", error: e.message }); }
    setBusy(false);
  }

  return (
    <div className="console">
      <textarea value={sql} onChange={(e) => setSql(e.target.value)} spellCheck={false} />
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 6 }}>
        <button className="btn-ghost" onClick={run} disabled={busy}>{busy ? "Running…" : "Validate & run"}</button>
        <select className="btn-ghost" value="" onChange={(e) => { if (e.target.value) { setSql(e.target.value); setRes(null); } }}>
          <option value="">Examples…</option>
          {EXAMPLES.map((x) => <option key={x} value={x}>{x.slice(0, 40)}</option>)}
        </select>
      </div>
      {res && (
        <div className={`res ${res.allowed ? "ok" : "bad"}`}>
          {res.allowed
            ? <>Allowed · {res.table?.total_rows ?? 0} rows · {res.elapsed_ms} ms{res.est_rows ? ` · est. ${res.est_rows.toLocaleString()} rows` : ""}</>
            : <><b>Blocked ({res.kind})</b> {res.error}</>}
        </div>
      )}
    </div>
  );
}
