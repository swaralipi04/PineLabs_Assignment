import { useEffect, useState } from "react";
import { getAlerts } from "../api.js";

const inr = (v) => `₹${Math.round(v).toLocaleString("en-IN")}`;
const TYPE_LABEL = { success_rate_drop: "Success-rate drop", mix_shift: "Mix shift", volume_drop: "Volume gap" };

function Metric({ a }) {
  if (a.type === "success_rate_drop")
    return (
      <div className="aw-metrics">
        <span><b>{(a.baseline_rate * 100).toFixed(1)}% → {(a.rate * 100).toFixed(1)}%</b> success</span>
        <span><b>{a.extra_failed_payments.toLocaleString("en-IN")}</b> extra failed payments</span>
        <span><b>{inr(a.est_gmv_at_risk_inr)}</b> GMV at risk</span>
        <span className="muted">z = {a.worst_z}</span>
      </div>
    );
  if (a.type === "volume_drop")
    return <div className="aw-metrics"><span><b>{a.attempts.toLocaleString()}</b> attempts vs ~{a.typical_attempts.toLocaleString()}/day typical</span></div>;
  return <div className="aw-metrics"><span>Share {(a.baseline_share * 100).toFixed(0)}% → <b>{(a.share * 100).toFixed(0)}%</b></span></div>;
}

export default function AnomalyWatch({ role, onExplain, busy }) {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [open, setOpen] = useState(true);

  useEffect(() => {
    setData(null); setErr(null);
    getAlerts(role).then(setData).catch((e) => setErr(e.message));
  }, [role]);

  return (
    <section className="aw card">
      <div className="aw-head" onClick={() => setOpen((o) => !o)}>
        <div>
          <span className="aw-title">🛰️ Anomaly Watch</span>
          <span className="muted small"> · proactive scan of every bank, method, channel, region & category
            {data ? ` · ${data.window.start} → ${data.window.end}` : ""}</span>
        </div>
        <span className="aw-count">{data ? `${data.alerts.length} alert${data.alerts.length === 1 ? "" : "s"}` : err ? "error" : "scanning…"} {open ? "▾" : "▸"}</span>
      </div>
      {open && (
        <div className="aw-body">
          {err && <div className="alert critical">Scan failed: {err}</div>}
          {!data && !err && <div className="muted small"><span className="spinner inline" /> Scanning segments…</div>}
          {data?.alerts.length === 0 && <div className="alert info">No anomalies detected in your scope.</div>}
          <div className="aw-grid">
            {data?.alerts.map((a, i) => (
              <div key={i} className={`aw-alert ${a.severity}`}>
                <div className="aw-row">
                  <span className={`sev ${a.severity}`}>{a.severity}</span>
                  <span className="muted small">{TYPE_LABEL[a.type]} · {a.dimension} · {a.start}{a.end !== a.start ? ` → ${a.end}` : ""}</span>
                </div>
                <div className="aw-alert-title">{a.title}</div>
                <Metric a={a} />
                {a.hint && <div className="aw-hint">💡 {a.hint}</div>}
                {a.related?.length > 0 && (
                  <div className="aw-related">
                    {a.related.slice(0, 4).map((r, k) => <span key={k} className="rel">{r.title}</span>)}
                  </div>
                )}
                <button className="btn-explain" disabled={busy} onClick={() => onExplain(a.explain_question)}>Explain this →</button>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
