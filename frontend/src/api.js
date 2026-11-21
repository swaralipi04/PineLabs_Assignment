// Thin client for the FastAPI backend.

export async function getMeta() {
  const r = await fetch("/api/meta");
  if (!r.ok) throw new Error(`meta: HTTP ${r.status}`);
  return r.json();
}

export async function validateSql(sql, role) {
  const r = await fetch("/api/sql/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sql, role }),
  });
  return r.json();
}

export async function clearCache() {
  await fetch("/api/cache/clear", { method: "POST" });
}

export async function getAlerts(role) {
  const r = await fetch(`/api/alerts?role=${encodeURIComponent(role)}`);
  if (!r.ok) throw new Error(`alerts: HTTP ${r.status}`);
  return r.json();
}

export async function resetSession(sessionId) {
  await fetch("/api/session/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
}

/**
 * Ask a question and stream LangGraph progress.
 * onNode({node, steps}) is called as each agent node finishes; resolves with the final result.
 */
export async function askStream(question, role, onNode, sessionId) {
  const r = await fetch("/api/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, role, session_id: sessionId }),
  });
  if (!r.ok || !r.body) {
    let msg = `HTTP ${r.status}`;
    try { msg = (await r.json()).detail || msg; } catch { /* ignore */ }
    throw new Error(msg);
  }
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let result = null;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const event = /^event: (.*)$/m.exec(chunk)?.[1];
      const dataLine = /^data: (.*)$/m.exec(chunk)?.[1];
      if (!event || !dataLine) continue;
      const data = JSON.parse(dataLine);
      if (event === "node") onNode?.(data);
      else if (event === "result") result = data;
      else if (event === "error") throw new Error(data.message);
    }
  }
  if (!result) throw new Error("Stream ended without a result");
  return result;
}
