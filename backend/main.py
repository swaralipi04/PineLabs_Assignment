"""FastAPI backend for the Payments AI Business Analyst.

Run (dev):   cd backend && uvicorn main:app --reload --port 8000
Endpoints:
  GET  /api/health
  GET  /api/meta              roles, demo questions, LLM status, data coverage, agent graph
  POST /api/ask               {question, role} -> full analysis result
  POST /api/ask/stream        same, as Server-Sent Events: one event per LangGraph node, then the result
  POST /api/sql/validate      {sql, role} -> run a raw SQL through the guard (safety demo)
  POST /api/cache/clear
If frontend/dist exists (npm run build), it is served at /.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))   # make core/, config importable from any cwd

import serialize  # noqa: E402
from config import ANCHOR_DATE, DATA_END, DATA_START, DB_PATH, LLM_MODEL_BACKUP, LLM_MODEL_FAST, LLM_MODEL_STRONG
from core import anomaly, executor, llm, memory
from core.orchestrator import GRAPH, answer, graph_mermaid, stream_answer
from core.rbac import ROLES
from core.sql_guard import GuardError, pretty

RoleKey = Literal["admin", "business_leader", "regional_analyst_south"]

DEMO_QUESTIONS = [
    {"label": "Real decline (bank incident)", "question": "Why did success rate drop last week?", "role": "business_leader"},
    {"label": "Mix shift (Simpson's paradox)", "question": "Why did success rate fall in the week of Oct 12?", "role": "business_leader"},
    {"label": "Segment breakdown", "question": "Success rate by payment method last month", "role": "business_leader"},
    {"label": "Trend with filter", "question": "Daily UPI success rate for the last 3 weeks", "role": "business_leader"},
    {"label": "Top merchants", "question": "Top 5 merchants by GMV in October with their success rate", "role": "business_leader"},
    {"label": "Two sources disagree", "question": "How many successful transactions did we process in October, and do they match settlements?", "role": "business_leader"},
    {"label": "Ambiguous: metric", "question": "How did we do last month?", "role": "business_leader"},
    {"label": "Ambiguous: time period", "question": "What is the success rate by bank?", "role": "business_leader"},
    {"label": "Access: out of scope", "question": "Show settlement payouts for all regions in October", "role": "regional_analyst_south"},
    {"label": "Access: PII", "question": "List merchant contact emails for top Travel merchants", "role": "regional_analyst_south"},
    {"label": "Data gap check", "question": "Daily attempts by region from Oct 24 to Oct 30", "role": "business_leader"},
]

app = FastAPI(title="Payments AI Business Analyst", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    role: RoleKey = "business_leader"
    session_id: str | None = Field(default=None, max_length=64)   # enables follow-up questions


class SessionRequest(BaseModel):
    session_id: str = Field(max_length=64)


class SQLRequest(BaseModel):
    sql: str = Field(min_length=1, max_length=5000)
    role: RoleKey = "business_leader"


@app.get("/api/health")
def health():
    return {"ok": True, "db": DB_PATH.exists(), "llm": llm.available()}


@app.get("/api/meta")
def meta():
    g = GRAPH.get_graph()
    return {
        "roles": [{"key": r.key, "label": r.label, "description": r.description, "regions": list(r.regions),
                   "tables": sorted(r.tables)} for r in ROLES.values()],
        "demo_questions": DEMO_QUESTIONS,
        "llm": {"available": llm.available(), "provider": "Groq via LangChain", "strong": LLM_MODEL_STRONG, "fast": LLM_MODEL_FAST,
                "backup": LLM_MODEL_BACKUP, "quota": llm.quota_status() if llm.available() else {}},
        "data": {"start": DATA_START.isoformat(), "end": DATA_END.isoformat(), "anchor_date": ANCHOR_DATE.isoformat()},
        "graph": {
            "mermaid": graph_mermaid(),
            "nodes": [n for n in g.nodes],
            "edges": [{"source": e.source, "target": e.target, "conditional": bool(e.conditional)} for e in g.edges],
        },
    }


def _require_db():
    if not DB_PATH.exists():
        raise HTTPException(503, "Database not found. Run `python data/generate_data.py` first.")


@app.post("/api/ask")
def ask(req: AskRequest):
    _require_db()
    res = answer(req.question, req.role, memory.history(req.session_id, req.role))
    memory.remember(req.session_id, req.role, res)
    return serialize.result(res)


@app.post("/api/ask/stream")
def ask_stream(req: AskRequest):
    _require_db()
    turns = memory.history(req.session_id, req.role)

    def events():
        try:
            for kind, *payload in stream_answer(req.question, req.role, turns):
                if kind == "node":
                    node, steps = payload
                    data = {"node": node, "steps": [serialize.step(s) for s in steps]}
                else:
                    memory.remember(req.session_id, req.role, payload[0])
                    data = serialize.result(payload[0])
                yield f"event: {kind}\ndata: {json.dumps(data, default=str)}\n\n"
        except Exception as e:  # surface failures to the client instead of a broken stream
            yield f"event: error\ndata: {json.dumps({'message': f'{type(e).__name__}: {e}'})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/sql/validate")
def validate_sql(req: SQLRequest):
    _require_db()
    try:
        r = executor.run(req.sql, ROLES[req.role])
        return {"allowed": True, "sql": pretty(r.sql), "elapsed_ms": round(r.elapsed_ms, 1), "est_rows": r.est_rows,
                "table": serialize.table(r.df, 100)}
    except GuardError as e:
        return {"allowed": False, "kind": e.kind, "error": str(e)}


@app.get("/api/alerts")
def alerts(role: RoleKey = "business_leader"):
    """Anomaly Watch: proactive scan of every segment for unusual drops, mix shifts and volume gaps."""
    _require_db()
    return anomaly.scan(role)


@app.post("/api/session/reset")
def reset_session(req: SessionRequest):
    memory.forget(req.session_id)
    return {"ok": True}


@app.post("/api/cache/clear")
def clear_cache():
    executor.clear_cache()
    return {"ok": True}


# ---- serve the built React app (production mode)
DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = DIST / path
        return FileResponse(f if path and f.is_file() else DIST / "index.html")
