"""Central configuration. Values can be overridden through environment variables / .env."""
import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent          # backend/
PROJECT_ROOT = ROOT.parent                      # repo root (holds .env)
load_dotenv(PROJECT_ROOT / ".env")

DB_PATH = Path(os.getenv("DB_PATH", ROOT / "data" / "payments.duckdb"))

# Window the synthetic-data GENERATOR produces (13 full Mon-Sun weeks). The analyst itself never
# uses these: it reads the real coverage (min/max txn_date) from the database below.
GEN_START = date.fromisoformat(os.getenv("GEN_START", "2026-08-17"))
GEN_END = date.fromisoformat(os.getenv("GEN_END", "2026-11-15"))


def _db_coverage():
    """Actual data coverage, read from the database (falls back to the generator window before it exists)."""
    if DB_PATH.exists():
        try:
            import duckdb
            with duckdb.connect(str(DB_PATH), read_only=True) as con:
                lo, hi = con.execute("SELECT MIN(txn_date), MAX(txn_date) FROM transactions_certified").fetchone()
                if lo and hi:
                    return lo, hi
        except Exception:
            pass
    return GEN_START, GEN_END


DATA_START, DATA_END = _db_coverage()
# Relative phrases ("last week") resolve against ANCHOR_DATE: set it in .env for reproducible demos,
# otherwise the real current date is used.
ANCHOR_DATE = date.fromisoformat(os.environ["ANCHOR_DATE"]) if os.getenv("ANCHOR_DATE") else date.today()

# LLM (Groq). The app also runs without a key using deterministic fallbacks.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL_STRONG = os.getenv("LLM_MODEL_STRONG", "openai/gpt-oss-120b")
LLM_MODEL_FAST = os.getenv("LLM_MODEL_FAST", "openai/gpt-oss-20b")
# Third model shares the load: each Groq model has its own per-minute and per-day token quota.
LLM_MODEL_BACKUP = os.getenv("LLM_MODEL_BACKUP", "qwen/qwen3.8-27b")
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

# Query safety controls
QUERY_TIMEOUT_S = float(os.getenv("QUERY_TIMEOUT_S", "10"))
MAX_RESULT_ROWS = int(os.getenv("MAX_RESULT_ROWS", "5000"))
MAX_ESTIMATED_ROWS = int(os.getenv("MAX_ESTIMATED_ROWS", "20000000"))
SQL_REPAIR_ATTEMPTS = int(os.getenv("SQL_REPAIR_ATTEMPTS", "2"))
