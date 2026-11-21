"""Query executor: read-only DuckDB, per-role schema, EXPLAIN cost check, timeout, cache."""
from __future__ import annotations

import hashlib
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import duckdb
import pandas as pd

from config import DB_PATH, MAX_ESTIMATED_ROWS, MAX_RESULT_ROWS, QUERY_TIMEOUT_S
from core.rbac import Role
from core.sql_guard import GuardError, GuardResult, validate


@dataclass
class QueryResult:
    sql: str
    df: pd.DataFrame
    elapsed_ms: float
    cached: bool
    est_rows: int | None
    notes: list[str]


class _LRU(OrderedDict):
    def __init__(self, cap=256):
        super().__init__(); self.cap = cap

    def get_(self, k):
        if k in self:
            self.move_to_end(k); return self[k]
        return None

    def put(self, k, v):
        self[k] = v; self.move_to_end(k)
        if len(self) > self.cap:
            self.popitem(last=False)


_cache = _LRU()
_lock = threading.Lock()


def connect(role: Role, path=DB_PATH) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(path), read_only=True)
    con.execute(f"SET search_path = '{role.schema}'")
    con.execute("SET enable_external_access = false")
    return con


def estimate_rows(con, sql: str) -> int | None:
    """Largest cardinality estimate in the physical plan (DuckDB 'EC'/'~N rows')."""
    try:
        plan = "\n".join(str(r[-1]) for r in con.execute(f"EXPLAIN {sql}").fetchall())
    except duckdb.Error as e:
        raise GuardError(f"Query failed planning: {str(e).splitlines()[0]}")
    nums = [int(n.replace(",", "")) for n in re.findall(r"(?:EC:\s*|~)([\d,]+)\s*(?:rows)?", plan)]
    return max(nums) if nums else None


def run(sql: str, role: Role, use_cache: bool = True, timeout_s: float = QUERY_TIMEOUT_S) -> QueryResult:
    guard: GuardResult = validate(sql, role)
    key = hashlib.sha1(f"{role.key}|{guard.sql}".encode()).hexdigest()
    if use_cache:
        with _lock:
            hit = _cache.get_(key)
        if hit is not None:
            return QueryResult(guard.sql, hit[0].copy(), 0.0, True, hit[1], guard.notes + ["Served from cache."])

    con = connect(role)
    try:
        est = estimate_rows(con, guard.sql)
        if est is not None and est > MAX_ESTIMATED_ROWS:
            raise GuardError(f"Estimated {est:,} intermediate rows exceeds the cost limit of {MAX_ESTIMATED_ROWS:,}. "
                             "Narrow the time window or aggregate.", "expensive")
        timer = threading.Timer(timeout_s, con.interrupt)
        t0 = time.perf_counter()
        timer.start()
        try:
            df = con.execute(guard.sql).df()
        except duckdb.InterruptException:
            raise GuardError(f"Query exceeded the {timeout_s:.0f}s timeout and was cancelled.", "expensive")
        except duckdb.Error as e:
            raise GuardError(f"Execution error: {str(e).splitlines()[0]}")
        finally:
            timer.cancel()
        elapsed = (time.perf_counter() - t0) * 1000
    finally:
        con.close()
    notes = list(guard.notes)
    if len(df) >= MAX_RESULT_ROWS:
        notes.append(f"Result truncated at the {MAX_RESULT_ROWS:,}-row safety cap.")
    with _lock:
        _cache.put(key, (df, est))
    return QueryResult(guard.sql, df, elapsed, False, est, notes)


def clear_cache():
    with _lock:
        _cache.clear()
