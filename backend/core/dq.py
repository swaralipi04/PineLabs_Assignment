"""Data Quality agent (deterministic). Runs checks scoped to the question's window and the user's role.

It reads the raw feed through an internal read-only connection, but only aggregate
counts within the role's regions are ever surfaced.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import duckdb

from config import DATA_END, DATA_START, DB_PATH
from core.rbac import Role
from core.timeparse import Window


@dataclass
class DQIssue:
    severity: str   # info | warning | critical
    check: str
    message: str
    impact: str = ""

    def to_dict(self):
        return asdict(self)


def _con():
    return duckdb.connect(str(DB_PATH), read_only=True)


def _region_sql(role: Role, col="region") -> str:
    if not role.region_restricted:
        return "TRUE"
    return f"{col} IN ({', '.join(repr(r) for r in role.regions)})"


def run_checks(role: Role, windows: list[Window], tables: set[str] | None = None, reconcile: bool = False) -> list[DQIssue]:
    issues: list[DQIssue] = []
    if not windows:
        return issues
    start = min(w.start for w in windows)
    end = max(w.end for w in windows)
    rf = _region_sql(role)

    # Coverage
    if end < DATA_START or start > DATA_END:
        issues.append(DQIssue("critical", "coverage", f"Requested window {start}..{end} is outside available data ({DATA_START}..{DATA_END}).",
                              "No data can answer this question."))
        return issues
    if start < DATA_START or end > DATA_END:
        issues.append(DQIssue("warning", "coverage", f"Window {start}..{end} only partially overlaps available data ({DATA_START}..{DATA_END}).",
                              "Totals for the uncovered days are missing, not zero."))

    con = _con()
    try:
        w = f"txn_date BETWEEN '{start}' AND '{end}' AND {rf}"
        raw, cert = con.execute(f"SELECT (SELECT COUNT(*) FROM raw_transactions WHERE {w}), (SELECT COUNT(*) FROM transactions_certified WHERE {w})").fetchone()
        if raw and raw > cert:
            dup = raw - cert
            pct = dup / raw * 100
            worst = con.execute(f"""SELECT txn_date, COUNT(*) - COUNT(DISTINCT txn_id) d FROM raw_transactions WHERE {w}
                                    GROUP BY 1 ORDER BY d DESC LIMIT 1""").fetchone()
            issues.append(DQIssue(
                "warning" if pct > 0.3 else "info", "duplicates",
                f"{dup:,} duplicate records ({pct:.2f}%) found in the raw transaction feed for this window"
                + (f"; worst day {worst[0]} ({worst[1]:,} duplicates)." if worst and worst[1] else "."),
                "Duplicates were removed (certified layer keeps first ingested copy). Raw-feed counts would be overstated."))

        nulls, tot = con.execute(f"SELECT SUM(CASE WHEN bank_id IS NULL THEN 1 ELSE 0 END), COUNT(*) FROM transactions_certified WHERE {w}").fetchone()
        if tot and nulls:
            pct = nulls / tot * 100
            issues.append(DQIssue("warning" if pct >= 2 else "info", "missing_values",
                                  f"bank_id is missing on {nulls:,} transactions ({pct:.1f}%).",
                                  "These appear as 'Unknown (missing bank_id)' in bank breakdowns; bank-level rates exclude them."))

        gaps = con.execute(f"""
            WITH d AS (SELECT region, txn_date, COUNT(*) n FROM transactions_certified
                       WHERE txn_date BETWEEN DATE '{start}' - INTERVAL 14 DAY AND DATE '{end}' + INTERVAL 1 DAY AND {rf} GROUP BY 1,2),
                 days AS (SELECT CAST(x AS DATE) txn_date FROM range(DATE '{max(start, DATA_START)}', DATE '{min(end, DATA_END)}' + INTERVAL 1 DAY, INTERVAL 1 DAY) r(x)),
                 regs AS (SELECT DISTINCT region FROM d),
                 med AS (SELECT region, MEDIAN(n) m FROM d GROUP BY 1)
            SELECT regs.region, days.txn_date, COALESCE(d.n, 0) n, med.m
            FROM regs CROSS JOIN days JOIN med USING (region)
            LEFT JOIN d ON d.region = regs.region AND d.txn_date = days.txn_date
            WHERE COALESCE(d.n, 0) < 0.3 * med.m ORDER BY 2""").fetchall()
        for region, d, n, m in gaps[:5]:
            issues.append(DQIssue("warning", "ingestion_gap",
                                  f"{region} region has {n:,} transactions on {d} vs a typical {int(m):,}/day - likely an ingestion gap.",
                                  "Volumes for that day are understated; rates for that day are unreliable."))

        pend, resolved = con.execute(f"SELECT SUM(CASE WHEN status='PENDING' THEN 1 ELSE 0 END), COUNT(*) FROM transactions_certified WHERE {w}").fetchone()
        if resolved and pend / resolved > 0.02:
            issues.append(DQIssue("info", "pending", f"{pend / resolved * 100:.1f}% of attempts are still PENDING.", "Excluded from success-rate denominator."))

        if reconcile and "settlements" in role.tables:
            issues.extend(_reconcile(con, start, end))
    finally:
        con.close()
    return issues


def _reconcile(con, start, end) -> list[DQIssue]:
    r = con.execute(f"""
        WITH t AS (SELECT merchant_id, txn_date, COUNT(*) n FROM transactions_certified
                   WHERE status='SUCCESS' AND txn_date BETWEEN '{start}' AND '{end}' GROUP BY 1,2),
             s AS (SELECT merchant_id, txn_date, txn_count FROM settlements WHERE txn_date BETWEEN '{start}' AND '{end}')
        SELECT COUNT(*) FILTER (WHERE s.txn_count IS NOT NULL AND s.txn_count <> t.n) mismatched,
               COUNT(*) FILTER (WHERE s.txn_count IS NULL) missing,
               COUNT(*) total,
               SUM(t.n) txn_total, SUM(COALESCE(s.txn_count,0)) settle_total
        FROM t LEFT JOIN s USING (merchant_id, txn_date)""").fetchone()
    mismatched, missing, total, tt, st = r
    out = []
    if total and (mismatched or missing):
        out.append(DQIssue(
            "warning", "source_conflict",
            f"Transactions vs settlements disagree: {mismatched:,} of {total:,} merchant-days have different success counts "
            f"and {missing:,} merchant-days have no settlement record. Totals: transactions={int(tt):,}, settlements={int(st):,} "
            f"({(st - tt) / tt * 100:+.2f}%).",
            "Policy: transaction counts/success rates use the transactions table (system of record); money paid out uses settlements. "
            "Mismatches should be raised with the settlements team."))
    return out
