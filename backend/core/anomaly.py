"""Anomaly Watch agent: proactively scans every segment, day by day, for unusual behaviour.

Detectors (all deterministic, role-scoped through the same guarded executor):
  - success-rate drops  : segment's daily rate vs its own trailing-28-day pooled baseline (binomial z-test)
  - mix shifts          : a payment method's / channel's share of traffic moves far from its baseline
  - volume collapses    : daily attempts fall far below the trailing median (often an ingestion gap)
Consecutive flagged days are merged into one alert; overlapping alerts are clustered and the most specific
one leads (e.g. "Kaveri Bank x UPI" rather than "All traffic"). Each alert carries an "explain" question
that runs the full multi-agent pipeline.
"""
from __future__ import annotations

import math
from datetime import timedelta
from functools import lru_cache

import pandas as pd

from config import DATA_END
from core import executor
from core.rbac import ROLES, Role

LOOKBACK_DAYS = 42
BASELINE_DAYS = 28
MIN_N = 60          # z <= -4 keeps false positives rare even on small daily samples

# key: (label, segment SQL expr, joins, specificity, drop threshold)
SCAN_DIMS = {
    "overall": ("All traffic", "'All traffic'", "", 0, 0.010),
    "payment_method": ("Payment method", "pm.method_name", "LEFT JOIN payment_methods pm ON pm.payment_method_id = t.payment_method_id", 1, 0.020),
    "bank": ("Issuer bank", "b.bank_name", "JOIN banks b ON b.bank_id = t.bank_id", 1, 0.020),
    "channel": ("Channel", "t.channel", "", 1, 0.020),
    "region": ("Region", "t.region", "", 1, 0.020),
    "merchant_category": ("Merchant category", "m.category", "LEFT JOIN merchants m ON m.merchant_id = t.merchant_id", 1, 0.025),
    "bank_method": ("Bank x method", "b.bank_name || ' x ' || pm.method_name",
                    "JOIN banks b ON b.bank_id = t.bank_id LEFT JOIN payment_methods pm ON pm.payment_method_id = t.payment_method_id", 2, 0.030),
}
SHARE_DIMS = ("payment_method", "channel")


def _daily(role: Role, key: str, start, end) -> tuple[pd.DataFrame, str]:
    _, expr, join, *_ = SCAN_DIMS[key]
    sql = f"""SELECT t.txn_date, {expr} AS segment,
       SUM(CASE WHEN t.status IN ('SUCCESS','FAILED') THEN 1 ELSE 0 END) AS n,
       SUM(CASE WHEN t.status = 'SUCCESS' THEN 1 ELSE 0 END) AS s,
       COUNT(*) AS attempts,
       SUM(CASE WHEN t.status = 'SUCCESS' THEN t.amount_inr ELSE 0 END) AS gmv
FROM transactions t {join}
WHERE t.txn_date BETWEEN DATE '{start}' AND DATE '{end}'
GROUP BY 1, 2"""
    return executor.run(sql, role).df, sql


def _runs(days: list) -> list[tuple]:
    """Group sorted dates into runs, bridging single-day gaps (an incident rarely trips the test every day)."""
    runs, cur = [], []
    for d in sorted(days):
        if cur and (d - cur[-1]).days > 2:
            runs.append(cur); cur = []
        cur.append(d)
    if cur:
        runs.append(cur)
    return runs


def _seg_question(key: str, seg: str, start, end) -> str:
    phrase = {"overall": "", "payment_method": f"{seg} ", "bank": f"{seg} ", "channel": f"{seg} channel ",
              "region": f"{seg} region ", "merchant_category": f"{seg.lower()} merchants ",
              "bank_method": f"{seg.replace(' x ', ' ')} "}[key]
    return f"Why did {phrase}success rate drop from {start} to {end}?"


def scan(role_key: str) -> dict:
    return _scan_cached(role_key)


@lru_cache(maxsize=8)
def _scan_cached(role_key: str) -> dict:
    role = ROLES[role_key]
    end = DATA_END
    scan_start = end - timedelta(days=LOOKBACK_DAYS - 1)
    q_start = scan_start - timedelta(days=BASELINE_DAYS)
    alerts, queries = [], []
    avg_ticket = None

    for key, (label, _, _, spec, thr) in SCAN_DIMS.items():
        if key == "region" and role.region_restricted:
            continue
        df, sql = _daily(role, key, q_start, end)
        queries.append({"title": f"Daily metrics by {label.lower()}", "sql": sql})
        if df.empty:
            continue
        df["txn_date"] = pd.to_datetime(df["txn_date"]).dt.date
        for c in ("n", "s", "attempts", "gmv"):
            df[c] = df[c].astype(float)
        if key == "overall":
            tot = df[df["txn_date"] >= scan_start]
            avg_ticket = tot["gmv"].sum() / max(tot["s"].sum(), 1)
            totals = df.set_index("txn_date")["attempts"]

        for seg, g in df.groupby("segment"):
            g = g.set_index("txn_date").sort_index()
            full = pd.date_range(q_start, end).date
            g = g.reindex(full, fill_value=0.0)
            base_n = g["n"].shift(1).rolling(BASELINE_DAYS, min_periods=14).sum()
            base_s = g["s"].shift(1).rolling(BASELINE_DAYS, min_periods=14).sum()
            p0 = base_s / base_n
            r = g["s"] / g["n"].where(g["n"] > 0)
            se = (p0 * (1 - p0) / g["n"].where(g["n"] > 0)).pow(0.5)
            z = (r - p0) / se
            in_scan = pd.Series(g.index >= scan_start, index=g.index)

            # 1) success-rate drops
            flag = in_scan & (g["n"] >= MIN_N) & (z <= -4) & ((r - p0) <= -thr)
            for run in _runs([d for d, f in flag.items() if f]):
                run = [d for d in pd.date_range(run[0], run[-1]).date if g.loc[d, "n"] > 0]
                sub = g.loc[run]
                extra = float(((p0.loc[run] - r.loc[run]) * sub["n"]).sum())
                rate = sub["s"].sum() / sub["n"].sum()
                base = float((p0.loc[run] * sub["n"]).sum() / sub["n"].sum())
                alerts.append({
                    "type": "success_rate_drop", "dimension": label, "dim_key": key, "segment": str(seg), "specificity": spec,
                    "start": str(run[0]), "end": str(run[-1]), "days": len(run),
                    "rate": rate, "baseline_rate": base, "drop_pts": round((rate - base) * 100, 2),
                    "worst_z": round(float(z.loc[run].min()), 1), "resolved_attempts": int(sub["n"].sum()),
                    "extra_failed_payments": int(round(extra)),
                    "title": f"{seg if key != 'overall' else 'Overall'} success rate down {abs(rate - base) * 100:.1f} pts",
                    "explain_question": _seg_question(key, str(seg), run[0], run[-1]),
                })

            # 2) mix shifts (share of daily attempts)
            if key in SHARE_DIMS:
                share = g["attempts"] / totals.reindex(full).fillna(0).replace(0, math.nan)
                base_share = share.shift(1).rolling(BASELINE_DAYS, min_periods=14).mean()
                sflag = in_scan & ((share - base_share).abs() >= 0.08)
                for run in _runs([d for d, f in sflag.items() if f]):
                    s_now, s_base = float(share.loc[run].mean()), float(base_share.loc[run].mean())
                    alerts.append({
                        "type": "mix_shift", "dimension": label, "dim_key": key, "segment": str(seg), "specificity": spec,
                        "start": str(run[0]), "end": str(run[-1]), "days": len(run),
                        "share": s_now, "baseline_share": s_base, "share_change_pts": round((s_now - s_base) * 100, 1),
                        "extra_failed_payments": 0,
                        "title": f"{seg} share of traffic {'up' if s_now > s_base else 'down'} {abs(s_now - s_base) * 100:.0f} pts",
                        "explain_question": f"Why did success rate fall from {run[0]} to {run[-1]}?",
                    })

            # 3) volume collapse (ingestion gaps) - per region / overall
            if key in ("region", "overall"):
                med = g["attempts"].shift(1).rolling(BASELINE_DAYS, min_periods=14).median()
                vflag = in_scan & (g["attempts"] < 0.4 * med)
                for run in _runs([d for d, f in vflag.items() if f]):
                    alerts.append({
                        "type": "volume_drop", "dimension": label, "dim_key": key, "segment": str(seg), "specificity": spec,
                        "start": str(run[0]), "end": str(run[-1]), "days": len(run),
                        "attempts": int(g.loc[run, "attempts"].sum()), "typical_attempts": int(med.loc[run].mean()),
                        "extra_failed_payments": 0,
                        "title": f"{seg} volume collapsed ({int(g.loc[run, 'attempts'].sum()):,} vs ~{int(med.loc[run].mean()):,}/day)",
                        "explain_question": f"Daily attempts by region from {run[0] - timedelta(days=3)} to {run[-1] + timedelta(days=3)}",
                    })

    for a in alerts:
        a["est_gmv_at_risk_inr"] = round(a["extra_failed_payments"] * (avg_ticket or 0), 0)
        a["severity"] = ("critical" if a["type"] == "success_rate_drop" and (abs(a["drop_pts"]) >= 5 or a["extra_failed_payments"] >= 1500)
                         else "warning" if a["type"] in ("success_rate_drop", "volume_drop") else "info")
        for k in ("rate", "baseline_rate", "share", "baseline_share"):
            if k in a:
                a[k] = round(a[k], 4)
    return {"window": {"start": str(scan_start), "end": str(end)}, "alerts": _cluster(alerts), "queries": queries,
            "avg_ticket_inr": round(avg_ticket or 0, 2)}


def _cluster(alerts: list[dict]) -> list[dict]:
    """Group overlapping alerts; the most specific high-impact alert leads, the rest become 'related'."""
    order = sorted(alerts, key=lambda a: a["start"])
    clusters: list[list[dict]] = []
    for a in order:
        for c in clusters:
            if any(a["start"] <= b["end"] and b["start"] <= a["end"] for b in c):
                c.append(a); break
        else:
            clusters.append([a])
    out = []
    sev_rank = {"critical": 0, "warning": 1, "info": 2}
    for c in clusters:
        drops = [a for a in c if a["type"] == "success_rate_drop"]
        pool = drops or [a for a in c if a["type"] == "volume_drop"] or c
        top = max(a["extra_failed_payments"] for a in pool)
        cands = [a for a in pool if a["extra_failed_payments"] >= 0.6 * top] or pool
        lead = max(cands, key=lambda a: (a["specificity"], a["extra_failed_payments"]))
        lead = dict(lead)
        lead["severity"] = min((a["severity"] for a in c), key=sev_rank.get)
        if lead["type"] == "success_rate_drop":  # describe the whole incident window, not just the lead's flagged days
            lead["start"], lead["end"] = min(a["start"] for a in c), max(a["end"] for a in c)
            lead["days"] = (pd.Timestamp(lead["end"]) - pd.Timestamp(lead["start"])).days + 1
            lead["explain_question"] = _seg_question(lead["dim_key"], lead["segment"], lead["start"], lead["end"])
        lead["related"] = [{"title": a["title"], "dimension": a["dimension"], "type": a["type"], "start": a["start"], "end": a["end"]}
                           for a in c if a is not None and a["title"] != lead["title"]][:6]
        # a cluster with a big mix shift but only an overall drop is a mix story
        if lead["dim_key"] == "overall" and any(a["type"] == "mix_shift" for a in c):
            lead["hint"] = "Overall drop coincides with a traffic-mix shift - possibly not a performance problem."
        out.append(lead)
    out.sort(key=lambda a: (sev_rank[a["severity"]], -a["extra_failed_payments"], a["start"]))
    return out
