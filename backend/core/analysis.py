"""Driver analysis: explains a change in success rate between two windows.

For each dimension we decompose the change exactly into
    Delta = sum_i (w_b,i - w_a,i) * (r_a,i - R_a)    <- MIX effect   (traffic moved between segments)
          + sum_i  w_b,i * (r_b,i - r_a,i)            <- RATE effect  (segments themselves got better/worse)
where w = share of resolved attempts, r = segment success rate, R_a = overall baseline rate.

All SQL is templated from the semantic layer and executed through the same guard as LLM SQL,
so every number is backed by a visible query.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import pandas as pd

from core import executor
from core.rbac import Role
from core.semantic import DIMENSIONS, DRIVER_DIMENSIONS
from core.timeparse import Window

MIN_SEGMENT_N = 200


@dataclass
class Evidence:
    title: str
    sql: str
    df: pd.DataFrame
    elapsed_ms: float = 0
    cached: bool = False


@dataclass
class DimResult:
    dimension: str
    label: str
    table: pd.DataFrame
    mix_total: float
    rate_total: float
    top_segment: str
    top_contribution: float
    concentration: float     # share of |delta| explained by the single top segment


@dataclass
class DriverReport:
    base: Window
    focus: Window
    base_rate: float
    focus_rate: float
    base_n: int
    focus_n: int
    delta: float
    z: float
    significant: bool
    classification: str      # rate_driven | mix_shift | no_material_change
    dims: list[DimResult] = field(default_factory=list)
    primary: DimResult | None = None
    drilldown: list[DimResult] = field(default_factory=list)
    drill_filter: str = ""
    failure_codes: pd.DataFrame | None = None
    evidence: list[Evidence] = field(default_factory=list)

    def summary(self) -> dict:
        """Compact, LLM-friendly summary (numbers pre-computed, never invented by the LLM)."""
        def dim_rows(d: DimResult, k=4):
            t = d.table.reindex(d.table["contribution"].abs().sort_values(ascending=False).index).head(k)
            return [{"segment": r.segment, "base_rate": _p(r.base_rate), "focus_rate": _p(r.focus_rate),
                     "base_share": _p(r.base_share), "focus_share": _p(r.focus_share),
                     "rate_effect_pts": round(r.rate_effect * 100, 2), "mix_effect_pts": round(r.mix_effect * 100, 2),
                     "focus_attempts": int(r.focus_n)} for r in t.itertuples()]
        return {
            "base_window": self.base.to_dict(), "focus_window": self.focus.to_dict(),
            "base_success_rate": _p(self.base_rate), "focus_success_rate": _p(self.focus_rate),
            "change_pts": round(self.delta * 100, 2), "z_score": round(self.z, 1), "statistically_significant": self.significant,
            "classification": self.classification,
            "dimension_summary": [{"dimension": d.label, "mix_effect_pts": round(d.mix_total * 100, 2), "rate_effect_pts": round(d.rate_total * 100, 2),
                                   "top_segment": d.top_segment, "top_segment_contribution_pts": round(d.top_contribution * 100, 2),
                                   "concentration": round(d.concentration, 2)} for d in self.dims],
            "primary_dimension": dim_rows(self.primary) if self.primary else None,
            "primary_dimension_name": self.primary.label if self.primary else None,
            "drilldown_within": self.drill_filter,
            "drilldown": {d.label: dim_rows(d, 3) for d in self.drilldown[:2]},
            "failure_codes_in_focus_segment": (self.failure_codes.head(5).to_dict("records") if self.failure_codes is not None else None),
        }


def _p(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x) * 100, 2)


def merge_joins(*parts: str) -> str:
    """Combine JOIN clauses from several sources, dropping exact duplicates (avoids duplicate-alias errors)."""
    out = []
    for p in parts:
        for j in re.split(r"(?=\b(?:LEFT\s+|INNER\s+)?JOIN\b)", p or "", flags=re.I):
            j = " ".join(j.split())
            if j and j not in out:
                out.append(j)
    return " ".join(out)


def _sr_sql(dim_key: str, base: Window, focus: Window, extra_where: str = "", extra_join: str = "") -> str:
    label, expr, join = DIMENSIONS[dim_key]
    joins = merge_joins(join, extra_join)
    lo, hi = min(base.start, focus.start), max(base.end, focus.end)
    def agg(w: Window, succ: bool):
        cond = "t.status = 'SUCCESS'" if succ else "t.status IN ('SUCCESS','FAILED')"
        return f"SUM(CASE WHEN t.txn_date BETWEEN DATE '{w.start}' AND DATE '{w.end}' AND {cond} THEN 1 ELSE 0 END)"
    return f"""SELECT {expr} AS segment,
       {agg(base, False)} AS base_n, {agg(base, True)} AS base_s,
       {agg(focus, False)} AS focus_n, {agg(focus, True)} AS focus_s
FROM transactions t {joins}
WHERE t.txn_date BETWEEN DATE '{lo}' AND DATE '{hi}'{(' AND ' + extra_where) if extra_where else ''}
GROUP BY 1"""


def _decompose(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float, float]:
    df = df.copy()
    for c in ["base_n", "base_s", "focus_n", "focus_s"]:
        df[c] = df[c].astype(float)
    Na, Nb = df.base_n.sum(), df.focus_n.sum()
    Ra = df.base_s.sum() / Na if Na else float("nan")
    Rb = df.focus_s.sum() / Nb if Nb else float("nan")
    df["base_share"] = df.base_n / Na if Na else 0
    df["focus_share"] = df.focus_n / Nb if Nb else 0
    df["base_rate"] = (df.base_s / df.base_n).where(df.base_n > 0)
    df["focus_rate"] = (df.focus_s / df.focus_n).where(df.focus_n > 0)
    br = df.base_rate.fillna(Ra)
    fr = df.focus_rate.fillna(br)
    df["mix_effect"] = (df.focus_share - df.base_share) * (br - Ra)
    df["rate_effect"] = df.focus_share * (fr - br)
    df["contribution"] = df.mix_effect + df.rate_effect
    return df.sort_values("contribution"), Ra, Rb, Nb


def _dim_result(key: str, df: pd.DataFrame, delta: float) -> DimResult:
    t, *_ = _decompose(df)
    sig = t[(t.focus_n >= MIN_SEGMENT_N) | (t.base_n >= MIN_SEGMENT_N)]
    pool = sig if len(sig) else t
    # segment pushing hardest in the direction of the overall change
    sign = -1 if delta < 0 else 1
    top = pool.loc[(pool.contribution * sign).idxmax()]
    denom = abs(delta) if abs(delta) > 1e-9 else 1e-9
    return DimResult(key, DIMENSIONS[key][0], t, t.mix_effect.sum(), t.rate_effect.sum(),
                     str(top.segment), float(top.contribution), float(abs(top.contribution) / denom))


def explain_change(role: Role, focus: Window, base: Window | None = None, where: str = "", join: str = "") -> DriverReport:
    base = base or focus.previous()
    evidence: list[Evidence] = []
    dim_frames = {}
    for key in DRIVER_DIMENSIONS:
        if key == "region" and role.region_restricted and len(role.regions) == 1:
            continue
        sql = _sr_sql(key, base, focus, where, join)
        r = executor.run(sql, role)
        dim_frames[key] = r.df
        evidence.append(Evidence(f"Success rate by {DIMENSIONS[key][0].lower()}, {base.label} vs {focus.label}", sql, r.df, r.elapsed_ms, r.cached))

    first = next(iter(dim_frames.values()))
    _, Ra, Rb, Nb = _decompose(first)
    Na = float(first.base_n.astype(float).sum())
    delta = Rb - Ra
    p = (first.base_s.astype(float).sum() + first.focus_s.astype(float).sum()) / (Na + Nb)
    se = math.sqrt(max(p * (1 - p) * (1 / Na + 1 / Nb), 1e-12)) if Na and Nb else float("inf")
    z = delta / se if se else 0.0
    significant = abs(z) >= 3 and abs(delta) >= 0.002

    dims = [_dim_result(k, df, delta) for k, df in dim_frames.items()]
    rep = DriverReport(base, focus, Ra, Rb, int(Na), int(Nb), delta, z, significant, "no_material_change", dims, evidence=evidence)
    if not significant:
        rep.primary = max(dims, key=lambda d: d.concentration)
        return rep

    # Mix shift: one dimension's mix effect explains most of the change while its rate effect is small
    mix_dim = max(dims, key=lambda d: (d.mix_total / delta))
    if mix_dim.mix_total / delta >= 0.6 and abs(mix_dim.rate_total) < 0.4 * abs(delta):
        rep.classification = "mix_shift"
        rep.primary = mix_dim
        return rep

    rep.classification = "rate_driven"
    rate_dims = [d for d in dims if d.dimension != "region" or not role.region_restricted]
    rep.primary = max(rate_dims, key=lambda d: d.concentration)

    # Drill down one level inside the top segment of the primary dimension
    prim = rep.primary
    _, expr, pjoin = DIMENSIONS[prim.dimension]
    seg = prim.top_segment.replace("'", "''")
    drill_where = f"{expr} = '{seg}'" + (f" AND {where}" if where else "")
    drill_join = merge_joins(pjoin, join)
    rep.drill_filter = f"{prim.label} = {prim.top_segment}"
    for key in DRIVER_DIMENSIONS:
        if key == prim.dimension or (key == "region" and role.region_restricted):
            continue
        sql = _sr_sql(key, base, focus, drill_where, drill_join)
        r = executor.run(sql, role)
        rep.drilldown.append(_dim_result(key, r.df, prim.top_contribution))
        evidence.append(Evidence(f"Drill-down within {rep.drill_filter}: by {DIMENSIONS[key][0].lower()}", sql, r.df, r.elapsed_ms, r.cached))
    rep.drilldown.sort(key=lambda d: d.concentration, reverse=True)

    # Failure reasons inside the focus segment
    fsql = f"""SELECT t.failure_code, fc.failure_category,
       SUM(CASE WHEN t.txn_date BETWEEN DATE '{base.start}' AND DATE '{base.end}' THEN 1 ELSE 0 END) AS base_failures,
       SUM(CASE WHEN t.txn_date BETWEEN DATE '{focus.start}' AND DATE '{focus.end}' THEN 1 ELSE 0 END) AS focus_failures
FROM transactions t {merge_joins(drill_join, "LEFT JOIN failure_codes fc ON fc.failure_code = t.failure_code")}
WHERE t.status = 'FAILED' AND t.txn_date BETWEEN DATE '{min(base.start, focus.start)}' AND DATE '{max(base.end, focus.end)}' AND {drill_where}
GROUP BY 1, 2 ORDER BY focus_failures - base_failures DESC"""
    r = executor.run(fsql, role)
    rep.failure_codes = r.df
    evidence.append(Evidence(f"Failure reasons within {rep.drill_filter}", fsql, r.df, r.elapsed_ms, r.cached))
    return rep
