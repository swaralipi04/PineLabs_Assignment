"""Evaluate the pipeline against known-answer questions.

  numeric    : generated SQL's result must match a hand-written reference query (values, order-insensitive)
  driver     : the driver agent must reach the planted classification and name the planted segments
  behaviour  : access denials, clarifications and data-quality warnings must fire when expected

Run: python backend/eval/run_eval.py      (writes eval/results.md)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import anomaly, executor, llm, memory  # noqa: E402
from core.orchestrator import answer  # noqa: E402
from core.rbac import ROLES  # noqa: E402

HERE = Path(__file__).parent


def _num_cols(df: pd.DataFrame) -> list[np.ndarray]:
    out = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(np.sort(df[c].astype(float).dropna().to_numpy()))
    return out


def results_match(gen: pd.DataFrame, ref: pd.DataFrame) -> tuple[bool, str]:
    if gen is None or gen.empty:
        return False, "no rows"
    g_cols = _num_cols(gen)
    for i, r in enumerate(_num_cols(ref)):
        ok = any(len(g) == len(r) and (np.allclose(g, r, rtol=1e-3, atol=1e-6) or np.allclose(g, r * 100, rtol=1e-3)) for g in g_cols)
        if not ok:
            return False, f"reference column {i} ({len(r)} values) not found in generated result"
    return True, "match"


def main():
    cases = json.loads((HERE / "questions.json").read_text())
    rows = []
    executor.clear_cache()
    for c in cases:
        t0 = time.perf_counter()
        ok, why = False, ""
        if c["type"] == "anomaly":
            res = None
            scan = anomaly.scan(c["role"])
            leads = [a["segment"] for a in scan["alerts"]]
            types = {a["type"] for a in scan["alerts"]}
            dims = {a["dimension"] for a in scan["alerts"]} | {r["dimension"] for a in scan["alerts"] for r in a["related"]}
            ok = all(l in leads for l in c["expect_leads"]) and all(t in types for t in c["expect_types"])
            if c.get("expect_mix_hint"):
                ok = ok and any(a.get("hint") for a in scan["alerts"])
            if c.get("expect_no_dims"):
                ok = ok and not (set(c["expect_no_dims"]) & dims)
            why = f"{len(scan['alerts'])} alerts, leads={leads}, types={sorted(types)}"
            rows.append({"id": c["id"], "type": c["type"], "role": c["role"], "question": "(anomaly scan)", "pass": ok,
                         "detail": why, "sql_source": "-", "llm_calls": 0, "seconds": round(time.perf_counter() - t0, 2)})
            print(f"{'PASS' if ok else 'FAIL'}  {c['id']:4s} {'(anomaly scan)':70s} {why}")
            continue
        if c["type"] == "followup":
            sid = f"eval-{c['id']}"
            memory.forget(sid)
            for q in c["turns"]:
                res = answer(q, c["role"], memory.history(sid, c["role"]))
                memory.remember(sid, c["role"], res)
            e, it = c["expect"], res.intent
            checks = [res.status == e.get("status", res.status)]
            if "metric" in e:
                checks.append(it is not None and it.metric == e["metric"])
            if "window_start" in e:
                checks.append(it is not None and bool(it.windows) and str(it.windows[0].start) == e["window_start"])
            for k, v in (e.get("filters") or {}).items():
                checks.append(it is not None and str(it.filters.get(k, "")).upper() == v.upper())
            ok = all(checks)
            why = f"rewritten='{res.question[:60]}', status={res.status}" + (f", metric={it.metric}, filters={it.filters}" if it else "")
            rows.append({"id": c["id"], "type": c["type"], "role": c["role"], "question": " → ".join(c["turns"]), "pass": ok,
                         "detail": why, "sql_source": res.sql_source or "-", "llm_calls": res.llm_stats.calls,
                         "seconds": round(time.perf_counter() - t0, 2)})
            print(f"{'PASS' if ok else 'FAIL'}  {c['id']:4s} {' → '.join(c['turns'])[:70]:70s} {why}")
            continue
        res = answer(c["question"], c["role"])
        if c["type"] == "numeric":
            if res.status != "answered":
                why = f"status={res.status}: {res.message[:80]}"
            else:
                ref = executor.run(c["reference_sql"], ROLES[c["role"]], use_cache=False).df
                ok, why = results_match(res.result, ref)
        elif c["type"] == "driver":
            d = res.drivers
            if not d:
                why = f"no driver report (status={res.status})"
            else:
                named = {d.primary.top_segment} | {x.top_segment for x in d.drilldown[:1]} if d.primary else set()
                seg_ok = all(s in named for s in c["expect_segments"])
                ok = d.classification == c["expect_classification"] and seg_ok
                why = f"got {d.classification}, segments {sorted(named)}"
                if c.get("expect_impact"):
                    ok = ok and bool(res.impact and res.impact.get("gmv_impact_inr"))
                    why += f", impact={'yes' if res.impact else 'no'}"
                if c.get("expect_supported_min"):
                    sup = sum(1 for t in res.hypothesis_tests if t["verdict"] == "supported")
                    ok = ok and sup >= c["expect_supported_min"]
                    why += f", hypotheses supported {sup}/{len(res.hypothesis_tests)}"
        else:
            ok = res.status == c["expect_status"]
            why = f"status={res.status}"
            if ok and c.get("expect_dq"):
                found = {i.check for i in res.dq_issues}
                missing = [x for x in c["expect_dq"] if x not in found]
                ok = not missing
                why += f", dq={sorted(found)}" + (f", missing {missing}" if missing else "")
        rows.append({"id": c["id"], "type": c["type"], "role": c["role"], "question": c["question"], "pass": ok,
                     "detail": why, "sql_source": res.sql_source or "-", "llm_calls": res.llm_stats.calls,
                     "seconds": round(time.perf_counter() - t0, 2)})
        print(f"{'PASS' if ok else 'FAIL'}  {c['id']:4s} {c['question'][:70]:70s} {why}")

    df = pd.DataFrame(rows)
    summary = df.groupby("type")["pass"].agg(["sum", "count"])
    print("\n", summary.to_string())
    print(f"\nOVERALL {df['pass'].sum()}/{len(df)}  (LLM {'on' if llm.available() else 'off - template fallback'})")
    md = [f"# Evaluation results\n\nLLM: {'Groq' if llm.available() else 'offline (template fallback)'}\n",
          f"**Overall: {df['pass'].sum()}/{len(df)}**\n", summary.rename(columns={'sum': 'passed', 'count': 'total'}).to_markdown(), "\n",
          df.assign(**{"pass": df["pass"].map({True: "✅", False: "❌"})}).to_markdown(index=False)]
    (HERE / "results.md").write_text("\n".join(md))
    return 0 if df["pass"].all() else 1


if __name__ == "__main__":
    sys.exit(main())
