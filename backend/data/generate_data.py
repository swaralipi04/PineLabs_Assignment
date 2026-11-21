"""Generate the synthetic payments warehouse (DuckDB).

Everything is fictional. Scenarios are planted deliberately so the system's reasoning can be
checked against known answers:

  S1  Real decline     : Kaveri Bank UPI technical failures (BANK_TIMEOUT), 2026-11-09 .. 2026-11-15
  S2  Mix shift        : netbanking cashback campaign, 2026-10-12 .. 2026-10-18. Netbanking share
                         jumps; every method's own success rate is unchanged, yet overall SR falls.
  S3  Duplicates       : ~0.5% duplicated txn_ids in the raw feed, ~4% on 2026-10-05 .. 10-07 (replay bug)
  S4  Missing values   : bank_id NULL on ~1.5% of rows (higher in November)
  S5  Ingestion gap    : no East-region rows on 2026-10-27
  S6  Source conflict  : settlements.txn_count disagrees with transactions for some merchant-days,
                         and one settlement batch (2026-11-10) is partly missing

Run:  python data/generate_data.py
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import GEN_END as DATA_END, GEN_START as DATA_START, DB_PATH  # noqa: E402
from core.rbac import build_role_views  # noqa: E402

RNG = np.random.default_rng(42)
BASE_DAILY_TXNS = 5500

S1_START, S1_END = date(2026, 11, 9), date(2026, 11, 15)
S2_START, S2_END = date(2026, 10, 12), date(2026, 10, 18)
S3_START, S3_END = date(2026, 10, 5), date(2026, 10, 7)
S5_DAY = date(2026, 10, 27)
S6_MISSING_BATCH = date(2026, 11, 10)

# ----------------------------------------------------------------------------- dimensions
BANKS = [  # bank_id, name, tier, volume weight, success-rate modifier
    ("B01", "Kaveri Bank", "Tier 1", 0.18, 0.005),
    ("B02", "Indus Federal Bank", "Tier 1", 0.16, 0.004),
    ("B03", "Sahyadri Bank", "Tier 1", 0.13, 0.002),
    ("B04", "Narmada National Bank", "Tier 1", 0.11, 0.000),
    ("B05", "Deccan Trust Bank", "Tier 2", 0.08, -0.004),
    ("B06", "Konkan Capital Bank", "Tier 2", 0.07, -0.002),
    ("B07", "Himalaya Bank", "Tier 2", 0.06, 0.001),
    ("B08", "Coromandel Bank", "Tier 2", 0.06, -0.003),
    ("B09", "Ganga Co-operative Bank", "Tier 3", 0.05, -0.012),
    ("B10", "Vindhya Bank", "Tier 3", 0.04, -0.008),
    ("B11", "Malabar Bank", "Tier 3", 0.03, -0.006),
    ("B12", "Thar Rural Bank", "Tier 3", 0.03, -0.015),
]

METHODS = [  # id, name, family, base share, base success rate
    ("UPI", "UPI", "Real-time", 0.45, 0.945),
    ("CC", "Credit Card", "Card", 0.15, 0.925),
    ("DC", "Debit Card", "Card", 0.15, 0.905),
    ("NB", "Netbanking", "Bank redirect", 0.12, 0.855),
    ("WALLET", "Wallet", "Stored value", 0.13, 0.960),
]

FAILURE_CODES = [  # code, description, category
    ("BANK_TIMEOUT", "Issuer bank did not respond within SLA", "TECHNICAL"),
    ("BANK_DOWN", "Issuer bank unavailable", "TECHNICAL"),
    ("NETWORK_ERROR", "Network / switch error", "TECHNICAL"),
    ("INSUFFICIENT_FUNDS", "Customer has insufficient balance", "BUSINESS"),
    ("LIMIT_EXCEEDED", "Daily or per-transaction limit exceeded", "BUSINESS"),
    ("RISK_DECLINED", "Declined by risk engine", "BUSINESS"),
    ("USER_DROPPED", "Customer abandoned the payment", "USER"),
    ("WRONG_PIN_OTP", "Incorrect PIN / OTP entered", "USER"),
]
FAIL_MIX = np.array([0.12, 0.06, 0.08, 0.18, 0.08, 0.10, 0.25, 0.13])

CATEGORIES = {  # category: (amount lognormal mu, sigma, mdr)
    "Ecommerce": (7.0, 0.9, 1.8), "Travel": (8.6, 0.8, 1.6), "Food Delivery": (6.0, 0.5, 2.0),
    "Utilities": (7.2, 0.6, 0.9), "Gaming": (5.8, 0.9, 2.4), "Education": (8.8, 0.7, 1.2),
    "Retail": (6.8, 0.8, 1.5), "Healthcare": (7.4, 0.8, 1.3),
}
REGIONS = ["North", "South", "East", "West"]
REGION_W = [0.28, 0.30, 0.17, 0.25]
WORDS_A = ["Blue", "Swift", "Prime", "Urban", "Royal", "Green", "Bright", "Metro", "Sun", "Lotus", "Peak", "Silver", "Coastal", "Golden", "Nova"]
WORDS_B = ["Mart", "Kart", "Travels", "Eats", "Power", "Learn", "Care", "Play", "Bazaar", "Pharma", "Stays", "Fresh", "Hub", "Store", "Works"]


def make_merchants(n=300) -> pd.DataFrame:
    cats = list(CATEGORIES)
    rows = []
    used = set()
    for i in range(n):
        while True:
            name = f"{RNG.choice(WORDS_A)} {RNG.choice(WORDS_B)}"
            name = name if name not in used else f"{name} {RNG.integers(2, 99)}"
            if name not in used:
                used.add(name)
                break
        cat = cats[RNG.integers(len(cats))]
        tier = RNG.choice(["Enterprise", "Mid-market", "SMB"], p=[0.1, 0.3, 0.6])
        rows.append({
            "merchant_id": f"M{i + 1:04d}",
            "merchant_name": name,
            "category": cat,
            "tier": tier,
            "region": RNG.choice(REGIONS, p=REGION_W),
            "onboarded_on": date(2022, 1, 1) + timedelta(days=int(RNG.integers(0, 1500))),
            "mdr_pct": round(CATEGORIES[cat][2] + RNG.normal(0, 0.15), 2),
            "contact_email": f"finance@{name.lower().replace(' ', '')}.example",
        })
    df = pd.DataFrame(rows)
    # Pareto-ish volume weights, enterprise merchants get more traffic
    w = RNG.pareto(1.3, n) + 0.2
    w *= df["tier"].map({"Enterprise": 4.0, "Mid-market": 1.8, "SMB": 1.0}).to_numpy()
    df["_w"] = w / w.sum()
    return df


def make_transactions(merchants: pd.DataFrame) -> pd.DataFrame:
    days = pd.date_range(DATA_START, DATA_END, freq="D")
    frames = []
    bank_ids = np.array([b[0] for b in BANKS])
    bank_w = np.array([b[3] for b in BANKS]); bank_w /= bank_w.sum()
    bank_mod = {b[0]: b[4] for b in BANKS}
    method_ids = np.array([m[0] for m in METHODS])
    base_share = np.array([m[3] for m in METHODS])
    method_sr = {m[0]: m[4] for m in METHODS}
    codes = np.array([f[0] for f in FAILURE_CODES])
    txn_counter = 0

    for d in days:
        dd = d.date()
        factor = {5: 1.18, 6: 1.10}.get(d.weekday(), 1.0)
        if date(2026, 10, 5) <= dd <= date(2026, 10, 25):  # festive season uplift
            factor *= 1.25
        n = int(RNG.poisson(BASE_DAILY_TXNS * factor))

        m_idx = RNG.choice(len(merchants), size=n, p=merchants["_w"].to_numpy())
        m = merchants.iloc[m_idx]
        share = base_share.copy()
        if S2_START <= dd <= S2_END:  # S2: netbanking cashback campaign -> mix shift
            share = np.array([0.36, 0.12, 0.12, 0.30, 0.10])
        share /= share.sum()
        method = RNG.choice(method_ids, size=n, p=share)
        bank = RNG.choice(bank_ids, size=n, p=bank_w)
        channel = RNG.choice(["app", "web", "pos"], size=n, p=[0.48, 0.42, 0.10])

        sr = np.vectorize(method_sr.get)(method) + np.vectorize(bank_mod.get)(bank)
        sr += np.where(channel == "app", 0.006, np.where(channel == "web", -0.006, 0.0))
        sr += RNG.normal(0, 0.004, n)
        incident = np.zeros(n, dtype=bool)
        if S1_START <= dd <= S1_END:  # S1: Kaveri Bank UPI degradation
            incident = (bank == "B01") & (method == "UPI")
            sr = np.where(incident, 0.64, sr)
        sr = np.clip(sr, 0, 1)

        u = RNG.random(n)
        status = np.where(u < 0.01, "PENDING", np.where(RNG.random(n) < sr, "SUCCESS", "FAILED"))
        fail_code = RNG.choice(codes, size=n, p=FAIL_MIX / FAIL_MIX.sum())
        # incident failures are overwhelmingly issuer timeouts
        fail_code = np.where(incident & (RNG.random(n) < 0.85), "BANK_TIMEOUT", fail_code)
        fail_code = np.where(status == "FAILED", fail_code, None)

        mu = m["category"].map(lambda c: CATEGORIES[c][0]).to_numpy()
        sg = m["category"].map(lambda c: CATEGORIES[c][1]).to_numpy()
        amount = np.round(np.exp(RNG.normal(mu, sg)), 2)
        secs = np.sort(RNG.integers(0, 86400, n))
        ts = pd.Timestamp(dd) + pd.to_timedelta(secs, unit="s")
        resp = np.round(np.exp(RNG.normal(6.4, 0.45, n)))
        resp = np.where(incident, resp * 4.5, resp)
        card_net = np.where(np.isin(method, ["CC", "DC"]), RNG.choice(["VISA", "MASTERCARD", "RUPAY"], n, p=[0.45, 0.35, 0.20]), None)

        df = pd.DataFrame({
            "txn_id": [f"T{txn_counter + i:08d}" for i in range(n)],
            "txn_ts": ts,
            "txn_date": dd,
            "merchant_id": m["merchant_id"].to_numpy(),
            "bank_id": bank,
            "payment_method_id": method,
            "card_network": card_net,
            "channel": channel,
            "region": m["region"].to_numpy(),
            "amount_inr": amount,
            "status": status,
            "failure_code": fail_code,
            "response_ms": resp.astype(int),
        })
        txn_counter += n
        if dd == S5_DAY:  # S5: East ingestion gap
            df = df[df["region"] != "East"]
        frames.append(df)

    tx = pd.concat(frames, ignore_index=True)
    tx["ingested_at"] = tx["txn_ts"] + pd.to_timedelta(RNG.integers(30, 900, len(tx)), unit="s")

    # S4: missing bank_id (1% baseline, 3% in November)
    nov = tx["txn_date"] >= date(2026, 11, 1)
    null_mask = RNG.random(len(tx)) < np.where(nov, 0.03, 0.01)
    tx.loc[null_mask, "bank_id"] = None

    # S3: duplicates (re-ingested copies with a later ingested_at)
    replay = (tx["txn_date"] >= S3_START) & (tx["txn_date"] <= S3_END)
    dup_mask = RNG.random(len(tx)) < np.where(replay, 0.04, 0.004)
    dups = tx[dup_mask].copy()
    dups["ingested_at"] = dups["ingested_at"] + pd.Timedelta(hours=6)
    tx = pd.concat([tx, dups], ignore_index=True).sort_values("txn_ts", kind="stable").reset_index(drop=True)
    return tx


def make_settlements(certified: pd.DataFrame, merchants: pd.DataFrame) -> pd.DataFrame:
    ok = certified[certified["status"] == "SUCCESS"]
    s = ok.groupby(["merchant_id", "txn_date"]).agg(txn_count=("txn_id", "count"), gross_amount_inr=("amount_inr", "sum")).reset_index()
    s = s.merge(merchants[["merchant_id", "mdr_pct"]], on="merchant_id")
    # S6: settlement system disagrees on ~3% of merchant-days (late reversals / batching issues)
    bad = RNG.random(len(s)) < 0.03
    delta = np.maximum(1, np.round(s["txn_count"] * RNG.uniform(0.03, 0.12, len(s)))).astype(int)
    sign = np.where(RNG.random(len(s)) < 0.7, -1, 1)
    adj = np.where(bad, sign * delta, 0)
    avg = s["gross_amount_inr"] / s["txn_count"]
    s["txn_count"] = (s["txn_count"] + adj).clip(lower=0)
    s["gross_amount_inr"] = (s["gross_amount_inr"] + adj * avg).round(2)
    s["fees_inr"] = (s["gross_amount_inr"] * s["mdr_pct"] / 100).round(2)
    s["net_amount_inr"] = (s["gross_amount_inr"] - s["fees_inr"]).round(2)
    s["settle_date"] = pd.to_datetime(s["txn_date"]) + pd.Timedelta(days=1)
    s["settle_date"] = s["settle_date"].dt.date
    s["settlement_status"] = np.where(RNG.random(len(s)) < 0.02, "ON_HOLD", "SETTLED")
    # missing part of one batch
    batch = s["txn_date"] == S6_MISSING_BATCH
    drop_ids = s[batch].sample(frac=0.25, random_state=7).index
    s = s.drop(drop_ids)
    s.insert(0, "settlement_id", [f"S{i:07d}" for i in range(len(s))])
    return s[["settlement_id", "merchant_id", "txn_date", "settle_date", "txn_count", "gross_amount_inr", "fees_inr", "net_amount_inr", "settlement_status"]]


def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    print("Generating merchants ...")
    merchants = make_merchants()
    print("Generating transactions ...")
    tx = make_transactions(merchants)

    con = duckdb.connect(str(DB_PATH))
    con.register("tx_df", tx)
    con.execute("CREATE TABLE raw_transactions AS SELECT * FROM tx_df")
    # Certified layer: one row per txn_id (first ingested copy wins)
    con.execute("""
        CREATE TABLE transactions_certified AS
        SELECT * EXCLUDE (rn) FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY txn_id ORDER BY ingested_at) AS rn FROM raw_transactions
        ) WHERE rn = 1
    """)
    certified = con.execute("SELECT * FROM transactions_certified").df()
    print("Generating settlements ...")
    settlements = make_settlements(certified, merchants)

    con.register("m_df", merchants.drop(columns="_w"))
    con.execute("CREATE TABLE merchants AS SELECT * FROM m_df")
    con.register("s_df", settlements)
    con.execute("CREATE TABLE settlements AS SELECT * FROM s_df")
    con.execute("CREATE TABLE banks AS SELECT * FROM (VALUES " + ",".join(f"('{b[0]}','{b[1]}','{b[2]}','Issuer')" for b in BANKS) + ") t(bank_id, bank_name, bank_tier, bank_type)")
    con.execute("CREATE TABLE payment_methods AS SELECT * FROM (VALUES " + ",".join(f"('{m[0]}','{m[1]}','{m[2]}')" for m in METHODS) + ") t(payment_method_id, method_name, method_family)")
    con.execute("CREATE TABLE failure_codes AS SELECT * FROM (VALUES " + ",".join(f"('{f[0]}','{f[1]}','{f[2]}')" for f in FAILURE_CODES) + ") t(failure_code, description, failure_category)")
    build_role_views(con)

    for t in ["raw_transactions", "transactions_certified", "merchants", "settlements", "banks", "payment_methods", "failure_codes"]:
        print(f"  {t:24s} {con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]:>9,}")
    con.close()
    print(f"Wrote {DB_PATH}")


if __name__ == "__main__":
    main()
