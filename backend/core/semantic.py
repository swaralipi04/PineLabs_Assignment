"""Semantic layer: business definitions of tables, metrics and dimensions.

This is the single source of truth the LLM is grounded on, and the place where
denominators are fixed so "success rate" means the same thing in every answer.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.rbac import Role

TABLES = {
    "transactions": {
        "desc": "Certified payment attempts, one row per txn_id (duplicates removed upstream). Grain: one payment attempt.",
        "columns": {
            "txn_id": "Unique payment attempt id",
            "txn_ts": "Attempt timestamp (TIMESTAMP)",
            "txn_date": "Attempt date (DATE) - use for time filters",
            "merchant_id": "FK -> merchants.merchant_id",
            "bank_id": "Issuer bank, FK -> banks.bank_id. Can be NULL (missing in source feed)",
            "payment_method_id": "FK -> payment_methods.payment_method_id",
            "card_network": "VISA / MASTERCARD / RUPAY for cards, else NULL",
            "channel": "Checkout channel",
            "region": "Merchant region",
            "amount_inr": "Attempt amount in INR",
            "status": "SUCCESS, FAILED or PENDING",
            "failure_code": "FK -> failure_codes.failure_code, only when status='FAILED'",
            "response_ms": "Gateway round-trip latency in ms",
        },
    },
    "merchants": {
        "desc": "Merchant dimension.",
        "columns": {
            "merchant_id": "PK", "merchant_name": "Name", "category": "Merchant category",
            "tier": "Merchant tier", "region": "Merchant region", "onboarded_on": "DATE",
            "mdr_pct": "Merchant discount rate %", "contact_email": "PII - restricted",
        },
    },
    "banks": {"desc": "Issuer bank dimension.", "columns": {"bank_id": "PK", "bank_name": "Name", "bank_tier": "Tier 1/2/3", "bank_type": "Issuer"}},
    "payment_methods": {"desc": "Payment method dimension.", "columns": {"payment_method_id": "PK", "method_name": "Display name", "method_family": "Real-time, Card, Bank redirect, Stored value"}},
    "failure_codes": {"desc": "Failure reason lookup.", "columns": {"failure_code": "PK", "description": "Text", "failure_category": "TECHNICAL, BUSINESS, USER"}},
    "settlements": {
        "desc": "Daily merchant settlements from the settlement system (a SEPARATE source; may disagree with transactions). T+1.",
        "columns": {
            "settlement_id": "PK", "merchant_id": "FK -> merchants", "txn_date": "Business date of settled transactions",
            "settle_date": "Date funds were settled", "txn_count": "Successful txns settled (per settlement system)",
            "gross_amount_inr": "Gross settled amount", "fees_inr": "MDR fees", "net_amount_inr": "Net paid to merchant",
            "settlement_status": "SETTLED / ON_HOLD",
        },
    },
}


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    sql: str            # aggregate expression over transactions `t`
    fmt: str            # pct | int | inr | ms
    definition: str
    higher_is_better: bool = True


METRICS: dict[str, Metric] = {m.key: m for m in [
    Metric("success_rate", "Success rate",
           "SUM(CASE WHEN t.status='SUCCESS' THEN 1 ELSE 0 END)::DOUBLE / NULLIF(SUM(CASE WHEN t.status IN ('SUCCESS','FAILED') THEN 1 ELSE 0 END),0)",
           "pct", "Successful / (successful + failed) attempts. PENDING excluded from the denominator (not yet resolved)."),
    Metric("success_rate_all_attempts", "Success rate (all attempts)",
           "SUM(CASE WHEN t.status='SUCCESS' THEN 1 ELSE 0 END)::DOUBLE / NULLIF(COUNT(*),0)",
           "pct", "Successful / all attempts including PENDING. Alternative denominator."),
    Metric("failure_rate", "Failure rate",
           "SUM(CASE WHEN t.status='FAILED' THEN 1 ELSE 0 END)::DOUBLE / NULLIF(SUM(CASE WHEN t.status IN ('SUCCESS','FAILED') THEN 1 ELSE 0 END),0)",
           "pct", "Failed / (successful + failed).", higher_is_better=False),
    Metric("attempts", "Attempts", "COUNT(*)", "int", "Number of payment attempts."),
    Metric("successful_txns", "Successful transactions", "SUM(CASE WHEN t.status='SUCCESS' THEN 1 ELSE 0 END)", "int", "Count of SUCCESS attempts."),
    Metric("gmv", "GMV (INR)", "SUM(CASE WHEN t.status='SUCCESS' THEN t.amount_inr ELSE 0 END)", "inr", "Sum of amount_inr over successful attempts."),
    Metric("avg_ticket", "Average ticket (INR)", "AVG(CASE WHEN t.status='SUCCESS' THEN t.amount_inr END)", "inr", "Mean successful amount."),
    Metric("avg_latency", "Avg latency (ms)", "AVG(t.response_ms)", "ms", "Mean gateway response time.", higher_is_better=False),
]}

# Dimensions available for drill-down: key -> (label, SQL expression, join clause)
DIMENSIONS = {
    "payment_method": ("Payment method", "pm.method_name", "LEFT JOIN payment_methods pm ON pm.payment_method_id = t.payment_method_id"),
    "bank": ("Issuer bank", "COALESCE(b.bank_name, 'Unknown (missing bank_id)')", "LEFT JOIN banks b ON b.bank_id = t.bank_id"),
    "channel": ("Channel", "t.channel", ""),
    "region": ("Region", "t.region", ""),
    "merchant_category": ("Merchant category", "m.category", "LEFT JOIN merchants m ON m.merchant_id = t.merchant_id"),
    "merchant_tier": ("Merchant tier", "m.tier", "LEFT JOIN merchants m ON m.merchant_id = t.merchant_id"),
    "card_network": ("Card network", "COALESCE(t.card_network, 'Non-card')", ""),
    "failure_category": ("Failure category", "COALESCE(fc.failure_category, 'N/A')", "LEFT JOIN failure_codes fc ON fc.failure_code = t.failure_code"),
}
DRIVER_DIMENSIONS = ["payment_method", "bank", "channel", "region", "merchant_category", "merchant_tier"]

BUSINESS_RULES = [
    "Success rate = SUCCESS / (SUCCESS + FAILED); PENDING is excluded from the denominator unless the user asks for 'all attempts'.",
    "Always filter time on transactions.txn_date (DATE). Use inclusive BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'.",
    "Join banks via bank_id (LEFT JOIN - bank_id may be NULL), payment_methods via payment_method_id, merchants via merchant_id.",
    "Counts of transactions come from transactions (system of record for counts); money paid to merchants comes from settlements.",
    "Present rates as fractions (0-1); the UI formats them as percentages.",
]


def schema_prompt(role: Role, tables: list[str] | None = None) -> str:
    """Render the schema the role is allowed to see, for grounding the SQL agent."""
    out = []
    for name, spec in TABLES.items():
        if name not in role.tables or (tables and name not in tables):
            continue
        cols = "\n".join(f"    - {c}: {d}" for c, d in spec["columns"].items() if c not in role.blocked_columns)
        out.append(f"TABLE {name}: {spec['desc']}\n{cols}")
    out.append(_values_prompt(role))
    return "\n\n".join(out)


def _values_prompt(role: Role) -> str:
    """Valid filter values, read live from the dimension tables (core.catalog) and scoped to the role."""
    from core import catalog
    methods = ", ".join(f"{k} = {v}" for k, v in catalog.methods().items())
    banks = ", ".join(f"{k} = {v}" for k, v in catalog.banks().items())
    return ("VALID VALUES (from the data):\n"
            f"    - payment_method_id: {methods}\n"
            f"    - bank_id: {banks}\n"
            f"    - region: {', '.join(role.regions)}\n"
            f"    - merchant category: {', '.join(catalog.categories())}\n"
            f"    - merchant tier: {', '.join(catalog.tiers())}\n"
            f"    - channel: {', '.join(catalog.channels())}")


def metrics_prompt() -> str:
    return "\n".join(f"- {m.key} ({m.label}): {m.definition}  SQL: {m.sql}" for m in METRICS.values())


def fmt_value(v, fmt: str) -> str:
    if v is None:
        return "n/a"
    try:
        if fmt == "pct":
            return f"{float(v) * 100:.2f}%"
        if fmt == "int":
            return f"{int(round(float(v))):,}"
        if fmt == "inr":
            return f"₹{float(v):,.0f}"
        if fmt == "ms":
            return f"{float(v):,.0f} ms"
    except (TypeError, ValueError):
        pass
    return str(v)
