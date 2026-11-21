"""Reference vocabulary read from the database (never typed into code).

Region, bank, payment-method, category and channel values used by the rule-based parsers,
the SQL guard, the critic and the prompts all come from the dimension tables, so adding a
bank or region to the data needs no code change.
"""
from __future__ import annotations

from functools import lru_cache

import duckdb

from config import DB_PATH

# Natural-language synonyms only (language, not data). A synonym is used only if its target exists in the data.
METHOD_SYNONYMS = {"net banking": "netbanking", "credit": "credit card", "debit": "debit card"}
CHANNEL_SYNONYMS = {"mobile app": "app", "website": "web", "online": "web", "in-store": "pos", "point of sale": "pos"}


@lru_cache(maxsize=1)
def _load() -> dict:
    out = {"regions": [], "banks": {}, "methods": {}, "categories": [], "channels": [], "tiers": [], "statuses": []}
    if not DB_PATH.exists():
        return out
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        q = lambda sql: [r for r in con.execute(sql).fetchall()]
        out["regions"] = [r[0] for r in q("SELECT DISTINCT region FROM merchants WHERE region IS NOT NULL ORDER BY 1")]
        out["banks"] = {r[0]: r[1] for r in q("SELECT bank_id, bank_name FROM banks ORDER BY 1")}
        out["methods"] = {r[0]: r[1] for r in q("SELECT payment_method_id, method_name FROM payment_methods ORDER BY 1")}
        out["categories"] = [r[0] for r in q("SELECT DISTINCT category FROM merchants WHERE category IS NOT NULL ORDER BY 1")]
        out["tiers"] = [r[0] for r in q("SELECT DISTINCT tier FROM merchants WHERE tier IS NOT NULL ORDER BY 1")]
        out["channels"] = [r[0] for r in q("SELECT DISTINCT channel FROM transactions_certified WHERE channel IS NOT NULL ORDER BY 1")]
        out["statuses"] = [r[0] for r in q("SELECT DISTINCT status FROM transactions_certified ORDER BY 1")]
    return out


def regions() -> list[str]: return _load()["regions"]
def banks() -> dict[str, str]: return _load()["banks"]            # bank_id -> name
def bank_names() -> list[str]: return list(banks().values())
def bank_id(name: str) -> str | None: return {v: k for k, v in banks().items()}.get(name)
def methods() -> dict[str, str]: return _load()["methods"]        # payment_method_id -> name
def categories() -> list[str]: return _load()["categories"]
def channels() -> list[str]: return _load()["channels"]
def tiers() -> list[str]: return _load()["tiers"]


def method_words() -> dict[str, str]:
    """phrase (lower case) -> payment_method_id, built from the payment_methods table."""
    words = {}
    for mid, name in methods().items():
        words[name.lower()] = mid
        words[mid.lower()] = mid
    for syn, target in METHOD_SYNONYMS.items():
        if target in words:
            words[syn] = words[target]
    return dict(sorted(words.items(), key=lambda kv: -len(kv[0])))   # longest phrase first


def channel_words() -> dict[str, str]:
    words = {c.lower(): c for c in channels()}
    words.update({s: t for s, t in CHANNEL_SYNONYMS.items() if t in words})
    return dict(sorted(words.items(), key=lambda kv: -len(kv[0])))


def refresh() -> None:
    _load.cache_clear()
