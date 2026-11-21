"""Role-based access control.

Enforcement happens in three layers (defence in depth):
1. Question pre-check  - refuses questions that explicitly ask for out-of-scope data.
2. SQL guard           - rejects SQL touching tables/columns/region values the role may not see.
3. Database views      - each role queries its own schema of views (row + column filtered),
                         so even a guard bug cannot leak rows outside the role's scope.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from core import catalog

PII_COLUMNS = {"contact_email"}


@dataclass(frozen=True)
class Role:
    key: str
    label: str
    schema: str
    tables: frozenset[str]
    region_scope: tuple[str, ...] | None = None    # access policy; None = every region in the data
    blocked_columns: frozenset[str] = field(default_factory=frozenset)
    description: str = ""

    @property
    def regions(self) -> tuple[str, ...]:
        return self.region_scope if self.region_scope is not None else tuple(catalog.regions())

    @property
    def region_restricted(self) -> bool:
        return self.region_scope is not None


_BASE_TABLES = {"transactions", "merchants", "banks", "payment_methods", "failure_codes"}

ROLES: dict[str, Role] = {
    "admin": Role(
        key="admin",
        label="Data Admin",
        schema="r_admin",
        tables=frozenset(_BASE_TABLES | {"settlements"}),
        description="Full access to all tables and columns, including merchant contact details.",
    ),
    "business_leader": Role(
        key="business_leader",
        label="Business Leader (National)",
        schema="r_leader",
        tables=frozenset(_BASE_TABLES | {"settlements"}),
        blocked_columns=frozenset(PII_COLUMNS),
        description="All regions and settlements. No merchant PII.",
    ),
    "regional_analyst_south": Role(
        key="regional_analyst_south",
        label="Regional Analyst (South)",
        schema="r_south",
        tables=frozenset(_BASE_TABLES),
        region_scope=("South",),
        blocked_columns=frozenset(PII_COLUMNS),
        description="South region transactions and merchants only. No settlements / money movement, no PII.",
    ),
}

MERCHANT_COLUMNS = ["merchant_id", "merchant_name", "category", "tier", "region", "onboarded_on", "mdr_pct", "contact_email"]


def build_role_views(con) -> None:
    """Create one schema of views per role. Called by the data generator."""
    for role in ROLES.values():
        con.execute(f"DROP SCHEMA IF EXISTS {role.schema} CASCADE")
        con.execute(f"CREATE SCHEMA {role.schema}")
        region_list = ", ".join(f"'{r}'" for r in role.regions)
        region_filter = f"WHERE region IN ({region_list})" if role.region_restricted else ""
        merch_cols = ", ".join(c for c in MERCHANT_COLUMNS if c not in role.blocked_columns)

        con.execute(f"CREATE VIEW {role.schema}.transactions AS SELECT * FROM main.transactions_certified {region_filter}")
        con.execute(f"CREATE VIEW {role.schema}.merchants AS SELECT {merch_cols} FROM main.merchants {region_filter}")
        for t in ("banks", "payment_methods", "failure_codes"):
            con.execute(f"CREATE VIEW {role.schema}.{t} AS SELECT * FROM main.{t}")
        if "settlements" in role.tables:
            con.execute(f"CREATE VIEW {role.schema}.settlements AS SELECT * FROM main.settlements")


# --------------------------------------------------------------------------- question pre-check
_OTHER_SCOPE_WORDS = re.compile(r"\b(all regions|every region|nationwide|national|pan[- ]india|all of india|country[- ]wide|across regions|by region|region[- ]wise|each region)\b", re.I)
_MONEY_WORDS = re.compile(r"\b(settle\w*|payout\w*|net amount|fees?|mdr)\b", re.I)
_PII_WORDS = re.compile(r"\b(e-?mail|contact|phone)\b", re.I)


def precheck_question(question: str, role: Role) -> str | None:
    """Return a denial reason if the question clearly asks for data outside the role's scope."""
    q = question.lower()
    if role.region_restricted:
        others = [r for r in catalog.regions() if r not in role.regions and re.search(rf"\b{r.lower()}\b", q)]
        if others:
            return (f"Your role ({role.label}) can only see data for {', '.join(role.regions)}. "
                    f"The question asks about {', '.join(others)}.")
        if _OTHER_SCOPE_WORDS.search(question):
            return (f"Your role ({role.label}) is limited to {', '.join(role.regions)}; "
                    "cross-region or national figures are outside your access scope.")
    if "settlements" not in role.tables and _MONEY_WORDS.search(question):
        return f"Your role ({role.label}) does not have access to settlement / payout data."
    if role.blocked_columns & PII_COLUMNS and _PII_WORDS.search(question) and "merchant" in q:
        return f"Merchant contact details are PII and are not available to the {role.label} role."
    return None
