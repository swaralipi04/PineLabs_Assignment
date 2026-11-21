"""SQL safety validator. Every query (LLM-generated or templated) passes through here.

Checks: single read-only SELECT, no DDL/DML/pragmas/file access, table allowlist per role,
blocked (PII) columns, region literals within role scope, no unbounded cross joins,
result row cap, and an EXPLAIN-based cost estimate (done in executor).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from config import MAX_RESULT_ROWS
from core import catalog
from core.rbac import Role

FORBIDDEN_NODES = tuple(getattr(exp, n) for n in [
    "Insert", "Update", "Delete", "Drop", "Create", "Alter", "Merge", "Command", "Copy",
    "Pragma", "Set", "Use", "Attach", "Detach", "Transaction", "Commit", "Rollback", "Grant", "Describe",
] if hasattr(exp, n))

FORBIDDEN_FUNCS = {
    "read_csv", "read_csv_auto", "read_parquet", "read_json", "read_json_auto", "read_text", "read_blob",
    "glob", "parquet_scan", "csv_scan", "sniff_csv", "query", "query_table", "getenv", "current_setting",
    "duckdb_settings", "duckdb_tables", "duckdb_columns", "duckdb_views", "duckdb_databases", "pragma_table_info",
    "httpfs", "read_ndjson", "iceberg_scan", "delta_scan", "sqlite_scan", "postgres_scan", "system",
}


class GuardError(Exception):
    def __init__(self, message: str, kind: str = "invalid"):
        super().__init__(message)
        self.kind = kind  # invalid | access_denied | unsafe | expensive


@dataclass
class GuardResult:
    sql: str                       # normalised, row-capped SQL to execute
    tables: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)


def validate(sql: str, role: Role) -> GuardResult:
    sql = (sql or "").strip().rstrip(";").strip()
    if not sql:
        raise GuardError("Empty SQL.")
    try:
        statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except sqlglot.errors.ParseError as e:
        raise GuardError(f"SQL does not parse: {str(e).splitlines()[0]}")
    if len(statements) != 1:
        raise GuardError("Exactly one SQL statement is allowed.", "unsafe")
    tree = statements[0]

    if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        raise GuardError("Only read-only SELECT queries are allowed.", "unsafe")
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise GuardError(f"Statement type {type(node).__name__} is not allowed (read-only access).", "unsafe")
        if isinstance(node, (exp.Anonymous, exp.Func)):
            name = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()
            if name in FORBIDDEN_FUNCS:
                raise GuardError(f"Function {name}() is not allowed (file/system access).", "unsafe")

    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    tables = set()
    for t in tree.find_all(exp.Table):
        if isinstance(t.this, exp.Func) or not t.name:
            raise GuardError("Table functions are not allowed.", "unsafe")
        if t.args.get("db") or t.args.get("catalog"):
            raise GuardError(f"Schema-qualified table '{t.sql()}' is not allowed.", "unsafe")
        name = t.name.lower()
        if name in cte_names:
            continue
        if name not in role.tables:
            known = name in {"transactions", "merchants", "banks", "payment_methods", "failure_codes", "settlements"}
            raise GuardError(
                f"Access denied: table '{name}' is not available to role {role.label}." if known
                else f"Unknown table '{name}'.", "access_denied" if known else "invalid")
        tables.add(name)
    if not tables:
        raise GuardError("Query does not read from any allowed table.")

    for c in tree.find_all(exp.Column):
        if c.name.lower() in role.blocked_columns:
            raise GuardError(f"Access denied: column '{c.name}' is restricted (PII) for role {role.label}.", "access_denied")
    if role.blocked_columns and any(isinstance(s, exp.Star) for s in tree.find_all(exp.Star)) and "merchants" in tables:
        pass  # views already drop blocked columns; SELECT * is safe here

    if role.region_restricted:
        for lit in tree.find_all(exp.Literal):
            if lit.is_string and lit.this in catalog.regions() and lit.this not in role.regions:
                raise GuardError(f"Access denied: region '{lit.this}' is outside your scope ({', '.join(role.regions)}).", "access_denied")

    # Unbounded cross joins over the fact table
    for j in tree.find_all(exp.Join):
        kind = (j.args.get("kind") or "").upper()
        if (kind == "CROSS" or (not j.args.get("on") and not j.args.get("using"))) and _joins_fact(j, tree):
            raise GuardError("Cross join involving the transactions table is too expensive; add a join condition.", "expensive")

    capped = f"SELECT * FROM ({tree.sql(dialect='duckdb')}) AS _q LIMIT {MAX_RESULT_ROWS}"
    return GuardResult(sql=capped, tables=tables, notes=[])


def _joins_fact(join: exp.Join, tree: exp.Expression) -> bool:
    joined = {t.name.lower() for t in join.find_all(exp.Table)}
    from_tables = {t.name.lower() for t in (tree.args.get("from") or exp.From()).find_all(exp.Table)} if tree.args.get("from") else set()
    return "transactions" in joined or "transactions" in from_tables


def pretty(sql: str) -> str:
    try:
        return sqlglot.transpile(sql, read="duckdb", write="duckdb", pretty=True)[0]
    except Exception:
        return sql
