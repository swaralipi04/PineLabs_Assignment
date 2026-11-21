import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import executor  # noqa: E402
from core.rbac import ROLES, precheck_question  # noqa: E402
from core.sql_guard import GuardError, validate  # noqa: E402

LEADER, SOUTH = ROLES["business_leader"], ROLES["regional_analyst_south"]


@pytest.mark.parametrize("sql,kind", [
    ("DROP TABLE transactions", "unsafe"),
    ("DELETE FROM transactions", "unsafe"),
    ("UPDATE transactions SET status='SUCCESS'", "unsafe"),
    ("SELECT 1; DROP TABLE transactions", "unsafe"),
    ("SELECT * FROM read_csv('/etc/passwd')", "unsafe"),
    ("SELECT * FROM main.raw_transactions", "unsafe"),
    ("ATTACH 'x.db' AS x", "unsafe"),
    ("SELECT * FROM transactions a, transactions b", "expensive"),
    ("SELECT * FROM transactions a CROSS JOIN transactions b", "expensive"),
    ("SELECT * FROM secret_table", "invalid"),
])
def test_unsafe_sql_rejected(sql, kind):
    with pytest.raises(GuardError) as e:
        validate(sql, LEADER)
    assert e.value.kind == kind


def test_role_table_and_column_restrictions():
    with pytest.raises(GuardError) as e:
        validate("SELECT SUM(net_amount_inr) FROM settlements", SOUTH)
    assert e.value.kind == "access_denied"
    with pytest.raises(GuardError) as e:
        validate("SELECT contact_email FROM merchants", LEADER)
    assert e.value.kind == "access_denied"
    with pytest.raises(GuardError) as e:
        validate("SELECT COUNT(*) FROM transactions WHERE region = 'North'", SOUTH)
    assert e.value.kind == "access_denied"


def test_valid_query_is_row_capped_and_scoped():
    g = validate("SELECT region, COUNT(*) n FROM transactions GROUP BY 1", SOUTH)
    assert "LIMIT" in g.sql
    df = executor.run("SELECT region, COUNT(*) n FROM transactions GROUP BY 1", SOUTH, use_cache=False).df
    assert set(df["region"]) == {"South"}  # view-level row security


def test_question_precheck():
    assert precheck_question("settlement payouts for all regions", SOUTH)
    assert precheck_question("success rate in the East", SOUTH)
    assert precheck_question("success rate in the East", LEADER) is None
