import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import memory  # noqa: E402
from core.analysis import merge_joins  # noqa: E402
from core.anomaly import scan  # noqa: E402
from core.critic import deterministic_checks  # noqa: E402
from core.impact import inr_words  # noqa: E402
from core.orchestrator import fact_check  # noqa: E402
from core.agents import Intent  # noqa: E402
from core.timeparse import parse_windows  # noqa: E402


def test_fact_check_respects_units():
    ev = {"business_impact": {"gmv_impact_inr": 1965523.0}}
    ok = lambda s: fact_check({"headline": s}, ev)[0]["verified"]
    assert ok("worth ₹19.66 lakh") and ok("worth ₹19,65,523") and ok("worth ₹1.97 million")
    assert not ok("worth ₹1.97 crore")          # lakh figure mislabelled as crore must fail


def test_inr_words():
    assert inr_words(1965523) == "₹19.66 lakh"
    assert inr_words(123456789) == "₹12.35 crore"


def test_merge_joins_dedupes():
    j = "LEFT JOIN banks b ON b.bank_id = t.bank_id"
    assert merge_joins(j, j + " LEFT JOIN merchants m ON m.merchant_id = t.merchant_id").count("banks b") == 1


def test_critic_rule_checks():
    it = Intent(question="UPI success rate last week", metric="success_rate", windows=parse_windows("last week"),
                filters={"payment_method": "UPI"})
    good = ("SELECT SUM(CASE WHEN t.status='SUCCESS' THEN 1 ELSE 0 END)::DOUBLE / SUM(CASE WHEN t.status IN ('SUCCESS','FAILED') "
            "THEN 1 ELSE 0 END) FROM transactions t WHERE t.payment_method_id='UPI' AND t.txn_date BETWEEN DATE '2026-11-09' AND DATE '2026-11-15'")
    assert deterministic_checks(good, it) == []
    bad = "SELECT AVG(CASE WHEN status='SUCCESS' THEN 1 ELSE 0 END) FROM transactions t WHERE t.txn_date BETWEEN DATE '2026-11-01' AND DATE '2026-11-15'"
    issues = deterministic_checks(bad, it)
    assert any("2026-11-09" in i for i in issues) and any("UPI" in i for i in issues) and any("FAILED" in i for i in issues)


def test_followup_rule_resolution():
    prev = {"question": "Why did success rate drop last week?", "standalone": "Why did success rate drop last week?", "status": "answered",
            "intent": "explain_change", "metric": "success_rate", "windows": [{"start": "2026-11-09", "end": "2026-11-15", "label": "last week"}],
            "filters": {}, "group_by": [], "headline": "x"}
    r = memory._rule_resolve("what about netbanking?", prev)
    assert r["used_context"] and "2026-11-09" in r["question"] and "netbanking" in r["question"]


def test_anomaly_watch_finds_planted_incidents():
    leads = {a["segment"]: a for a in scan("business_leader")["alerts"]}
    assert "Kaveri Bank x UPI" in leads and leads["Kaveri Bank x UPI"]["severity"] == "critical"
    assert any(a["type"] == "volume_drop" and a["segment"] == "East" for a in leads.values())
