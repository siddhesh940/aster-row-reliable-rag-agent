"""Multi-turn context resolution + session isolation tests."""

from __future__ import annotations


def test_order_followup_resolves_id(agent):
    a1 = agent.handle("Where is ORD-1007?", "mt1")
    assert a1.used_order_tool
    a2 = agent.handle("When will it arrive?", "mt1")
    assert a2.used_order_tool
    assert a2.tool_calls[-1]["arguments"] == {"order_id": "ORD-1007"}
    assert "August 22, 2026" in a2.answer


def test_policy_topic_followup_canada(agent):
    agent.handle("Do you ship internationally?", "mt2")
    a2 = agent.handle("What about Canada, and how long does it take?", "mt2")
    low = a2.answer.lower()
    assert "canada" in low and "5–9 business days after dispatch" in low


def test_policy_exception_followup(agent):
    agent.handle("Can I return an unused jacket within 30 days?", "mt3")
    a2 = agent.handle("What about the exception for damaged items?", "mt3")
    low = a2.answer.lower()
    assert "damaged" in low


def test_session_isolation_separate_sessions_do_not_share_state(agent):
    a = agent.handle("Where is ORD-1007?", "iso-A")
    assert a.used_order_tool
    # Brand-new session must not inherit the order reference.
    b = agent.handle("When will it arrive?", "iso-B")
    assert not b.used_order_tool
    assert "order id" in b.answer.lower()


def test_explicit_new_id_overrides_memory(agent):
    agent.handle("Where is ORD-1007?", "ovr")
    a2 = agent.handle("Where is ORD-1011?", "ovr")
    a3 = agent.handle("When will it arrive?", "ovr")
    assert a3.tool_calls[-1]["arguments"] == {"order_id": "ORD-1011"}
    assert "Canada Post" in a3.answer


def test_ambiguous_without_any_context_asks(agent):
    resp = agent.handle("What about that order?", "amb1")
    assert not resp.used_order_tool


def test_sessions_store_has_no_cross_leak():
    from agent.sessions import SessionStore
    store = SessionStore()
    s1 = store.get("A")
    s2 = store.get("B")
    s1.last_order_id = "ORD-1007"
    s1.add_user("secret context")
    assert s2.last_order_id is None
    assert s2.history == []
