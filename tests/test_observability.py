"""Observability + response contract tests."""

from __future__ import annotations

import json


def test_response_contract_shape(agent):
    resp = agent.handle("What is the return policy?", "ob1")
    assert isinstance(resp.answer, str) and resp.answer
    assert isinstance(resp.handoff, bool)
    assert resp.sources == [] or all(
        {"file", "heading", "document_id"} <= s.to_dict().keys() for s in resp.sources)
    assert resp.session_id == "ob1"


def test_order_answer_marks_tool_source_not_kb(agent):
    resp = agent.handle("Where is ORD-1003?", "ob2")
    assert resp.used_order_tool
    assert resp.sources == []  # order facts must not pretend to be KB citations


def test_debug_trace_contains_pipeline_stages(agent):
    resp = agent.handle("Where is ORD-1007 and when should it arrive?", "ob3")
    d = resp.debug
    assert d["user_message"]
    assert d["resolved_query"]
    assert "tool_calls" in str(d) or resp.tool_calls


def test_retrieval_trace_has_scores_and_metadata(agent):
    resp = agent.handle("Do you ship internationally?", "ob4")
    cands = resp.debug["retrieval"]["candidates"]
    assert cands, "retrieval trace missing"
    top = cands[0]
    for key in ("file", "heading", "doc_id", "status", "authority", "audience", "score"):
        assert key in top
    assert any(c["selected"] for c in cands)


def test_error_fallback_and_handoff_traced():
    from agent.agent import SupportAgent
    from agent.config import Config
    agent = SupportAgent(Config(profile="full"))
    # Force an internal error by breaking the order tool path.
    original = agent.orders.lookup

    def boom(_):
        raise RuntimeError("simulated outage")
    agent.orders.lookup = boom  # type: ignore[method-assign]
    try:
        resp = agent.handle("Where is ORD-1002?", "ob5")
    finally:
        agent.orders.lookup = original  # type: ignore[method-assign]
    # Fail closed: no invented status, handoff offered.
    low = resp.answer.lower()
    assert ("wrong" in low) or ("try again" in low) or ("contact" in low)
    assert resp.handoff is True or resp.abstained


def test_redact_scrubs_sensitive_structures():
    from agent.redaction import redact
    dirty = {
        "customer": {"name": "X", "email": "x@y.test"},
        "internal": {"risk_score": 82},
        "note": "mail me at x@y.test",
        "safe": 1,
    }
    out = redact(dirty)
    assert out["customer"] == "[redacted]"
    assert out["internal"] == "[redacted]"
    assert "[redacted-email]" in out["note"]
    assert out["safe"] == 1
