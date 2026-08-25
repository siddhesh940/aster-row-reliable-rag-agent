"""Conflict detection + groundedness regression tests."""

from __future__ import annotations


def test_breeze_tumbler_conflict_detected_and_surfaced(agent):
    resp = agent.handle("Can I put the entire Breeze Tumbler in the dishwasher?", "cf1")
    assert resp.conflict_detected is True
    assert resp.handoff is True
    files = {s.file for s in resp.sources}
    assert {"11-product-care.md", "12-breeze-tumbler-product-card.md"} <= files
    low = resp.answer.lower()
    assert "conflict" in low
    assert "hand-wash" in low
    assert "dishwasher safe" in low
    # Must NOT silently pick one side.
    assert not low.startswith("yes") and not low.startswith("no,")


def test_conflict_paraphrase_still_detected(agent):
    resp = agent.handle(
        "Is the Breeze Tumbler lid and body okay in the dishwasher or should I wash it by hand?",
        "cf2",
    )
    assert resp.conflict_detected is True
    assert resp.handoff is True


def test_trailplus_delegation_is_not_a_conflict(agent):
    """30-day standard + 45-day TrailPlus is delegation, not contradiction."""
    resp = agent.handle("What return window do TrailPlus members get?", "cf3")
    assert resp.conflict_detected is False
    assert "45 calendar days" in resp.answer


def test_no_lifetime_warranty_grounded(agent):
    resp = agent.handle("Do all Aster & Row products have a lifetime warranty?", "gr1")
    assert "does not offer a lifetime warranty" in resp.answer.lower()
    assert "2 years" in resp.answer and "1 year" in resp.answer
    assert any(s.file == "07-warranty.md" for s in resp.sources)


def test_unsupported_country_not_invented(agent):
    resp = agent.handle("Can you ship an Atlas Weekender to Germany?", "gr2")
    assert "germany is not currently available" in resp.answer.lower()
    assert "canada" in resp.answer.lower()
    assert resp.handoff is False
