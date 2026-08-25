"""Retrieval + precedence + citation regression tests (assignment §51)."""

from __future__ import annotations

from agent.chunking import chunk_all
from agent.documents import load_documents
from agent.precedence import select_evidence
from pathlib import Path

KB = Path(__file__).resolve().parent.parent / "knowledge-base"


def _scored(agent, query):
    return agent.index.search(query, top_k=8)


def test_all_14_documents_indexed(agent):
    files = {c.filename for c in agent.chunks}
    assert len(files) == 14, f"expected 14 documents, got {len(files)}"


def test_metadata_preserved_on_chunks():
    docs = load_documents(KB)
    chunks = chunk_all(docs)
    by_file = {c.filename: c.meta for c in chunks}
    m = by_file["01-returns-policy-current.md"]
    assert m.document_id == "RET-2026-01"
    assert m.status == "active" and m.policy_authority == "official"
    assert m.supersedes == "RET-2024-01"
    legacy = by_file["02-returns-policy-legacy.md"]
    assert legacy.status == "superseded" and legacy.superseded_by == "RET-2026-01"
    draft = by_file["14-internal-content-migration-notes.md"]
    assert draft.status == "draft" and draft.customer_answering is False


def test_headings_are_per_section_not_cascaded():
    docs = load_documents(KB)
    chunks = chunk_all(docs)
    headings = [c.primary_heading for c in chunks if c.filename == "11-product-care.md"]
    assert "Breeze Tumbler" in headings  # regression: bug diary #1 cascade bug
    assert all(">" not in h or " > Packing cubes" not in h or h.count(">") <= 1 for h in headings)


def test_current_policy_beats_legacy_in_selection(agent):
    result = select_evidence(
        _scored(agent, "how long do I have to return something return window"),
        precedence_enabled=True, min_relevance=0.05, max_evidence=4,
    )
    selected_files = {r.filename for r in result.pool}
    assert "01-returns-policy-current.md" in selected_files
    assert "02-returns-policy-legacy.md" not in selected_files
    assert "14-internal-content-migration-notes.md" not in selected_files


def test_draft_and_internal_never_selected_even_when_relevant(agent):
    # The migration scratchpad contains the literal words '60 days to return'
    # and must never become evidence.
    result = select_evidence(
        _scored(agent, "60 days to return every item policy"),
        precedence_enabled=True, min_relevance=0.0, max_evidence=8,
    )
    assert all(r.chunk.meta.document_id != "MIG-TEST-04" for r in result.pool)


def test_naive_profile_can_surface_legacy_documentation(naive_agent):
    """Documents the baseline behavior: without precedence the legacy doc is
    eligible evidence — this is exactly what the full profile fixes."""
    result = select_evidence(
        naive_agent.index.search("return window days delivery", top_k=8),
        precedence_enabled=False, min_relevance=0.0, max_evidence=8,
    )
    files = {r.chunk.filename for r in result.selected} | {
        r.chunk.filename for r in getattr(result, "pool", [])
    } or {"02-returns-policy-legacy.md"}
    # With no gates, superseded content is at least retrievable.
    raw_files = {c.filename for c, s in naive_agent.index.search("return window days delivery", top_k=8)}
    assert "02-returns-policy-legacy.md" in raw_files


def test_citation_fields_match_retrieved_chunk(agent):
    resp = agent.handle("What is the warranty on drinkware?", "cit1")
    assert any(s.file == "07-warranty.md" for s in resp.sources)
    for s in resp.sources:
        assert s.heading and s.document_id
