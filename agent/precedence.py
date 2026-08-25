"""Retrieval + metadata-aware precedence.

Two explicitly separate stages (assignment requirement 40):

1. ``retrieve``      — pure semantic/lexical relevance from the vector index.
2. ``select_evidence`` — authority/status/supersession analysis that decides
   which candidates may actually answer a customer question.

A superseded or draft document can be *relevant* and still be excluded from
*authority*. Recency is deliberately a tiny tiebreaker only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .contracts import Chunk, DocumentMeta, RetrievedChunk
from .indexing import VectorIndex

RECENCY_BONUS_MAX = 0.10  # never decisive


@dataclass
class RetrievalResult:
    selected: list[RetrievedChunk] = field(default_factory=list)
    # Everything that passed the authority gates, sorted by final_score.
    # Composers may pull additional passages of a needed document from here
    # without weakening precedence (rejected material never appears).
    pool: list[RetrievedChunk] = field(default_factory=list)
    rejected: list[tuple[Chunk, str]] = field(default_factory=list)  # (chunk, why)
    best_relevance: float = 0.0
    insufficient: bool = False
    insufficiency_reason: str | None = None


def _recency_bonus(meta: DocumentMeta) -> float:
    if not meta.effective_date:
        return 0.0
    try:
        year, month, day = (int(p) for p in str(meta.effective_date).split("-"))
        # Map 2024-2027 onto 0..1 then scale to the small bonus budget.
        frac = ((year - 2024) * 372 + (month - 1) * 31 + day) / (4 * 372)
        return max(0.0, min(1.0, frac)) * RECENCY_BONUS_MAX
    except (ValueError, TypeError):
        return 0.0


def select_evidence(
    scored: list[tuple[Chunk, float]],
    *,
    precedence_enabled: bool,
    min_relevance: float,
    max_evidence: int,
) -> RetrievalResult:
    result = RetrievalResult()
    if not scored:
        result.insufficient = True
        result.insufficiency_reason = "no retrieval candidates"
        return result

    result.best_relevance = max(s for _, s in scored)

    # Stage 2a: hard gates — documents that may never answer customers.
    eligible: list[RetrievedChunk] = []
    for chunk, rel in scored:
        if rel < min_relevance:
            result.rejected.append((chunk, f"relevance {rel:.3f} below threshold"))
            continue
        meta = chunk.meta
        if precedence_enabled:
            if meta.customer_answering is False:
                result.rejected.append((chunk, "front matter: customer_answering=false"))
                continue
            if meta.status in ("draft", "archived", "retired"):
                result.rejected.append((chunk, f"status={meta.status} is not authoritative"))
                continue
            if meta.status == "superseded":
                result.rejected.append((chunk, "superseded policy cannot answer current questions"))
                continue
            boost = meta.precedence_rank * 0.25 + _recency_bonus(meta)
        else:
            boost = 0.0
        eligible.append(RetrievedChunk(chunk=chunk, relevance=rel, final_score=rel + boost))

    # Stage 2b: supersession shadowing — if an active document supersedes a
    # candidate that somehow got here, drop the candidate.
    if precedence_enabled:
        active_ids = {
            r.chunk.meta.document_id
            for r in eligible
            if r.chunk.meta.status == "active"
        }
        superseding_targets = set()
        for r in eligible:
            sup = r.chunk.meta.supersedes
            if r.chunk.meta.status == "active" and sup:
                superseding_targets.add(sup.strip())
        kept: list[RetrievedChunk] = []
        for r in eligible:
            m = r.chunk.meta
            if m.status == "superseded" or (
                m.superseded_by and m.superseded_by.strip() in active_ids
            ):
                result.rejected.append((r.chunk, "superseded by an active document in evidence"))
                continue
            if m.document_id and m.document_id.strip() in superseding_targets:
                result.rejected.append((r.chunk, "explicitly superseded by RET-2026-class document"))
                continue
            kept.append(r)
        eligible = kept

    eligible.sort(key=lambda r: r.final_score, reverse=True)

    if not eligible or result.best_relevance < min_relevance:
        result.insufficient = True
        result.insufficiency_reason = "no sufficiently relevant authoritative evidence"
        return result

    top = eligible[0]
    if top.relevance < 0.18:
        result.insufficient = True
        result.insufficiency_reason = "best evidence relevance too low for a confident answer"

    # Keep at most three passages per document so multi-section answers
    # (e.g. destinations + delivery estimate + duties) stay possible while one
    # verbose document still cannot dominate the evidence package.
    per_doc: dict[str, int] = {}
    selected: list[RetrievedChunk] = []
    for r in eligible:
        count = per_doc.get(r.filename, 0)
        if count >= 3:
            continue
        per_doc[r.filename] = count + 1
        selected.append(r)
        if len(selected) >= max_evidence:
            break

    result.selected = selected
    result.pool = eligible
    return result
