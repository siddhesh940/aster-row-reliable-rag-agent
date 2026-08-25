"""Explicit conflict detection between active authoritative sources.

Rules (deterministic, rule-assisted per assignment §41):

* Known cross-document contradiction signatures are checked first
  (Breeze Tumbler cleaning guidance).
* A generic guard detects two *active + official + customer-audience*
  passages asserting different numeric policy windows for the same topic.
  Divergence involving superseded/draft material is NOT a conflict — that is
  handled by precedence, not surfaced as a contradiction.
"""

from __future__ import annotations

import re

from .contracts import Conflict, RetrievedChunk

_DAY_WINDOW_RE = re.compile(r"(\d+)\s*[- ]?\s*calendar[\s-]*days?", re.IGNORECASE)


def _is_active_authoritative(r: RetrievedChunk) -> bool:
    m = r.chunk.meta
    return m.status == "active" and m.policy_authority == "official"


def detect_conflicts(
    selected: list[RetrievedChunk], rewritten_query: str
) -> list[Conflict]:
    conflicts: list[Conflict] = []
    q = rewritten_query.lower()

    # --- Signature 1: Breeze Tumbler cleaning -------------------------------
    care = [r for r in selected if r.chunk.meta.document_id == "CARE-2026-01"]
    card = [r for r in selected if r.chunk.meta.document_id == "PROD-BREEZE-20"]
    care_terms = ("dishwasher", "hand-wash", "hand wash", "wash", "clean")
    if (
        care
        and card
        and _is_active_authoritative(care[0])
        and _is_active_authoritative(card[0])
        and any(t in q for t in care_terms)
    ):
        conflicts.append(
            Conflict(
                topic="Breeze Tumbler cleaning",
                sources=[care[0].chunk.citation, card[0].chunk.citation],
                summary_a=(
                    "The Product Care Guide says the stainless-steel body should be "
                    "hand-washed and only the lid is top-rack dishwasher safe."
                ),
                summary_b=(
                    "The Breeze Tumbler product card says all components are "
                    "dishwasher safe, top rack recommended."
                ),
                note="Both documents are active, official, customer-facing, and neither supersedes the other.",
            )
        )

    # --- Generic numeric-window divergence ----------------------------------
    # Restricted to documents whose subject IS the return window, and
    # delegation-aware: when one active policy explicitly hands the other case
    # to another document ("TrailPlus members receive a different return
    # window"), differing numbers are NOT a conflict (bug diary #2). Without
    # the delegation guard this rule false-fired on the legitimate
    # standard-vs-TrailPlus question.
    RETURN_WINDOW_DOC_IDS = {"RET-2026-01", "RET-2024-01", "MEM-2026-01"}
    DELEGATION_MARKERS = (
        "different return window",
        "see the trailplus membership",
        "membership was active when the order was placed receives",
    )
    entries: list[tuple[str, set[int], str]] = []
    for r in selected:
        if not _is_active_authoritative(r):
            continue
        if r.chunk.meta.document_id not in RETURN_WINDOW_DOC_IDS:
            continue
        values = {int(v) for v in _DAY_WINDOW_RE.findall(r.chunk.text)}
        low = r.chunk.text.lower()
        if not values or not (
            "return window" in low or "calendar days of delivery" in low
        ):
            continue
        entries.append((r.filename, values, low))

    conflicting = []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            file_a, vals_a, low_a = entries[i]
            file_b, vals_b, low_b = entries[j]
            if vals_a & vals_b:
                continue
            delegated = any(m in low_a or m in low_b for m in DELEGATION_MARKERS)
            if not delegated:
                conflicting.append((file_a, file_b, sorted(vals_a | vals_b)))

    if conflicting:
        files = [f for pair in conflicting for f in pair[:2]]
        conflicts.append(
            Conflict(
                topic="Return window length across active policies",
                sources=[
                    {"file": f, "heading": "Return window", "document_id": ""}
                    for f in dict.fromkeys(files)
                ],
                summary_a=(
                    f"Diverging return windows found in active sources without "
                    f"a delegation relationship: {sorted({v for _,_,vals in [(a,b,c) for a,b,c in conflicting] for v in c})} days."
                ),
                summary_b="Neither document supersedes or delegates to the other on this point.",
                note="Requires human confirmation of the correct window.",
            )
        )

    return conflicts
