"""Typed data contracts shared across the pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocumentMeta:
    """Front-matter metadata for one knowledge-base document."""

    document_id: str
    title: str
    status: str                 # active | superseded | draft | ...
    audience: str               # customer | internal
    policy_authority: str       # official | none | ...
    effective_date: str | None = None
    superseded_date: str | None = None
    last_reviewed: str | None = None
    supersedes: str | None = None
    superseded_by: str | None = None
    customer_answering: bool | None = None  # explicit opt-out flag when present

    @property
    def may_answer_customers(self) -> bool:
        """Whether this document is allowed to be used as evidence at all."""
        if self.customer_answering is False:
            return False
        if self.status in ("superseded", "draft", "archived", "retired"):
            return False
        if self.audience == "internal" and self.status != "active":
            return False
        return True

    @property
    def precedence_rank(self) -> float:
        """Static authority score. Higher wins. Explicitly NOT recency-driven."""
        rank = 0.0
        if self.status == "active":
            rank += 3.0
        elif self.status == "superseded":
            rank -= 2.0
        else:  # draft, archived, anything unrecognised
            rank -= 3.0
        if self.policy_authority == "official":
            rank += 1.5
        else:
            rank -= 1.5
        if self.audience == "customer":
            rank += 0.5
        elif self.audience == "internal":
            # Internal active documents (e.g. escalation rules) are usable as
            # behavioural guidance but must never outrank customer-facing policy.
            rank -= 0.75
        return rank


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage of a Markdown document."""

    chunk_id: str
    filename: str
    heading_path: tuple[str, ...]     # e.g. ("Breeze Tumbler",) or ("Return window",)
    text: str
    meta: DocumentMeta

    @property
    def primary_heading(self) -> str:
        return self.heading_path[-1] if self.heading_path else self.meta.title

    @property
    def citation(self) -> dict[str, str]:
        return {
            "file": self.filename,
            "heading": self.primary_heading,
            "document_id": self.meta.document_id,
            "status": self.meta.status,
            "policy_authority": self.meta.policy_authority,
            "audience": self.meta.audience,
        }


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    relevance: float          # pure similarity score
    final_score: float        # relevance + precedence boost (full profile)

    @property
    def filename(self) -> str:
        return self.chunk.filename


@dataclass
class Conflict:
    topic: str
    sources: list[dict[str, str]]      # citations of both sides
    summary_a: str
    summary_b: str
    note: str = ""


@dataclass
class OrderLookupResult:
    """Sanitized, customer-safe order tool result.

    Only whitelisted fields ever make it into this object; sensitive fields
    (customer name/email/address, risk scores, warehouse notes, support tags)
    are stripped inside the tool itself and can never reach the model or logs.
    """

    found: bool
    order_id: str | None = None
    status: str | None = None
    membership_tier: str | None = None
    items: list[dict] = field(default_factory=list)   # name/quantity/final_sale only
    placed_at: str | None = None
    delivered_at: str | None = None
    carrier: str | None = None
    tracking_number: str | None = None
    estimated_delivery: str | None = None
    customer_safe_message: str | None = None
    stale_fields_suppressed: bool = False
    error: str | None = None                          # malformed-id / not-found detail

    def to_dict(self) -> dict:
        d = {
            "found": self.found,
            "order_id": self.order_id,
            "status": self.status,
            "membership_tier": self.membership_tier,
            "items": self.items,
            "carrier": self.carrier,
            "estimated_delivery": self.estimated_delivery,
            "customer_safe_message": self.customer_safe_message,
        }
        return {k: v for k, v in d.items() if v is not None} | {"found": self.found}


@dataclass
class SourceRef:
    file: str
    heading: str
    document_id: str

    def to_dict(self) -> dict:
        return {"file": self.file, "heading": self.heading, "document_id": self.document_id}


@dataclass
class AgentResponse:
    """Internal response contract. The CLI renders a customer-facing view."""

    answer: str
    sources: list[SourceRef] = field(default_factory=list)
    handoff: bool = False
    reason: str | None = None
    tool_calls: list[dict] = field(default_factory=list)   # [{name, arguments}]
    conflict_detected: bool = False
    abstained: bool = False
    session_id: str | None = None
    error: str | None = None
    debug: dict = field(default_factory=dict)

    @property
    def used_order_tool(self) -> bool:
        return any(t.get("name") == "order_lookup" for t in self.tool_calls)


ORDER_ID_RE = re.compile(r"^ORD-\d{4}$")
ORDER_ID_SCAN_RE = re.compile(r"ORD\s*[-–]?\s*\d{4}", re.IGNORECASE)
