"""Query/context resolution for multi-turn conversations.

Deterministic rules resolve follow-ups ("What about Canada?", "When will it
arrive?") against session state before retrieval or tool use. Ambiguity is
never guessed: if a question could refer to one of several known orders, the
agent asks instead of picking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .contracts import ORDER_ID_SCAN_RE
from .orders import extract_order_ids
from .sessions import Session

TOPIC_SIGNATURES: list[tuple[tuple[str, ...], str]] = [
    (("return", "refund", "send it back", "exchange"), "returns policy"),
    (("ship", "shipping", "deliver", "delivery", "arrive"), "shipping"),
    (("international", "abroad", "overseas", "canada", "germany", "country"), "international shipping"),
    (("warranty", "guarantee", "defect"), "warranty"),
    (("cancel", "cancellation", "address change"), "order changes"),
    (("trailplus", "membership"), "TrailPlus membership"),
    (("gift card",), "gift cards"),
    (("dishwasher", "wash", "clean", "care"), "product care"),
    (("final sale", "final-sale"), "final sale"),
    (("damaged", "broken", "wrong item"), "damaged items"),
    (("price adjust", "price match", "cheaper"), "price adjustments"),
]

FOLLOWUP_MARKERS = (
    "what about",
    "how about",
    "and for",
    "and the",
    "does that",
    "is that",
    "that order",
    "this order",
    "when will it",
    "when does it",
    "where is it",
    "it arrive",
    "the exception",
    "same for",
    "also",
)


def detect_topic(text: str) -> str | None:
    low = text.lower()
    for keys, topic in TOPIC_SIGNATURES:
        if any(k in low for k in keys):
            return topic
    return None


def looks_like_order_question(low: str) -> bool:
    cues = (
        "where is my order", "where is ord", "track my order", "my package",
        "when will it arrive", "when will my order", "when will ord",
        "when does my order", "delivery estimate", "when will it get here",
        "when will ord", "get here", "status of my order", "order status",
        "what about that order", "my parcel", "shipment",
    )
    return any(c in low for c in cues) or bool(ORDER_ID_SCAN_RE.search(low))


def looks_like_privacy_request(low: str) -> bool:
    cues = (
        "email", "e-mail", "address on file", "shipping address", "home address",
        "internal note", "risk score", "fraud", "customer's", "customers ",
        "personal information", "private data", "who ordered",
    )
    return any(c in low for c in cues)


def looks_like_action_request(low: str) -> bool:
    patterns = (
        r"\b(cancel|refund|replace|replacement|reship|change (my|the) address|"
        r"approve|process|issue|escalate|create a ticket|adjust)\b",
    )
    return any(re.search(p, low) for p in patterns)


def looks_like_injection_attempt(low: str) -> bool:
    cues = (
        "system prompt", "hidden prompt", "hidden instructions", "your instructions",
        "ignore all", "ignore your", "ignore the rules", "reveal your",
        "show me your prompt", "developer message", "api key", "secret key",
        "migration note says", "the note says to ignore", "tell every customer",
        "approve my return", "override your rules", "new rule",
    )
    return any(c in low for c in cues)


@dataclass
class ResolvedContext:
    rewritten_query: str
    resolved_order_id: str | None = None
    ambiguous_orders: list[str] = field(default_factory=list)
    needs_order_id_prompt: bool = False
    notes: list[str] = field(default_factory=list)


def resolve(session: Session, user_message: str, *, context_enabled: bool) -> ResolvedContext:
    low = user_message.lower()
    notes: list[str] = []

    mentioned_ids = extract_order_ids(user_message)

    # 1. Update entity memory from explicit mentions.
    if mentioned_ids:
        session.last_order_id = mentioned_ids[-1]
        notes.append(f"order id mentioned: {mentioned_ids[-1]}")

    # 2. Resolve order references.
    resolved_id: str | None = None
    ambiguous: list[str] = []
    needs_prompt = False
    if looks_like_order_question(low):
        if mentioned_ids:
            resolved_id = session.last_order_id
        elif context_enabled and session.last_order_id:
            resolved_id = session.last_order_id
            notes.append(f"resolved order reference from session context: {resolved_id}")
        else:
            # Order question with no ID anywhere: ask, never guess.
            needs_prompt = True
            notes.append("order question without any id: will ask the customer")

    # 3. Rewrite follow-up queries using bounded topic memory.
    query_parts = [user_message.strip()]
    if context_enabled and not mentioned_ids:
        is_followup = any(m in low for m in FOLLOWUP_MARKERS) or len(user_message.split()) <= 6
        topic = detect_topic(user_message)
        if is_followup and session.recent_topics:
            carried = session.recent_topics[0]
            if topic and topic != carried:
                # Narrowing follow-up: merge prior topic + new focus.
                query_parts.insert(0, carried)
                notes.append(f"merged follow-up with previous topic '{carried}'")
            elif not topic:
                query_parts.insert(0, carried)
                notes.append(f"resolved pronoun-style follow-up using topic '{carried}'")
        if topic:
            session.remember_topic(topic)
    elif context_enabled and mentioned_ids and detect_topic(user_message):
        session.remember_topic(detect_topic(user_message))  # type: ignore[arg-type]

    rewritten = " ".join(query_parts)
    return ResolvedContext(
        rewritten_query=rewritten,
        resolved_order_id=resolved_id,
        ambiguous_orders=ambiguous,
        needs_order_id_prompt=needs_prompt,
        notes=notes,
    )
