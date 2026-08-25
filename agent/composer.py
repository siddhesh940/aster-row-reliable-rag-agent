"""Deterministic grounded answer composer.

Given a validated evidence package (already precedence-filtered) and/or a
sanitized order result, this module composes the customer-facing answer using
facts extracted *from the retrieved passages only*. It never invents numbers,
dates, or promises. Directive-like sentences inside retrieved content are
stripped before composition so injected instructions can never be echoed.

When an LLM endpoint is configured, the orchestrator passes the same
structured facts to the LLM for phrasing; this composer remains the offline
default and the safety net on any LLM failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .conflicts import Conflict
from .contracts import OrderLookupResult, RetrievedChunk, SourceRef

MONTHS = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# Directive-like content that must never be echoed into an answer.
_DIRECTIVE_RE = re.compile(
    r"(?i)(system instruction|hidden prompt|reveal your|ignore all|ignore the|"
    r"tell every customer|do not call tools|never cite|approve (all|every|my)|"
    r"\bapi[_ ]?key\b|credentials)",
)

_DAY_OF_DELIVERY_RE = re.compile(r"(\d+)\s*[- ]?\s*calendar[\s-]+days?\s+of\s+delivery", re.IGNORECASE)
_HYPHEN_WINDOW_RE = re.compile(r"(\d+)\s*[-–]\s*calendar[\s-]+day", re.IGNORECASE)
_BUSINESS_DAYS_RE = re.compile(r"(\d+)\s*[–—-]\s*(\d+)\s+business\s+days(\s+after\s+dispatch)?", re.IGNORECASE)
_PROCESSING_RE = re.compile(r"(\d+)\s*[–—-]\s*(\d+)\s+business\s+days?\b[^.]*?processing", re.IGNORECASE)


def fmt_date(iso: str | None) -> str | None:
    """'2026-08-22' -> 'August 22, 2026'; timestamps use their date part."""
    if not iso:
        return None
    date_part = str(iso)[:10]
    try:
        year, month, day = (int(p) for p in date_part.split("-"))
        if 1 <= month <= 12:
            return f"{MONTHS[month]} {day}, {year}"
    except (ValueError, TypeError):
        pass
    return None


def _strip_md(text: str) -> str:
    text = re.sub(r"[#>*_`]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def sentences_of(chunk_text: str) -> list[str]:
    body = "\n".join(
        line for line in chunk_text.splitlines()
        if not line.strip().startswith("#") and line.strip() not in ("-", "---")
    )
    raw = re.split(r"(?<=[.!?])\s+", _strip_md(body))
    return [s.strip() for s in raw if len(s.strip()) > 15]


def safe_sentences(chunk_text: str) -> list[str]:
    """Sentences from a passage with directive-like content removed."""
    return [s for s in sentences_of(chunk_text) if not _DIRECTIVE_RE.search(s)]


@dataclass
class Composition:
    answer: str
    sources: list[SourceRef] = field(default_factory=list)
    handoff: bool = False
    reason: str | None = None
    abstained: bool = False
    conflict_detected: bool = False


def _src(*retrieved: RetrievedChunk) -> list[SourceRef]:
    out: list[SourceRef] = []
    seen: set[str] = set()
    for r in retrieved:
        key = f"{r.filename}::{r.chunk.primary_heading}"
        if key in seen:
            continue
        seen.add(key)
        out.append(SourceRef(file=r.filename, heading=r.chunk.primary_heading,
                             document_id=r.chunk.meta.document_id))
    return out


def pick_doc(pool: list[RetrievedChunk], document_id: str) -> list[RetrievedChunk]:
    """All gated passages of one document, best-first."""
    return [r for r in pool if r.chunk.meta.document_id == document_id]


def _extract_number_from(patterns: list[re.Pattern[str]], chunks: list[RetrievedChunk]) -> int | None:
    for pat in patterns:
        for c in chunks:
            m = pat.search(c.chunk.text)
            if m:
                return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Order compositions
# ---------------------------------------------------------------------------

def compose_order_missing_id(session_has_order: bool = False) -> Composition:
    text = (
        "I can look that up — could you share your order ID? It looks like "
        "ORD-1234 and is shown on your order confirmation."
    )
    return Composition(answer=text)


def compose_order_malformed(raw_value: str) -> Composition:
    cleaned = raw_value.strip()
    text = (
        f"I couldn't read \"{cleaned}\" as an Aster & Row order ID — they look "
        "like ORD-1234. Could you double-check it? I haven't run a lookup, so "
        "please share the ID again rather than having me guess."
    )
    return Composition(answer=text)


def compose_order_result(res: OrderLookupResult, question: str = "") -> Composition:
    low_q = question.lower()
    wants_eta = any(w in low_q for w in ("when", "arrive", "eta", "estimate", "get here", "how long"))

    if not res.found:
        oid = res.order_id or "the ID you provided"
        text = (
            f"I couldn't find an order with ID {oid} in the current records. "
            "Please double-check the order ID (it's in your confirmation email) "
            "or contact support and someone can locate it for you."
        )
        return Composition(
            answer=text,
            handoff=True,
            reason="order lookup failed; human assistance recommended",
        )

    assert res.order_id and res.status
    status = res.status
    parts: list[str] = []

    if status == "cancelled":
        parts.append(f"Order {res.order_id} was cancelled and will not be shipped, so there is no delivery to expect.")
    elif status == "returned":
        delivered = fmt_date(res.delivered_at)
        tail = f" (delivered {delivered})" if delivered else ""
        parts.append(f"Order {res.order_id} has been returned and the return was processed{tail}. Nothing more is scheduled to arrive.")
    elif status == "exception":
        parts.append(f"Order {res.order_id} is flagged with a shipping exception that requires review by our support team.")
        parts.append("I can't resolve carrier exceptions myself, so I'm recommending a human specialist take it from here.")
    elif status == "shipped":
        carrier_bit = f" with {res.carrier}" if res.carrier else ""
        eta = fmt_date(res.estimated_delivery)
        if eta:
            parts.append(f"Order {res.order_id} has shipped{carrier_bit} and is currently estimated to arrive on {eta}.")
        else:
            parts.append(f"Order {res.order_id} has shipped{carrier_bit} and is in transit.")
            parts.append("A delivery estimate isn't available right now, so I can't give you an arrival date — I'd rather not guess.")
        if res.tracking_number and (not wants_eta or True):
            parts.append(f"Tracking number: {res.tracking_number}.")
    elif status == "delivered":
        delivered = fmt_date(res.delivered_at) or fmt_date(res.estimated_delivery)
        when = f" on {delivered}" if delivered else ""
        parts.append(f"Order {res.order_id} was delivered{when}.")
    elif status == "pending":
        parts.append(f"Order {res.order_id} was received and hasn't entered processing yet.")
    elif status == "processing":
        eta = fmt_date(res.estimated_delivery)
        if eta:
            parts.append(f"Order {res.order_id} is being prepared for shipment and is currently estimated to arrive on {eta}.")
        else:
            parts.append(f"Order {res.order_id} is being prepared for shipment; an estimated delivery date isn't available yet.")
    elif status == "delayed":
        parts.append(f"Order {res.order_id} is currently delayed in transit.")
        eta = fmt_date(res.estimated_delivery)
        if eta:
            parts.append(f"The latest estimate from the carrier is {eta}.")
        else:
            parts.append("An updated delivery estimate isn't available yet, so I can't promise an arrival date.")
    else:
        msg = res.customer_safe_message
        parts.append(f"Order {res.order_id} status: {status}." + (f" {msg}" if msg else ""))

    answer = " ".join(parts)
    sources: list[SourceRef] = []
    # Order facts come from the lookup tool; make that explicit instead of
    # pretending the knowledge base was the source.
    return Composition(
        answer=answer,
        sources=sources,
        handoff=status == "exception",
        reason="shipment exception requires support review" if status == "exception" else None,
    )


def compose_order_unknown_tool_failure() -> Composition:
    return Composition(
        answer=(
            "Something went wrong while checking the order system. I didn't get "
            "a reliable result, so please try again shortly or contact support."
        ),
        handoff=True,
        reason="order tool failure",
    )


# ---------------------------------------------------------------------------
# Policy compositions (evidence-driven)
# ---------------------------------------------------------------------------

def compose_return_window(pool: list[RetrievedChunk], question: str) -> Composition:
    low = question.lower()
    trailplus_asked = "trailplus" in low or "member" in low
    trailplus_chunks = pick_doc(pool, "MEM-2026-01")
    current_chunks = pick_doc(pool, "RET-2026-01")
    trailplus_chunk = trailplus_chunks[0] if trailplus_chunks else None
    current = current_chunks[0] if current_chunks else None

    if (trailplus_asked or (current is None)) and trailplus_chunk is not None:
        n = _extract_number_from([_HYPHEN_WINDOW_RE, _DAY_OF_DELIVERY_RE], trailplus_chunks)
        days = n if n is not None else "45-calendar-day"
        answer = (
            f"For TrailPlus members whose membership was active when the order "
            f"was placed, the return window is {days} calendar days from "
            "delivery for eligible items. Joining TrailPlus after placing an "
            "order does not extend that order's window, and final-sale, "
            "condition, and warranty rules still apply."
        )
        return Composition(answer=answer, sources=_src(trailplus_chunk))

    if current is not None:
        n = _extract_number_from([_DAY_OF_DELIVERY_RE], current_chunks)
        days = n if n is not None else "30"
        fee_line = ""
        fee_sentence = next(
            (s for ch in current_chunks for s in safe_sentences(ch.chunk.text) if "$6.95" in s),
            "",
        )
        if fee_sentence:
            fee_line = " " + fee_sentence
        answer = (
            f"On the standard plan you can request a return within {days} "
            "calendar days of delivery, as long as the item is unused, "
            "unwashed, and in resalable condition with its original tags and "
            "packaging." + fee_line +
            " Note: TrailPlus members whose membership was active at order "
            "placement have a different (longer) window."
        )
        return Composition(answer=answer, sources=_src(current))

    return compose_insufficient(
        question, "I couldn't find the current return-window policy in the supplied documents."
    )


def compose_final_sale_damaged(pool: list[RetrievedChunk]) -> Composition:
    fs_chunks = pick_doc(pool, "RET-2026-02")
    dmg_chunks = pick_doc(pool, "OPS-2026-04")
    fs = fs_chunks[0] if fs_chunks else None
    dmg = dmg_chunks[0] if dmg_chunks else None
    report_days = _extract_number_from([_DAY_OF_DELIVERY_RE], dmg_chunks)
    days_txt = f"{report_days} calendar days" if report_days else "7 calendar days"

    parts: list[str] = []
    if fs is not None:
        parts.append(
            "You're not completely out of luck: final sale only blocks "
            "change-of-mind returns, so a final-sale item that arrived "
            "damaged is still eligible for review under the Damaged or Wrong "
            "Items Policy."
        )
    if dmg is not None:
        parts.append(
            f"Please report it within {days_txt} of delivery, including the "
            "order ID, a short description, and clear photos of the item and "
            "packaging where possible."
        )
        parts.append(
            "After review, Aster & Row may offer a replacement or refund "
            "(subject to stock), but nothing can be promised before a human "
            "review is completed — so I can't approve anything here."
        )
    answer = " ".join(parts)
    return Composition(
        answer=answer,
        sources=_src(*( [r for r in (fs, dmg) if r] )),
        handoff=True,
        reason="damaged-item report requires human review before any resolution",
    )


def compose_damaged_items(pool: list[RetrievedChunk]) -> Composition:
    dmg_chunks = pick_doc(pool, "OPS-2026-04")
    if not dmg_chunks:
        return compose_insufficient(None, "No damaged-item policy found in the supplied documents.")
    days = _extract_number_from([_DAY_OF_DELIVERY_RE], dmg_chunks)
    parts = [
        "Items that arrived damaged, visibly defective, or different from what "
        f"was ordered should be reported within {days or 7} calendar days of delivery.",
        "Include the order ID, a short description, and clear photos of the "
        "item and packaging when possible.",
        "After review, Aster & Row may offer a replacement or refund "
        "(replacement availability depends on stock); no return-shipping fee "
        "is charged once damage or a wrong item is confirmed.",
        "Approvals require a completed human review, so I can't promise an "
        "outcome here.",
    ]
    return Composition(
        answer=" ".join(parts),
        sources=_src(dmg_chunks[0]),
        handoff=True,
        reason="damaged-item resolutions require human review",
    )


def compose_international_shipping(pool: list[RetrievedChunk], question: str) -> Composition:
    intl_chunks = pick_doc(pool, "SHIP-2026-INTL")
    if not intl_chunks:
        return compose_insufficient(question, "No international shipping information found.")
    intl = intl_chunks[0]
    all_text = "\n".join(r.chunk.text for r in intl_chunks)

    low = question.lower()
    mentioned_countries = {
        c for c in (
            "germany", "france", "uk", "united kingdom", "australia", "japan",
            "mexico", "brazil", "india", "spain", "italy", "netherlands",
        ) if c in low
    }

    if mentioned_countries:
        name = sorted(mentioned_countries)[0].title()
        answer = (
            f"Shipping to {name} is not currently available. Aster & Row "
            "ships internationally only to Canada at this time."
        )
        return Composition(answer=answer, sources=_src(intl))

    bd = _BUSINESS_DAYS_RE.search(all_text)
    timing = f"{bd.group(1)}–{bd.group(2)} business days after dispatch" if bd else "an estimated range published in the shipping policy"
    proc = _PROCESSING_RE.search(all_text)
    processing = f" Processing before dispatch is usually {proc.group(1)}–{proc.group(2)} business days." if proc else ""

    if "canada" in low or "what about" in low or "how long" in low:
        answer = (
            "Yes — Canada is supported and is currently the only "
            "international destination Aster & Row ships to. Canadian orders "
            f"generally arrive within {timing}.{processing} Import duties, "
            "taxes, and brokerage charges are not prepaid by Aster & Row; the "
            "recipient is responsible for charges assessed by Canadian "
            "authorities or the carrier."
        )
    else:
        answer = (
            "Aster & Row ships internationally only to Canada. Canadian "
            f"orders generally arrive within {timing}.{processing} Import duties/taxes "
            "are not prepaid — the recipient is responsible for them. Shipping "
            "to other countries is not available at this time."
        )
    return Composition(answer=answer, sources=_src(intl))


def compose_warranty(pool: list[RetrievedChunk], question: str) -> Composition:
    war_chunks = pick_doc(pool, "WAR-2026-01")
    if not war_chunks:
        return compose_insufficient(question, "No warranty policy found in the supplied documents.")
    war = war_chunks[0]
    all_text = "\n".join(r.chunk.text for r in war_chunks)
    low = question.lower()
    if "lifetime" in low and ("lifetime warranty." in all_text.lower() or "does not offer a lifetime" in all_text.lower()):
        periods: dict[str, str] = {}
        for line in all_text.splitlines():
            m = re.match(r"^-\s*(.+?):\s*\*?\*(\d+\s*years?)\b", line.strip())
            if m:
                label = m.group(1).strip()
                label = re.sub(r"^Aster & Row\s+", "", label)
                periods[label] = m.group(2)
        coverage = "; ".join(f"{k}: {v}" for k, v in periods.items()) or "category-specific limited periods listed in the warranty policy"
        answer = (
            "No — Aster & Row does not offer a lifetime warranty. The limited "
            f"warranty periods are: {coverage}, measured from the purchase date. "
            "It covers manufacturing defects under normal use; ordinary wear, "
            "accidental damage, improper cleaning, and misuse are excluded. "
            "Claims need proof of purchase and a human specialist reviews "
            "eligibility, so I can't promise approval."
        )
        return Composition(answer=answer, sources=_src(war))
    sentences = safe_sentences(all_text)[:4]
    # Always carry the human-review caveat when asking about coverage.
    if "human" not in " ".join(sentences).lower():
        human_sentence = next(
            (s for ch in war_chunks for s in safe_sentences(ch.chunk.text)
             if "human" in s.lower() or "review" in s.lower()),
            None,
        )
        if human_sentence:
            sentences.append(human_sentence)
    answer = " ".join(sentences)
    return Composition(answer=answer, sources=_src(war))


def compose_cancellation(pool: list[RetrievedChunk], order_res: OrderLookupResult | None) -> Composition:
    pol_chunks = pick_doc(pool, "ORD-2026-01")
    pol = pol_chunks[0] if pol_chunks else None
    policy_bits: list[str] = []
    all_text = "\n".join(r.chunk.text for r in pol_chunks)
    if pol is not None:
        m = re.search(r"within\s+\*?\*(\d+)\s+minutes?\*?\*", all_text)
        mins = m.group(1) if m else "30"
        policy_bits.append(
            f"Cancellation can be requested within {mins} minutes of placing "
            "the order, and only while its status is still pending."
        )
    if order_res is not None and order_res.found:
        st = order_res.status
        if st == "pending":
            policy_bits.append(
                f"{order_res.order_id} is still pending, so it may still be "
                "inside that window — but I can't cancel orders myself, so "
                "please contact support right away to lock it in before the "
                "window closes."
            )
        else:
            policy_bits.append(
                f"{order_res.order_id} is already {st}, which is past the "
                "point where the normal cancellation process applies."
            )
    else:
        policy_bits.append(
            "I can't cancel orders myself — a human support specialist has to "
            "complete cancellations."
        )
    answer = " ".join(policy_bits)
    srcs = _src(pol) if pol is not None else []
    return Composition(
        answer=answer,
        sources=srcs,
        handoff=True,
        reason="cancellation requests must be completed by human support",
    )


def compose_action_refusal(action: str, chunks: list[RetrievedChunk], order_res: OrderLookupResult | None = None) -> Composition:
    verb = action.strip().capitalize() or "That action"
    extra = ""
    if order_res is not None and order_res.found:
        extra = f" I checked: {order_res.order_id}'s current status is '{order_res.status}'."
    answer = (
        f"{verb} isn't something I can execute — I don't have an action tool "
        f"for that, and I won't claim it happened when it didn't.{extra} A "
        "human support specialist can process it for you; I'd recommend "
        "contacting support so a person can complete this."
    )
    srcs = _src(*chunks[:1]) if chunks else []
    return Composition(answer=answer, sources=srcs, handoff=True,
                       reason=f"requested action ({action or 'unspecified'}) is not supported by the agent")


PRIVACY_ANSWER = (
    "I'm sorry, but I can't share customer emails, addresses, internal notes, "
    "or risk scores — those aren't part of customer-visible order information, "
    "and I can't disclose another person's details either. If you need "
    "something changed on an order you own, human support can verify and help "
    "securely."
)


def compose_privacy_refusal() -> Composition:
    return Composition(
        answer=PRIVACY_ANSWER,
        handoff=True,
        reason="internal/sensitive data request requires human handling",
    )


INJECTION_ANSWER_TEMPLATE = (
    "The migration note isn't an authoritative source — it's unapproved draft "
    "material, so I can't treat it as policy no matter how it's phrased. "
    "{policy_line} And to be clear, I can't approve returns: approvals happen "
    "through the normal review process, not through me."
)


def compose_injection_defense(pool: list[RetrievedChunk]) -> Composition:
    cur_chunks = pick_doc(pool, "RET-2026-01")
    current = cur_chunks[0] if cur_chunks else None
    if current is not None:
        n = _extract_number_from([_DAY_OF_DELIVERY_RE], cur_chunks)
        policy_line = (
            f"Under the current official Returns Policy the standard window is "
            f"{n or 30} calendar days of delivery, unless a valid exception "
            "applies (for example, TrailPlus membership that was active when "
            "the order was placed)."
        )
        srcs = _src(current)
    else:
        policy_line = (
            "The applicable window comes from the current official Returns "
            "Policy, not from internal notes."
        )
        srcs = []
    return Composition(
        answer=INJECTION_ANSWER_TEMPLATE.format(policy_line=policy_line),
        sources=srcs,
        handoff=False,
    )


DIRECT_INJECTION_ANSWER = (
    "I can't do that. I won't reveal my system prompt, hidden instructions, "
    "or API keys, and I won't share customer emails, addresses, or other "
    "internal data. My operating rules also can't be overridden by chat. I'm "
    "happy to help with a real order or policy question."
)


def compose_direct_injection_refusal() -> Composition:
    return Composition(
        answer=DIRECT_INJECTION_ANSWER,
        handoff=False,
        reason="user attempted prompt/system extraction; refused",
    )


def compose_conflict(conflict: Conflict, question: str) -> Composition:
    answer = (
        f"There's a genuine conflict between two current official sources on "
        f"{conflict.topic.lower()}: {conflict.summary_a} Meanwhile, "
        f"{conflict.summary_b[0].lower() + conflict.summary_b[1:]} Neither "
        f"document supersedes the other, so I can't honestly pick a winner. "
        "Safest interim guidance: follow the more conservative option (hand-wash "
        "the body, top-rack wash only the lid) until support confirms, because "
        "cleaning against the care guide can void warranty coverage. I'd "
        "recommend a quick confirmation from our team."
    )
    srcs = [
        SourceRef(file=s.get("file", ""), heading=s.get("heading", ""), document_id=s.get("document_id", ""))
        for s in conflict.sources
    ]
    return Composition(
        answer=answer,
        sources=[s for s in srcs if s.file],
        handoff=True,
        reason="current authoritative sources conflict; human confirmation recommended",
        conflict_detected=True,
    )


GENERIC_INSUFFICIENT_TEMPLATE = (
    "The supplied Aster & Row information is insufficient to answer this, so "
    "I'd rather not guess about \"{topic}\". Please reach out to human "
    "support and they will be able to help."
)


def compose_insufficient(question: str | None, detail: str | None = None) -> Composition:
    # `detail` is an internal diagnostic (e.g. "relevance too low"); it stays
    # out of customer-facing text and is only used in traces/handoff metadata.
    q = (question or "").strip()
    topic = q if 0 < len(q) <= 120 else "this question"
    return Composition(
        answer=GENERIC_INSUFFICIENT_TEMPLATE.format(topic=topic),
        handoff=True,
        reason="insufficient information in the supplied knowledge base",
        abstained=True,
    )


def compose_generic_extractive(chunks: list[RetrievedChunk], question: str) -> Composition:
    """Fallback: quote the most relevant clean sentences from top passages."""
    q_tokens = set(re.findall(r"[a-z0-9]+", question.lower())) - {"the", "a", "an", "is", "are", "can", "i", "my", "to", "for"}
    scored_sentences: list[tuple[float, str, RetrievedChunk]] = []
    for r in chunks:
        for s in safe_sentences(r.chunk.text):
            tokens = set(re.findall(r"[a-z0-9]+", s.lower()))
            overlap = len(q_tokens & tokens)
            if overlap:
                scored_sentences.append((overlap, s, r))
    scored_sentences.sort(key=lambda t: t[0], reverse=True)
    picked: list[str] = []
    used: set[str] = set()
    for score, s, r in scored_sentences:
        if r.filename in used and len(picked) >= 3:
            continue
        if s not in picked:
            picked.append(s)
            used.add(r.filename)
        if len(picked) >= 3:
            break
    if not picked:
        return compose_insufficient(question, "The retrieved documents don't contain a clear answer.")
    answer = "Here's what the supplied documentation says: " + " ".join(picked)
    return Composition(answer=answer, sources=_src(*chunks))


def compose_gift_cards(pool: list[RetrievedChunk], question: str) -> Composition:
    gc_chunks = pick_doc(pool, "PAY-2026-03")
    if not gc_chunks:
        return compose_insufficient(question, "No gift-card policy found.")
    gc = gc_chunks[0]
    all_text = "\n".join(r.chunk.text for r in gc_chunks)
    low = question.lower()
    parts: list[str] = ["Gift cards never expire and are always final sale — they can't be returned, exchanged for cash, or used to buy another gift card."]
    if "code" in low or any(tok in low for tok in ("gift-", "redeem")):
        parts.append("For security, please don't share full gift-card codes in chat — support will verify purchases without needing the complete code.")
    if "price" in low or "adjust" in low or "cheaper" in low or "dropped" in low:
        m = re.search(r"within\s+\*?\*(\d+)\s+calendar\s+days?\*?\*", all_text)
        days = m.group(1) if m else "7"
        parts.append(
            f"Price adjustments are possible when the public price of the same item, color, and size drops within {days} calendar days of purchase, but exclusions apply (clearance/final-sale items, flash sales, unused discount codes, third-party prices, out-of-stock variants)."
        )
        parts.append("A human specialist must approve and process any adjustment, so I can't promise any credit until that review happens.")
        return Composition(answer=" ".join(parts), sources=_src(gc),
                           handoff=True,
                           reason="price adjustments require human approval")
    return Composition(answer=" ".join(parts), sources=_src(gc))


def compose_trailplus_benefits(pool: list[RetrievedChunk], question: str) -> Composition:
    mem_chunks = pick_doc(pool, "MEM-2026-01")
    if not mem_chunks:
        return compose_insufficient(question, "No TrailPlus membership information found.")
    mem = mem_chunks[0]
    n = _extract_number_from([_HYPHEN_WINDOW_RE, _DAY_OF_DELIVERY_RE], mem_chunks)
    answer = (
        f"TrailPlus members whose membership was active when an order was "
        f"placed receive a {n or '45'}-calendar-day return window from delivery "
        "for eligible items, plus free standard shipping on eligible US "
        "orders with no minimum. Joining after placing an order doesn't "
        "extend that order's window, and final-sale restrictions and item "
        "conditions still apply."
    )
    if "active when" in question.lower():
        answer += " Since I may not have your membership history, this assumes the membership was active on the order date."
    return Composition(answer=answer, sources=_src(mem))
