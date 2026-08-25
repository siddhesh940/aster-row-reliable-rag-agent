"""Agent orchestrator: wires sessions, context resolution, routing, RAG,
precedence, conflict detection, the order tool, and response composition.

Design note: routing and tool-calling decisions are made by deterministic code,
NOT by the LLM. This guarantees tool-call discipline (the order tool is invoked
only when order data is actually needed, with normalized arguments), keeps raw
order data out of model context, and makes behavior testable.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone

from . import resolver as resolver_mod
from .composer import (
    Composition,
    compose_action_refusal,
    compose_cancellation,
    compose_conflict,
    compose_direct_injection_refusal,
    compose_damaged_items,
    compose_final_sale_damaged,
    compose_generic_extractive,
    compose_gift_cards,
    compose_insufficient,
    compose_international_shipping,
    compose_injection_defense,
    compose_order_malformed,
    compose_order_missing_id,
    compose_order_result,
    compose_privacy_refusal,
    compose_return_window,
    compose_trailplus_benefits,
    compose_warranty,
    fmt_date,
)
from .conflicts import detect_conflicts
from .config import Config, load_config
from .contracts import AgentResponse, OrderLookupResult, RetrievedChunk, SourceRef
from .documents import load_documents
from .chunking import chunk_all
from .indexing import VectorIndex
from .orders import OrderTool, extract_order_ids
from .precedence import select_evidence
from .redaction import redact
from .sessions import SessionStore

# Loose "attempted order id" pattern: catches ORD12 / ord-1 / ORD 10075 style
# inputs that should be rejected rather than guessed at.
_ATTEMPTED_ID_RE = re.compile(r"\bORD[\s\-–]*\d{1,6}\b", re.IGNORECASE)

_RETURN_WINDOW_RE = re.compile(r"(return|send\s+(it|this|them)?\s*back|refund window)", re.I)
_WINDOW_Q_RE = re.compile(r"(how\s+long|how\s+many\s+days|window|deadline|days\s+do\s+i\s+have|return\s+period)", re.I)


class SupportAgent:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or load_config()
        self.docs = load_documents(self.config.kb_dir)
        self.chunks = chunk_all(self.docs)
        self.index = VectorIndex()
        self.index.add(self.chunks)
        self.index.build()
        self.orders = OrderTool(self.config.orders_path)
        self.sessions = SessionStore()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle(self, message: str, session_id: str | None = None) -> AgentResponse:
        session_id = session_id or f"sess-{uuid.uuid4().hex[:10]}"
        session = self.sessions.get(session_id)
        trace: dict = {"user_message": message, "errors": [], "fallbacks": []}

        session.add_user(message)
        resolved = resolver_mod.resolve(
            session, message, context_enabled=self.config.context_resolution_enabled
        )
        trace["resolved_query"] = resolved.rewritten_query
        trace["context_notes"] = resolved.notes
        if session.history:
            trace["recent_history"] = [
                {"role": t.role, "content": t.content[:160]}
                for t in session.history[-4:]
            ]

        low = message.lower()

        # Friendly smalltalk: no evidence lookup, no handoff.
        if re.fullmatch(
            r"(hi|hello|hey|hi there|hello there|thanks|thank you|"
            r"good (morning|afternoon|evening))[!. ]*",
            message.strip().lower(),
        ):
            trace["decision"] = "smalltalk"
            composition: Composition | None = Composition(
                answer=(
                    "Hi! I'm the Aster & Row support agent. I can help with "
                    "policies (returns, shipping, warranty), product care, or "
                    "checking the status of an order — just ask."
                ),
            )
        else:
            composition = None

        if composition is None:
            try:
                composition = self._route(message, low, resolved, session, trace)
            except Exception as exc:  # fail closed, never crash on user input
                trace["errors"].append(f"pipeline error: {type(exc).__name__}")
                composition = Composition(
                    answer=(
                        "Sorry — something went wrong while handling that. I can't "
                        "safely answer right now; please try again or contact "
                        "human support."
                    ),
                    handoff=True,
                    reason="internal error",
                    abstained=True,
                )

        answer = composition.answer
        used_llm = False
        if self.config.llm_enabled:
            evidence = getattr(composition, "_evidence", [])
            tool_json = getattr(composition, "_tool_json", None)
            phrased = self._try_llm(
                question=resolved.rewritten_query,
                history=[(t.role, t.content) for t in session.history[-6:-1]],
                evidence=evidence,
                tool_result_json=tool_json,
                draft=answer,
                trace=trace,
            )
            if phrased:
                answer = phrased
                used_llm = True
            else:
                trace["fallbacks"].append("llm phrasing failed/invalid; used deterministic composer")

        session.add_agent(answer)

        response = AgentResponse(
            answer=answer,
            sources=composition.sources,
            handoff=composition.handoff,
            reason=composition.reason,
            conflict_detected=composition.conflict_detected,
            abstained=composition.abstained,
            session_id=session_id,
            debug=trace,
        )
        response.tool_calls = trace.get("tool_calls", [])  # type: ignore[assignment]
        trace["used_llm"] = used_llm
        return response

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route(self, message: str, low: str, resolved, session, trace: dict) -> Composition:
        privacy = resolver_mod.looks_like_privacy_request(low)
        injection = resolver_mod.looks_like_injection_attempt(low)
        action = resolver_mod.looks_like_action_request(low)
        order_q = bool(resolved.resolved_order_id) or resolver_mod.looks_like_order_question(low)

        # --- 1. Direct prompt/system extraction attempts -------------------
        if injection and any(
            k in low for k in ("system prompt", "hidden prompt", "hidden instructions",
                               "reveal your", "show me your prompt", "api key", "secret key",
                               "developer message")
        ):
            trace["decision"] = "direct-injection refusal"
            comp = compose_direct_injection_refusal()
            comp._evidence = []
            return comp

        # --- 2. Privacy / internal-data requests ---------------------------
        if privacy:
            trace["decision"] = "privacy refusal"
            comp = compose_privacy_refusal()
            comp._evidence = []
            return comp

        # --- 3. Retrieved-content injection referencing policy override -----
        # If the injection attempt is tied to a concrete order + action
        # request ("confirm ORD-1006 is refunded"), the on-topic safe path is
        # the action refusal; otherwise use the migration-note defense.
        injection_action_combo = (
            injection and action and extract_order_ids(message)
        )
        if injection and not injection_action_combo:
            trace["decision"] = "retrieved-injection defense"
            comp = self._kb_compose(
                query="standard return window policy current",
                original=message,
                trace=trace,
                forced_intent="injection",
            )
            return comp

        # --- 4. Malformed order ids -----------------------------------------
        attempted = _ATTEMPTED_ID_RE.search(message)
        well_formed = extract_order_ids(message)
        if attempted and not well_formed:
            trace["decision"] = "malformed order id rejection (no lookup performed)"
            comp = compose_order_malformed(attempted.group(0))
            comp._evidence = []
            return comp

        # --- 5. Action requests ---------------------------------------------
        if action and not order_q:
            kind = self._classify_action(low)
            if kind in ("cancel", "address"):
                comp = self._handle_cancellation_or_address(low, resolved, trace)
                return comp
            trace["decision"] = f"unsupported action refusal ({kind})"
            pol_chunks = self._quick_chunks("returns policy refunds price adjustments")
            comp = compose_action_refusal(kind, pol_chunks)
            comp._evidence = pol_chunks
            return comp
        if action and order_q:
            kind = self._classify_action(low)
            if kind in ("cancel", "address"):
                comp = self._handle_cancellation_or_address(low, resolved, trace)
                return comp
            # e.g. "Process a refund for ORD-1006" -> lookup for status context + refusal
            res = self._do_lookup(resolved.resolved_order_id, trace) if resolved.resolved_order_id else None
            pol_chunks = self._quick_chunks("returns refunds policy")
            comp = compose_action_refusal(kind, pol_chunks, res)
            comp._evidence = pol_chunks
            comp._tool_json = redact(res.to_dict()) if res else None
            return comp

        # --- 6. Order questions ----------------------------------------------
        if order_q:
            if resolved.needs_order_id_prompt and not resolved.resolved_order_id:
                trace["decision"] = "ask for missing order id (no lookup)"
                comp = compose_order_missing_id()
                comp._evidence = []
                return comp
            oid = resolved.resolved_order_id
            if not oid:
                trace["decision"] = "ask for missing order id (no lookup)"
                comp = compose_order_missing_id()
                comp._evidence = []
                return comp
            res = self._do_lookup(oid, trace)
            mixed_policy = self._is_mixed_return_question(low)
            if mixed_policy:
                comp = self._compose_return_eligibility(res, low, trace)
                return comp
            comp = compose_order_result(res, message)
            comp._evidence = []
            comp._tool_json = redact(res.to_dict())
            return comp

        # --- 7. Knowledge-base questions ---------------------------------------
        topic_note = ""
        if self.config.context_resolution_enabled and session.recent_topics:
            topic_note = session.recent_topics[0]
        return self._kb_compose(
            query=resolved.rewritten_query,
            original=message,
            trace=trace,
            topic_hint=topic_note,
        )

    # ------------------------------------------------------------------

    def _classify_action(self, low: str) -> str:
        if re.search(r"cancel", low):
            return "cancel"
        if re.search(r"address", low):
            return "address"
        if re.search(r"refund", low):
            return "refund"
        if re.search(r"replace|replacement|reship", low):
            return "replacement"
        if re.search(r"adjust|price match|cheaper", low):
            return "price adjustment"
        if re.search(r"approv", low):
            return "approval"
        if re.search(r"escalat|ticket", low):
            return "escalation"
        return "unsupported action"

    def _handle_cancellation_or_address(self, low: str, resolved, trace: dict) -> Composition:
        chunks = self._quick_chunks("cancel order cancellation window address change pending")
        res = None
        if resolved.resolved_order_id:
            res = self._do_lookup(resolved.resolved_order_id, trace)
        elif extract_order_ids(low):
            res = self._do_lookup(extract_order_ids(low)[-1], trace)
        comp = compose_cancellation(chunks, res)
        comp.sources = [SourceRef(file=r.filename, heading=r.chunk.primary_heading,
                                  document_id=r.chunk.meta.document_id) for r in chunks[:1]]
        comp._evidence = chunks
        comp._tool_json = redact(res.to_dict()) if res else None
        trace["decision"] = "cancellation/address-change request"
        return comp

    def _is_mixed_return_question(self, low: str) -> bool:
        return bool(re.search(
            r"(can i return|returnable|return my|return policy|return window|"
            r"exchange|send.*back|get a refund)", low))

    def _compose_return_eligibility(self, res: OrderLookupResult, low: str, trace: dict) -> Composition:
        """Combine order facts (tool) with return policy (RAG)."""
        chunks = self._quick_chunks(
            "return policy final sale damaged items trailPlus return window"
        )
        if not res.found:
            comp = compose_order_result(res, low)
            comp._evidence = chunks
            comp._tool_json = redact(res.to_dict())
            return comp

        # Status precedence: the order's current status dominates any
        # eligibility math. (Regression: a *returned* order was being run
        # through window arithmetic — bug diary #6.)
        st = res.status or ""
        if st == "returned":
            delivered = fmt_date(res.delivered_at)
            tail = f" (delivered {delivered})" if delivered else ""
            comp = Composition(
                answer=(
                    f"Order {res.order_id} has already been returned and the "
                    f"return was processed{tail}, so there's nothing left to "
                    "return. If something about the refund looks wrong, human "
                    "support can review it."
                ),
                sources=[],
                handoff=False,
            )
            comp._evidence = chunks
            comp._tool_json = redact(res.to_dict())
            return comp
        if st == "cancelled":
            comp = Composition(
                answer=(
                    f"Order {res.order_id} was cancelled and will not be "
                    "shipped, so there is nothing to return."
                ),
                sources=[],
                handoff=False,
            )
            comp._evidence = chunks
            comp._tool_json = redact(res.to_dict())
            return comp
        if st in ("pending", "processing"):
            comp = Composition(
                answer=(
                    f"Order {res.order_id} hasn't shipped yet (status: {st}). "
                    "Returns apply after delivery; if you want to stop this "
                    "order instead, cancellation is only possible within 30 "
                    "minutes of placing it and must be completed by human support."
                ),
                sources=[],
                handoff=True,
                reason="cancellation-style request on pre-shipment order needs human support",
            )
            comp._evidence = chunks
            comp._tool_json = redact(res.to_dict())
            return comp

        final_sale_items = [i for i in res.items if i.get("final_sale")]
        parts: list[str] = []
        src_chunks: list[RetrievedChunk] = []

        delivered_days = None
        if res.delivered_at and self.orders.snapshot_at:
            d0 = datetime.fromisoformat(res.delivered_at.replace("Z", "+00:00"))
            now = datetime.fromisoformat(self.orders.snapshot_at.replace("Z", "+00:00"))
            delivered_days = (now - d0).days

        if final_sale_items:
            fs = next((r for r in chunks if r.chunk.meta.document_id == "RET-2026-02"), None)
            dmg = next((r for r in chunks if r.chunk.meta.document_id == "OPS-2026-04"), None)
            names = ", ".join(i.get("name", "item") for i in final_sale_items)
            parts.append(
                f"{names} in order {res.order_id} is marked final sale in our records, so it can't be returned for a change of mind such as a different color or fit."
            )
            if fs:
                src_chunks.append(fs)
            if dmg:
                parts.append(
                    "If it arrived damaged or incorrect, that's different: final-sale items are still eligible for review under the Damaged or Wrong Items Policy within 7 calendar days of delivery."
                )
                src_chunks.append(dmg)
            comp = Composition(
                answer=" ".join(parts),
                sources=[SourceRef(file=r.filename, heading=r.chunk.primary_heading, document_id=r.chunk.meta.document_id) for r in src_chunks],
                handoff=False,
            )
        else:
            cur = next((r for r in chunks if r.chunk.meta.document_id == "RET-2026-01"), None)
            mem = next((r for r in chunks if r.chunk.meta.document_id == "MEM-2026-01"), None)
            tier = res.membership_tier or "standard"
            days = 45 if tier == "trailplus" and mem is not None else 30
            window_src = mem if (tier == "trailplus" and mem is not None) else cur
            if delivered_days is not None:
                remaining = days - delivered_days
                state = (
                    f"That leaves about {remaining} calendar days from today's records."
                    if remaining > 0
                    else "That window has already passed based on current records."
                )
                parts.append(
                    f"Order {res.order_id} was delivered {delivered_days} days ago. With your "
                    f"{'TrailPlus' if tier == 'trailplus' else 'standard'} plan the eligible-item return window is {days} calendar days from delivery. {state}"
                )
            else:
                parts.append(
                    f"With the {'TrailPlus' if tier == 'trailplus' else 'standard'} plan, eligible items can be returned within {days} calendar days of delivery."
                )
            parts.append("Items must be unused, unwashed, and in resalable condition; I can't create the return myself, but support can process an approved request.")
            if window_src:
                src_chunks.append(window_src)
            comp = Composition(
                answer=" ".join(parts),
                sources=[SourceRef(file=r.filename, heading=r.chunk.primary_heading, document_id=r.chunk.meta.document_id) for r in src_chunks],
                handoff=False,
            )
        comp._evidence = chunks
        comp._tool_json = redact(res.to_dict())
        trace["decision"] = "mixed order+policy return eligibility"
        return comp

    # ------------------------------------------------------------------
    # Knowledge-base path
    # ------------------------------------------------------------------

    INTENT_QUERIES: dict[str, str] = {
        "return_window": "standard return window calendar days of delivery condition",
        "trailplus_return": "TrailPlus membership return window active order placed",
        "trailplus_benefits": "TrailPlus membership benefits return window shipping",
        "final_sale_damaged": "final sale damaged wrong item report review resolution",
        "damaged_items": "damaged defective wrong item report resolutions photographs",
        "international_shipping": "international shipping Canada destinations delivery estimate duties taxes",
        "warranty": "limited warranty periods covered manufacturing defect",
        "gift_cards": "gift cards price adjustment final sale",
        "cancellation_policy": "cancellation window pending address change",
        "product_care": "product care cleaning washing dishwasher",
        "order_changes": "cancellation address changes pending",
    }

    def _kb_compose(self, *, query: str, original: str, trace: dict,
                    forced_intent: str | None = None, topic_hint: str | None = "") -> Composition:
        retrieval_query = query
        low = query.lower()
        # Country-aware expansion (full profile only): unsupported-country
        # questions share no vocabulary with the shipping policy beyond
        # "ship", which starves the retriever. Adding destination vocabulary
        # is deterministic query understanding, not answer injection.
        if self.config.precedence_enabled:
            if any(c in low for c in ("germany", "france", "australia", "japan",
                                      "mexico", "brazil", "india", "spain", "italy",
                                      "netherlands", "uk ", "united kingdom")):
                retrieval_query += " international shipping supported destinations Canada"
            elif "canada" in low or "internationally" in low or "international" in low:
                retrieval_query += " international delivery estimate duties taxes"

        scored = self.index.search(retrieval_query, top_k=8)

        intent = forced_intent or self._detect_intent(query)
        trace["intent"] = intent

        # Baseline (naive) profile: plain top-k similarity, no authority
        # gating, no intent templates, no query expansion, no conflict
        # detection. Answers are extractive quotes from whatever ranked
        # highest — exactly the "simple vector-search chatbot" the assignment
        # warns about. Order-tool safety subsystems stay identical so the
        # delta isolates retrieval/precedence quality.
        if not self.config.precedence_enabled:
            top = [RetrievedChunk(chunk=c, relevance=s, final_score=s) for c, s in scored[:3]]
            trace["retrieval"] = {
                "candidates": [
                    {"file": c.filename, "heading": c.primary_heading,
                     "doc_id": c.meta.document_id, "status": c.meta.status,
                     "score": round(s, 4), "selected": i < 3}
                    for i, (c, s) in enumerate(scored[:6])
                ],
                "naive": True,
            }
            comp = compose_generic_extractive(top, original)
            comp._evidence = top
            return comp

        # Dual retrieval: merge the user-query candidates with an intent
        # template search so multi-section answers stay complete while the
        # user's own words still dominate ranking (max score wins per chunk).
        template = self.INTENT_QUERIES.get(intent)
        if template:
            for chunk, s in self.index.search(template, top_k=5):
                existing = next((cs for c, cs in scored if c.chunk_id == chunk.chunk_id), None)
                if existing is None:
                    scored.append((chunk, s))
                else:
                    scored = [(c, max(cs, s) if c.chunk_id == chunk.chunk_id else cs)
                              for c, cs in scored]
            scored.sort(key=lambda pair: pair[1], reverse=True)

        result = select_evidence(
            scored,
            precedence_enabled=self.config.precedence_enabled,
            min_relevance=0.08,
            max_evidence=4,
        )
        trace["retrieval"] = {
            "candidates": [
                {
                    "file": c.filename,
                    "heading": c.primary_heading,
                    "doc_id": c.meta.document_id,
                    "status": c.meta.status,
                    "authority": c.meta.policy_authority,
                    "audience": c.meta.audience,
                    "score": round(s, 4),
                    "selected": any(x.chunk.chunk_id == c.chunk_id for x in result.selected),
                    "rejected_because": next(
                        (why for rc, why in result.rejected if rc.chunk_id == c.chunk_id), None
                    ),
                }
                for c, s in scored
            ],
            "insufficient": result.insufficient,
            "insufficiency_reason": result.insufficiency_reason,
        }

        selected: list[RetrievedChunk] = list(result.selected)
        pool: list[RetrievedChunk] = list(result.pool)

        # Same-document sibling expansion: sections of an already-vetted
        # document are added to the composer pool so multi-section answers
        # (e.g. warranty periods + review process) never hinge on lexical
        # luck. Authority was decided per DOCUMENT, so this cannot smuggle in
        # superseded/draft content.
        seen_ids = {r.chunk.chunk_id for r in pool}
        best_by_doc: dict[str, float] = {}
        for r in pool:
            best_by_doc[r.chunk.meta.document_id] = max(
                best_by_doc.get(r.chunk.meta.document_id, 0.0), r.final_score)
        for c in self.chunks:
            if c.meta.document_id in best_by_doc and c.chunk_id not in seen_ids:
                s = best_by_doc[c.meta.document_id]
                pool.append(RetrievedChunk(chunk=c, relevance=s, final_score=s))
        pool.sort(key=lambda r: r.final_score, reverse=True)
        conflicts = (
            detect_conflicts(selected, query)
            if self.config.conflicts_enabled
            else []
        )
        trace["conflicts"] = [c.topic for c in conflicts]

        if conflicts and intent != "injection":
            comp = compose_conflict(conflicts[0], original)
            comp._evidence = selected
            return comp

        if forced_intent == "injection":
            comp = compose_injection_defense(pool or selected)
            comp._evidence = selected
            return comp

        if intent == "generic":
            coverage = self._evidence_coverage(query, pool)
            trace["evidence_coverage"] = round(coverage, 3)
            if coverage < 0.6:
                trace["fallbacks"].append("low vocabulary coverage between question and evidence")
                comp = compose_insufficient(original, result.insufficiency_reason)
                comp._evidence = selected
                return comp

        if result.insufficient and intent == "generic":
            trace["fallbacks"].append(result.insufficiency_reason or "insufficient evidence")
            comp = compose_insufficient(original, result.insufficiency_reason)
            comp._evidence = selected
            return comp

        comp = self._dispatch_intent(intent, selected, original, pool)
        comp._evidence = selected
        return comp

    def _detect_intent(self, query: str) -> str:
        low = query.lower()
        has_final_sale = ("final sale" in low or "final-sale" in low)
        has_damage = any(k in low for k in ("damaged", "broken", "wrong item", "defective", "zipper"))
        if has_final_sale and has_damage:
            return "final_sale_damaged"
        if "trailplus" in low:
            if _RETURN_WINDOW_RE.search(low) or _WINDOW_Q_RE.search(low) or "window" in low:
                return "trailplus_return"
            return "trailplus_benefits"
        if "warrant" in low or "guarantee" in low or "lifetime" in low:
            return "warranty"
        if any(k in low for k in ("international", "canada", "germany", "abroad", "overseas")) or (
            ("ship" in low or "shipping" in low) and any(k in low for k in ("country", "countries"))
        ):
            return "international_shipping"
        if has_damage:
            return "damaged_items"
        if "gift card" in low:
            return "gift_cards"
        if re.search(r"price (adjust|match)|dropped|cheaper", low):
            return "gift_cards"
        if "cancel" in low or "address change" in low:
            return "cancellation_policy"
        if re.search(r"return|refund", low):
            return "return_window"
        if any(k in low for k in ("dishwasher", "hand wash", "hand-wash", "clean", "wash")):
            return "product_care"
        return "generic"

    _STOP_MINIMAL = {"the", "a", "an", "is", "are", "can", "i", "my", "to", "for",
                     "do", "does", "you", "your", "of", "in", "on", "and", "or"}

    def _evidence_coverage(self, query: str, pool: list[RetrievedChunk]) -> float:
        """Fraction of meaningful question terms present in the gated pool."""
        tokens = {
            t for t in re.findall(r"[a-z0-9]+", query.lower())
            if t not in self._STOP_MINIMAL and len(t) > 2
        }
        if not tokens:
            return 1.0
        corpus = " ".join(r.chunk.text.lower() for r in pool)
        hit = sum(1 for t in tokens if t in corpus)
        return hit / len(tokens)

    def _dispatch_intent(self, intent: str, chunks: list[RetrievedChunk],
                         original: str, pool: list[RetrievedChunk] | None = None) -> Composition:
        pool = pool or chunks
        if intent == "return_window":
            return compose_return_window(pool, original)
        if intent == "trailplus_return":
            return compose_return_window(pool, original)
        if intent == "trailplus_benefits":
            return compose_trailplus_benefits(pool, original)
        if intent == "final_sale_damaged":
            return compose_final_sale_damaged(pool)
        if intent == "damaged_items":
            return compose_damaged_items(pool)
        if intent == "international_shipping":
            return compose_international_shipping(pool, original)
        if intent == "warranty":
            return compose_warranty(pool, original)
        if intent == "gift_cards":
            return compose_gift_cards(pool, original)
        if intent == "cancellation_policy":
            return compose_cancellation(pool, None)
        if len(chunks) == 0:
            return compose_insufficient(original, "No sufficiently relevant authoritative evidence found.")
        return compose_generic_extractive(chunks, original)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _quick_chunks(self, query: str) -> list[RetrievedChunk]:
        scored = self.index.search(query, top_k=10)
        result = select_evidence(
            scored,
            precedence_enabled=self.config.precedence_enabled,
            min_relevance=0.05,
            max_evidence=6,
        )
        return list(result.pool) or list(result.selected)

    def _do_lookup(self, order_id: str | None, trace: dict) -> OrderLookupResult:
        args = {"order_id": order_id}
        try:
            res = self.orders.lookup(order_id)
            trace.setdefault("tool_calls", []).append({
                "name": "order_lookup",
                "arguments": args,
                "result": redact(res.to_dict()),
            })
            return res
        except Exception as exc:
            trace.setdefault("tool_calls", []).append({
                "name": "order_lookup",
                "arguments": args,
                "error": type(exc).__name__,
            })
            raise

    def _try_llm(self, *, question: str, history, evidence, tool_result_json, draft, trace) -> str | None:
        from .llm import phrase_with_llm

        conflict_note = "; ".join(trace.get("conflicts", [])) or None
        return phrase_with_llm(
            api_key=self.config.llm_api_key or "",
            base_url=self.config.llm_base_url or "",
            model=self.config.llm_model or "",
            timeout=self.config.llm_timeout_seconds,
            question=question,
            history=history,
            evidence=evidence,
            tool_result_json=tool_result_json,
            conflict_note=conflict_note,
            draft_answer=draft,
        )
