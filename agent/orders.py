"""Order lookup tool.

Security properties:
* The LLM never sees ``orders.json`` — only the sanitized result of a specific
  lookup, and only when the orchestrator decides order data is actually needed.
* Sanitization is a whitelist enforced *inside the tool*, before anything is
  returned: customer name/email/address and everything under ``internal``
  (risk score, warehouse notes, support tags) cannot survive the boundary.
* Status precedence: ``status`` is authoritative. Stale carrier/tracking/ETA
  fields are suppressed for cancelled/returned orders. Missing ETA stays
  missing — nothing is ever invented.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from .contracts import (
    ORDER_ID_RE,
    OrderLookupResult,
)

# Statuses for which operational carrier/ETA fields are considered stale.
_STALE_STATUSES = {"cancelled", "returned"}

# Whitelist of customer-safe fields copied out of the raw record.
_SAFE_ITEM_FIELDS = ("name", "quantity", "final_sale")

_ID_CLEAN_RE = re.compile(r"[\s'\"]+")


def normalize_order_id(raw: str | None) -> str:
    """Normalize harmless input differences without ever guessing.

    Handles surrounding whitespace/quotes/punctuation, case differences, and
    spacing around the dash (``ord - 1007``). Returns the cleaned candidate ID;
    validity itself is checked by :func:`validate_order_id`.
    """
    if not raw:
        return ""
    cleaned = _ID_CLEAN_RE.sub("", raw.strip())
    # Drop trailing punctuation that commonly follows IDs in prose.
    cleaned = cleaned.strip(".,;:!?)(")
    cleaned = re.sub(r"[-–—]+", "-", cleaned).upper()
    return cleaned


def validate_order_id(normalized: str) -> tuple[bool, str | None]:
    if not normalized:
        return False, "no order id provided"
    if not ORDER_ID_RE.fullmatch(normalized):
        return False, f"'{normalized}' is not a well-formed Aster & Row order id (expected format ORD-1234)"
    return True, None


class OrderTool:
    def __init__(self, orders_path) -> None:
        self._path = orders_path

    @lru_cache(maxsize=1)
    def _dataset(self) -> dict:
        with open(self._path, encoding="utf-8") as fh:
            return json.load(fh)

    @property
    def snapshot_at(self) -> str:
        return self._dataset().get("snapshot_at", "")

    def lookup(self, raw_id: str | None) -> OrderLookupResult:
        """Perform a real lookup; never claims success it did not have."""
        normalized = normalize_order_id(raw_id)
        ok, err = validate_order_id(normalized)
        if not ok:
            return OrderLookupResult(found=False, error=err)

        for record in self._dataset().get("orders", []):
            if record.get("order_id") == normalized:
                return self._sanitize(record)
        return OrderLookupResult(
            found=False,
            order_id=normalized,
            error="no order with this id exists in the current records",
        )

    # -- sanitization ------------------------------------------------------

    def _sanitize(self, record: dict) -> OrderLookupResult:
        status = str(record.get("status", "")).lower()
        stale = status in _STALE_STATUSES

        items = [
            {k: item.get(k) for k in _SAFE_ITEM_FIELDS}
            for item in record.get("items", [])
        ]

        result = OrderLookupResult(
            found=True,
            order_id=record.get("order_id"),
            status=status or None,
            membership_tier=record.get("membership_tier"),
            items=items,
            placed_at=record.get("placed_at"),
            delivered_at=record.get("delivered_at"),
            # Stale logistics fields are suppressed at the source so neither
            # the model nor any log can see them.
            carrier=None if stale else record.get("carrier"),
            tracking_number=None if stale else record.get("tracking_number"),
            estimated_delivery=None if stale else record.get("estimated_delivery"),
            customer_safe_message=record.get("customer_safe_message"),
            stale_fields_suppressed=stale,
        )
        return result


def extract_order_ids(text: str) -> list[str]:
    """Extract plausible order-ID mentions from free text (normalized)."""
    from .contracts import ORDER_ID_SCAN_RE

    found = []
    for match in ORDER_ID_SCAN_RE.findall(text):
        normalized = re.sub(r"[\s]+", "", match)
        normalized = re.sub(r"[-–—]+", "-", normalized).upper()
        found.append(normalized)
    return found
