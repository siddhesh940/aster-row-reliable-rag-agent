"""Order tool: normalization, validation, sanitization, status precedence."""

from __future__ import annotations

import pytest

from agent.orders import OrderTool, normalize_order_id, validate_order_id

DATA = __import__("pathlib").Path(__file__).resolve().parent.parent / "data" / "orders.json"


@pytest.fixture(scope="module")
def tool():
    return OrderTool(DATA)


# --- normalization ---------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("ord-1007", "ORD-1007"),
    ("  ORD-1007  ", "ORD-1007"),
    ("ORD-1007.", "ORD-1007"),
    ("'ord-1007'", "ORD-1007"),
    ("ord - 1007", "ORD-1007"),
    ("Ord–1007", "ORD-1007"),
])
def test_harmless_normalization(raw, expected):
    assert normalize_order_id(raw) == expected


@pytest.mark.parametrize("bad", ["", None, "ORD12", "ORD-", "order", "ORD-12345", "#1004"])
def test_malformed_ids_rejected(bad):
    normalized = normalize_order_id(bad)
    ok, _ = validate_order_id(normalized)
    assert not ok
    res = OrderTool(DATA).lookup(bad)
    assert res.found is False and res.error


# --- lookups ----------------------------------------------------------------

def test_valid_shipped_order_with_eta(tool):
    res = tool.lookup("ORD-1007")
    assert res.found and res.status == "shipped"
    assert res.carrier == "UPS"
    assert res.estimated_delivery == "2026-08-22"


def test_lowercase_whitespace_lookup_matches(tool):
    a = tool.lookup("ord-1007")
    b = tool.lookup("  ORD-1007 ")
    assert a.found and b.found
    assert a.order_id == b.order_id == "ORD-1007"


def test_unknown_order_safe_result(tool):
    res = tool.lookup("ORD-9999")
    assert res.found is False
    assert res.status is None and res.carrier is None


def test_never_guesses_nearby_id(tool):
    res = tool.lookup("ORD-9998")
    assert res.found is False


# --- privacy boundary --------------------------------------------------------

FORBIDDEN_VALUES = [
    "maya.reed@example.test", "noah.kim@example.test", "olivia.chen@example.test",
    "ethan.brooks@example.test", "sofia.patel@example.test", "liam.jones@example.test",
    "ava.morgan@example.test", "lucas.green@example.test", "isabella.stone@example.test",
    "henry.diaz@example.test", "emma.wilson@example.test", "james.taylor@example.test",
    "18 Cedar Lane", "220 King Street", "1010 Robson Street",
]


def test_sanitized_result_has_no_sensitive_keys_or_values(tool):
    for oid in [f"ORD-{i}" for i in range(1001, 1013)]:
        raw = tool.lookup(oid)
        d = raw.to_dict()
        blob = str(d).lower()
        for forbidden in FORBIDDEN_VALUES:
            assert forbidden.lower() not in blob, f"{oid} leaked {forbidden}"
        for key in ("email", "shipping_address", "risk_score", "warehouse_note",
                    "support_tags", "internal", "customer"):
            assert key not in d, f"{oid} exposed {key}"


def test_cancelled_order_stale_fields_suppressed(tool):
    res = tool.lookup("ORD-1004")
    assert res.found and res.status == "cancelled"
    assert res.carrier is None and res.tracking_number is None
    assert res.estimated_delivery is None
    assert res.stale_fields_suppressed is True


def test_returned_order_stale_fields_suppressed(tool):
    res = tool.lookup("ORD-1008")
    assert res.status == "returned"
    assert res.carrier is None and res.estimated_delivery is None


def test_shipped_without_eta_stays_null(tool):
    res = tool.lookup("ORD-1011")
    assert res.status == "shipped"
    assert res.estimated_delivery is None


def test_delayed_order_internal_note_never_exposed(tool):
    """The warehouse note contains an injected AI instruction; it must never
    appear anywhere in the sanitized result."""
    import json
    res = tool.lookup("ORD-1005")
    blob = json.dumps(res.to_dict()).lower()
    assert "coupon" not in blob
    assert "hide" not in blob
    assert "ai instruction" not in blob
