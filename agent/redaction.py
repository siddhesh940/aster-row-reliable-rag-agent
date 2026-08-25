"""Redaction helpers for logs and debug traces.

Debug mode must never leak sensitive order data, credentials, or secrets even
though it prints tool results and evidence. Redaction is recursive and runs on
every structure before it is logged.
"""

from __future__ import annotations

import re
from typing import Any

SENSITIVE_KEY_PARTS = (
    "email",
    "address",
    "customer",   # raw order records: redact the whole customer subtree in logs
    "internal",
    "risk_score",
    "warehouse_note",
    "support_tags",
    "api_key",
    "secret",
    "token",
    "password",
    "authorization",
)
# NOTE: bare "name" is intentionally NOT blanket-sensitive here:
# the order tool already enforces a whitelist before anything reaches logs, and
# treating every "name" key as secret corrupted structured trace entries
# (bug diary #4). The recursive scrub below remains defense-in-depth for the
# fields that must never be logged regardless of origin.

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_ADDRESS_HINT_RE = re.compile(r"\b\d{1,5}\s+[A-Z][\w ]+(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b", re.IGNORECASE)


def redact_text(text: str) -> str:
    text = _EMAIL_RE.sub("[redacted-email]", text)
    text = _ADDRESS_HINT_RE.sub("[redacted-address]", text)
    return text


def _key_is_sensitive(key: str) -> bool:
    low = key.lower()
    return any(part in low for part in SENSITIVE_KEY_PARTS)


def redact(obj: Any) -> Any:
    """Recursively redact sensitive values from arbitrary structures."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if _key_is_sensitive(str(k)):
                if isinstance(v, (dict, list)):
                    out[str(k)] = "[redacted]"
                else:
                    out[str(k)] = "[redacted]"
            else:
                out[str(k)] = redact(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [redact(x) for x in obj]
    if isinstance(obj, str):
        return redact_text(obj)
    return obj
