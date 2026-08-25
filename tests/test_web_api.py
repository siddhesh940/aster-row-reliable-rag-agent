"""Web API layer tests (additive; does not modify backend tests).

Starts the real stdlib server in-process on an ephemeral port and exercises
the endpoints against the real agent.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from web.server import Handler, SERVICE


@pytest.fixture(scope="module")
def base_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _post(base: str, path: str, payload) -> dict:
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as res:
        return json.loads(res.read().decode("utf-8"))


def _get(base: str, path: str):
    with urllib.request.urlopen(base + path, timeout=120) as res:
        return json.loads(res.read().decode("utf-8"))


def test_health_reports_profile(base_url):
    health = _get(base_url, "/api/health")
    assert health["status"] == "ok"
    assert health["profile"] in ("full", "naive")
    assert "api_key" not in json.dumps(health)


def test_chat_returns_sources_and_session(base_url):
    r = _post(base_url, "/api/chat",
              {"message": "What is your return policy?", "session_id": "wt1"})
    assert "30 calendar days" in r["answer"]
    assert any(s["document_id"] == "RET-2026-01" for s in r["sources"])
    assert r["session_id"] == "wt1"
    assert r["used_order_tool"] is False


def test_chat_order_payload_is_sanitized(base_url):
    r = _post(base_url, "/api/chat",
              {"message": "Where is ORD-1007?", "session_id": "wt2"})
    assert r["used_order_tool"] is True
    order = r["order"]
    assert order["found"] is True
    forbidden = {"email", "address", "risk_score", "internal_notes",
                 "warehouse_note", "support_tags", "customer"}
    assert not (set(order.keys()) & forbidden)


def test_reset_clears_context(base_url):
    _post(base_url, "/api/chat",
          {"message": "Where is ORD-1007?", "session_id": "wt3"})
    assert _post(base_url, "/api/session/reset", {"session_id": "wt3"})["ok"]
    r = _post(base_url, "/api/chat",
              {"message": "When will it arrive?", "session_id": "wt3"})
    assert "ORD-1007" not in r["answer"]


def test_malformed_body_rejected(base_url):
    req = urllib.request.Request(
        base_url + "/api/chat", data=b"not-json",
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(req, timeout=10)
    assert excinfo.value.code == 400


def test_evaluation_endpoint_real_results(base_url):
    ev = _get(base_url, "/api/eval")
    overall = ev["summary"]["overall"]
    assert overall["total"] == 29
    assert overall["passed"] == 29
    assert len(ev["cases"]) == 29


def test_static_index_served(base_url):
    with urllib.request.urlopen(base_url + "/", timeout=10) as res:
        html = res.read().decode("utf-8")
    assert "Aster" in html and "app.js" in html
