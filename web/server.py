"""Aster & Row support agent — local web UI server (stdlib only).

Serves the static frontend from web/static/ and exposes a minimal JSON API
that wraps the existing SupportAgent. The agent remains the single source of
truth: no answers, citations, order data, handoff states, or evaluation
results are produced by this server beyond what the agent itself returns.

Usage:
    python web/server.py            # http://127.0.0.1:8000
    python web/server.py --port 8080
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.agent import SupportAgent            # noqa: E402
from agent.config import load_config            # noqa: E402
from evaluation.run import (                    # noqa: E402
    load_cases,
    run_case,
    summarize,
)

STATIC_DIR = ROOT / "web" / "static"
MAX_MESSAGE_CHARS = 2000


class AgentService:
    """Thread-safe wrapper around one shared SupportAgent instance."""

    def __init__(self) -> None:
        self.config = load_config()
        self.agent = SupportAgent(self.config)
        self._lock = threading.Lock()
        self._eval_cache: dict | None = None
        self._eval_lock = threading.Lock()

    def chat(self, message: str, session_id: str | None, include_debug: bool) -> dict:
        with self._lock:
            resp = self.agent.handle(message, session_id=session_id)
        payload = {
            "answer": resp.answer,
            "sources": [s.to_dict() for s in resp.sources],
            "handoff": resp.handoff,
            "reason": resp.reason,
            "conflict_detected": resp.conflict_detected,
            "abstained": resp.abstained,
            "session_id": resp.session_id,
            "used_order_tool": resp.used_order_tool,
            # The sanitized order card is the backend's own redacted tool
            # result, re-exposed verbatim. No field is added client- or
            # server-side beyond what the order tool whitelisted.
            "order": next(
                (
                    t.get("result")
                    for t in reversed(resp.tool_calls)
                    if t.get("name") == "order_lookup" and isinstance(t.get("result"), dict)
                ),
                None,
            ),
            "debug": resp.debug if include_debug else None,
        }
        return payload

    def reset_session(self, session_id: str | None) -> dict:
        with self._lock:
            if session_id:
                self.agent.sessions.reset(session_id)
            else:
                self.agent.sessions.reset()
        return {"ok": True}

    def health(self) -> dict:
        return {
            "status": "ok",
            "profile": self.config.profile,
            "llm_phrasing_enabled": self.config.llm_enabled,
        }

    def evaluation(self) -> dict:
        """Run the REAL evaluation suite in-process (cached after first run).

        The CLI (`python -m evaluation.run`) remains the authoritative entry
        point; this endpoint surfaces the same real results for the dev panel.
        """
        with self._eval_lock:
            if self._eval_cache is not None:
                return self._eval_cache
            eval_agent = SupportAgent(load_config())
            reports = [run_case(eval_agent, case) for case in load_cases()]
            summary = summarize(reports)
            self._eval_cache = {
                "summary": summary,
                "cases": [
                    {
                        "id": r.id,
                        "category": r.category,
                        "passed": r.passed,
                        "failed_checks": r.failed_checks,
                    }
                    for r in reports
                ],
                "note": "Run via the CLI (python -m evaluation.run) for the authoritative report.",
            }
            return self._eval_cache


SERVICE = AgentService()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "AsterRowAgent/1.0"

    # -- helpers -------------------------------------------------------

    def _send_json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 64_000:
            raise ValueError("invalid request body size")
        raw = self.rfile.read(length)
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON object expected")
        return data

    def _serve_static(self, rel_path: str) -> None:
        if rel_path in ("", "/"):
            rel_path = "/index.html"
        target = (STATIC_DIR / rel_path.lstrip("/")).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
            self._send_json({"error": "not found"}, status=404)
            return
        content_types = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            ".svg": "image/svg+xml",
            ".png": "image/png",
            ".ico": "image/x-icon",
        }
        body = target.read_bytes()
        self.send_response(200)
        self.send_header(
            "Content-Type", content_types.get(target.suffix, "application/octet-stream")
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    # -- routes --------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        try:
            if path == "/api/health":
                self._send_json(SERVICE.health())
            elif path == "/api/eval":
                self._send_json(SERVICE.evaluation())
            elif path.startswith("/api/"):
                self._send_json({"error": "not found"}, status=404)
            else:
                self._serve_static(path)
        except Exception:
            self._send_json({"error": "internal error"}, status=500)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        try:
            if path == "/api/chat":
                data = self._read_json_body()
                message = str(data.get("message") or "").strip()
                if not message:
                    self._send_json({"error": "message is required"}, status=400)
                    return
                if len(message) > MAX_MESSAGE_CHARS:
                    message = message[:MAX_MESSAGE_CHARS]
                session_id = data.get("session_id") or None
                if session_id is not None:
                    session_id = str(session_id)[:64]
                include_debug = bool(data.get("debug"))
                self._send_json(SERVICE.chat(message, session_id, include_debug))
            elif path == "/api/session/reset":
                data = self._read_json_body()
                sid = data.get("session_id")
                self._send_json(SERVICE.reset_session(str(sid)[:64] if sid else None))
            else:
                self._send_json({"error": "not found"}, status=404)
        except (ValueError, json.JSONDecodeError):
            self._send_json({"error": "malformed request"}, status=400)
        except Exception:
            self._send_json({"error": "internal error"}, status=500)

    def log_message(self, fmt: str, *args) -> None:  # quiet access log
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Aster & Row support agent web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Aster & Row support agent UI:  http://{args.host}:{args.port}")
    print(f"profile={SERVICE.config.profile}  llm_phrasing={SERVICE.config.llm_enabled}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
