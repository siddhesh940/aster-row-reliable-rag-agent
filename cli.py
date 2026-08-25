"""Aster & Row support agent CLI.

Usage:
    python cli.py                # interactive chat
    python cli.py --debug        # show sanitized traces after each turn
    python cli.py --session X    # explicit session id
"""

from __future__ import annotations

import argparse
import json
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (ValueError, OSError):
        pass

from agent.agent import SupportAgent
from agent.config import load_config
from agent.redaction import redact

BANNER = r"""
==============================================================
  Aster & Row — Customer Support Agent
  Ask about policies, shipping, returns, or your order.
  Commands:  /debug  /new  /quit
=============================================================="""


def render(resp) -> None:
    print()
    print(f"Agent: {resp.answer}")
    if resp.sources:
        print("\nSources:")
        for s in resp.sources:
            print(f"  - {s.file} | {s.heading} ({s.document_id})")
    elif resp.used_order_tool or resp.tool_calls:
        print("\nSources: live order lookup (order system record)")
    if resp.conflict_detected:
        print("!! Conflicting active sources detected — see answer.")
    if resp.handoff:
        reason = f" ({resp.reason})" if resp.reason else ""
        print(f"[HUMAN HANDOFF RECOMMENDED{reason}]")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Aster & Row support agent")
    parser.add_argument("--debug", action="store_true", help="show sanitized trace per turn")
    parser.add_argument("--session", default=None, help="explicit session id")
    args = parser.parse_args()

    config = load_config()
    config.debug = config.debug or args.debug
    agent = SupportAgent(config)
    session_id = args.session or "cli-session"

    mode = []
    if config.llm_enabled:
        mode.append(f"LLM phrasing via {config.llm_model}")
    else:
        mode.append("deterministic grounded composer (set LLM_* env vars for LLM phrasing)")
    if config.profile == "naive":
        mode.append("BASELINE PROFILE (precedence/conflicts/context OFF)")

    print(BANNER)
    print(f"Session: {session_id} | Profile: {config.profile} | {' | '.join(mode)}")

    debug_on = config.debug
    while True:
        try:
            user = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return 0
        if not user:
            continue
        cmd = user.lower()
        if cmd in ("/quit", "/exit", "/q"):
            print("Goodbye!")
            return 0
        if cmd == "/debug":
            debug_on = not debug_on
            print(f"Debug traces: {'ON' if debug_on else 'OFF'}")
            continue
        if cmd == "/new":
            session_id = f"sess-{__import__('uuid').uuid4().hex[:10]}"
            print(f"Started new session: {session_id}")
            continue

        resp = agent.handle(user, session_id=session_id)
        render(resp)
        if debug_on:
            print("---- DEBUG TRACE (sanitized) ----")
            safe = redact({
                "user_message": resp.debug.get("user_message"),
                "resolved_query": resp.debug.get("resolved_query"),
                "context_notes": resp.debug.get("context_notes"),
                "recent_history": resp.debug.get("recent_history"),
                "decision": resp.debug.get("decision"),
                "retrieval": resp.debug.get("retrieval"),
                "conflicts": resp.debug.get("conflicts"),
                "tool_calls": resp.tool_calls,
                "errors": resp.debug.get("errors"),
                "fallbacks": resp.debug.get("fallbacks"),
                "used_llm": resp.debug.get("used_llm"),
            })
            print(json.dumps(safe, indent=2, ensure_ascii=False))
            print("---------------------------------")


if __name__ == "__main__":
    sys.exit(main())
