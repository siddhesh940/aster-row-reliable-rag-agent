"""Session-scoped conversation state.

Sessions are isolated by construction: state lives in per-session ``Session``
objects held in a ``SessionStore`` keyed by explicit session IDs. Nothing is
shared between sessions except the read-only knowledge index.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

_turn_counter = itertools.count(1)


@dataclass
class Turn:
    role: str                 # "user" | "agent"
    content: str


@dataclass
class Session:
    session_id: str
    history: list[Turn] = field(default_factory=list)
    # Resolved entity memory:
    last_order_id: str | None = None
    # Bounded topic memory: most recent first, pruned to keep context relevant.
    recent_topics: list[str] = field(default_factory=list)
    turn_index: int = 0

    def add_user(self, content: str) -> None:
        self.turn_index += 1
        self.history.append(Turn("user", content))

    def add_agent(self, content: str) -> None:
        self.history.append(Turn("agent", content))

    def remember_topic(self, topic: str, max_topics: int = 3) -> None:
        if not topic:
            return
        if topic in self.recent_topics:
            self.recent_topics.remove(topic)
        self.recent_topics.insert(0, topic)
        del self.recent_topics[max_topics:]

    @property
    def recent_user_messages(self) -> list[str]:
        return [t.content for t in self.history if t.role == "user"][-4:]


@dataclass
class SessionStore:
    sessions: dict[str, Session] = field(default_factory=dict)

    def get(self, session_id: str) -> Session:
        if session_id not in self.sessions:
            self.sessions[session_id] = Session(session_id=session_id)
        return self.sessions[session_id]

    def reset(self, session_id: str | None = None) -> None:
        if session_id is None:
            self.sessions.clear()
        else:
            self.sessions.pop(session_id, None)


def next_conversation_id() -> int:
    return next(_turn_counter)
