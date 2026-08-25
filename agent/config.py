"""Configuration and environment handling.

All knobs come from environment variables with safe defaults so the system
runs from a clean clone with no credentials at all (deterministic mode).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = REPO_ROOT / "knowledge-base"
ORDERS_PATH = REPO_ROOT / "data" / "orders.json"

# Retrieval tuning.
TOP_K_CANDIDATES = 8          # candidates pulled from the vector search
MAX_EVIDENCE = 4              # passages handed to the responder
MIN_RELEVANCE = 0.08          # below this a passage is considered non-relevant
MIN_SUFFICIENT_SCORE = 0.18   # evidence below this => insufficient-evidence path

# Agent profiles.
#   full  : precedence + conflict detection + context resolution (final)
#   naive : plain top-k similarity baseline used for the baseline evaluation
PROFILES = ("full", "naive")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    profile: str = "full"
    debug: bool = False
    # LLM settings. When api_key/base_url/model are all set the agent will use
    # an OpenAI-compatible chat completion endpoint to phrase answers from the
    # deterministic evidence package; otherwise it uses the built-in grounded
    # composer (fully offline/deterministic).
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_timeout_seconds: float = 45.0
    kb_dir: Path = KB_DIR
    orders_path: Path = ORDERS_PATH

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key and self.llm_base_url and self.llm_model)

    @property
    def precedence_enabled(self) -> bool:
        return self.profile == "full"

    @property
    def conflicts_enabled(self) -> bool:
        return self.profile == "full"

    @property
    def context_resolution_enabled(self) -> bool:
        return self.profile == "full"


def load_config() -> Config:
    profile = os.environ.get("AGENT_PROFILE", "full").strip().lower()
    if profile not in PROFILES:
        profile = "full"
    return Config(
        profile=profile,
        debug=_env_bool("AGENT_DEBUG", False),
        llm_api_key=os.environ.get("LLM_API_KEY") or None,
        llm_base_url=os.environ.get("LLM_BASE_URL") or None,
        llm_model=os.environ.get("LLM_MODEL") or None,
        llm_timeout_seconds=float(os.environ.get("LLM_TIMEOUT_SECONDS", "45")),
    )
