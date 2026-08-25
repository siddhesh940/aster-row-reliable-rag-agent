"""Shared fixtures: one deterministic agent for the whole test session."""

from __future__ import annotations

import pytest

from agent.agent import SupportAgent
from agent.config import Config


@pytest.fixture(scope="session")
def agent() -> SupportAgent:
    return SupportAgent(Config(profile="full"))


@pytest.fixture()
def naive_agent() -> SupportAgent:
    return SupportAgent(Config(profile="naive"))
