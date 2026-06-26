from __future__ import annotations

import pytest

from emfx_copilot.agent.tools import CopilotContext
from emfx_copilot.config import Settings
from emfx_copilot.data.market_data import MarketData
from emfx_copilot.nlp.llm import MockLLM


@pytest.fixture(scope="session")
def md() -> MarketData:
    return MarketData.synthetic(seed=7, n_days=420)


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def ctx(md: MarketData, settings: Settings) -> CopilotContext:
    return CopilotContext(market=md, llm=MockLLM(), settings=settings)
