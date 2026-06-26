"""Central configuration.

Settings are read from the environment (prefix ``EMFX_``) or a local ``.env``
file. The defaults are chosen so that ``import emfx_copilot`` and the full demo
run with no configuration at all.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EMFX_", env_file=".env", extra="ignore")

    # --- LLM ---------------------------------------------------------------
    # provider: "mock" (deterministic, offline) or "anthropic" (real Claude).
    llm_provider: str = "mock"
    anthropic_api_key: str | None = None
    llm_model: str = "claude-opus-4-8"
    llm_max_tokens: int = 4096
    # effort governs thinking depth / token spend on Opus-tier models.
    llm_effort: str = "medium"

    # --- Synthetic market --------------------------------------------------
    data_seed: int = 7
    history_days: int = 520

    # --- Risk limits (USD notional) ---------------------------------------
    gross_limit_usd: float = 100_000_000.0
    per_ccy_limit_usd: float = 25_000_000.0
    max_concentration: float = 0.40
    var_confidence: float = 0.99

    # --- Strategy ----------------------------------------------------------
    signal_weights: dict[str, float] = Field(
        default_factory=lambda: {"carry": 0.40, "momentum": 0.40, "value": 0.20}
    )
    cost_bps: float = 2.0  # round-trip transaction cost assumption for backtests

    def __hash__(self) -> int:  # allow use as an lru_cache return value
        return id(self)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-wide settings (cached)."""
    return Settings()
