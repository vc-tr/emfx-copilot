"""Market-data access layer.

``MarketData`` wraps a spot-price history plus a rate curve and exposes the
derived series the rest of the package needs (returns, realised vol, a
point-in-time snapshot). The synthetic constructor is the default; a real
deployment would add a ``from_provider`` classmethod that fills the same
structure from a market-data vendor — nothing downstream would change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import synthetic
from .universe import BY_CODE, CODES, USD_RATE

_TRADING_DAYS = 252


@dataclass(frozen=True)
class MarketSnapshot:
    """A point-in-time view of the desk's markets."""

    asof: pd.Timestamp
    spot: dict[str, float]  # local per USD
    local_rates: dict[str, float]  # annual decimal
    usd_rate: float
    realized_vol: dict[str, float]  # annualised, trailing window

    def carry(self, code: str) -> float:
        """Annualised long-EM carry (local rate minus USD funding)."""
        return self.local_rates[code] - self.usd_rate


class MarketData:
    """Spot history + rate curve, with no-lookahead slicing for backtests."""

    def __init__(
        self,
        spot: pd.DataFrame,
        local_rates: dict[str, float],
        usd_rate: float = USD_RATE,
    ) -> None:
        self.spot = spot.sort_index()
        self.local_rates = local_rates
        self.usd_rate = usd_rate

    # --- construction ------------------------------------------------------
    @classmethod
    def synthetic(
        cls,
        seed: int = 7,
        n_days: int = 520,
        end: pd.Timestamp | None = None,
    ) -> MarketData:
        spot = synthetic.generate_spot(seed=seed, n_days=n_days, end=end)
        rates = {code: BY_CODE[code].policy_rate for code in spot.columns}
        return cls(spot=spot, local_rates=rates, usd_rate=USD_RATE)

    @classmethod
    def from_provider(cls, *args, **kwargs) -> MarketData:  # pragma: no cover
        """Hook for a live market-data vendor (Bloomberg, Refinitiv, ...).

        Populate ``spot`` (USD/CCY), ``local_rates``, and ``usd_rate`` from the
        feed and return ``cls(...)``. Left unimplemented on purpose — the
        synthetic market keeps the project self-contained and reproducible.
        """
        raise NotImplementedError(
            "Wire a market-data provider here; the rest of the stack is feed-agnostic."
        )

    # --- derived series ----------------------------------------------------
    @property
    def codes(self) -> list[str]:
        return list(self.spot.columns)

    def usdccy_returns(self) -> pd.DataFrame:
        """Daily simple returns of USD/CCY (local per USD)."""
        return self.spot.pct_change().iloc[1:]

    def emlong_returns(self) -> pd.DataFrame:
        """Daily returns of being *long the EM currency* (i.e. short USD/CCY)."""
        return -self.usdccy_returns()

    def realized_vol(self, window: int = 20) -> pd.Series:
        """Latest annualised realised vol per currency over a trailing window."""
        rets = self.usdccy_returns().tail(window)
        return rets.std(ddof=0) * np.sqrt(_TRADING_DAYS)

    def slice_until(self, asof: pd.Timestamp) -> MarketData:
        """Return a copy containing only data up to and including ``asof``.

        Used by the backtester to guarantee signals never peek at the future.
        """
        sub = self.spot.loc[:asof]
        return MarketData(spot=sub, local_rates=self.local_rates, usd_rate=self.usd_rate)

    def latest_snapshot(self, vol_window: int = 20) -> MarketSnapshot:
        last = self.spot.index[-1]
        rv = self.realized_vol(vol_window)
        return MarketSnapshot(
            asof=last,
            spot={c: float(self.spot[c].iloc[-1]) for c in CODES if c in self.spot.columns},
            local_rates=dict(self.local_rates),
            usd_rate=self.usd_rate,
            realized_vol={c: float(rv[c]) for c in rv.index},
        )
