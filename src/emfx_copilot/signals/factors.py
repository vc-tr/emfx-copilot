"""Cross-sectional EM FX factors.

Each factor returns a per-currency z-score (positive => the model wants to be
*long the EM currency*). All factors are computed point-in-time from a
``MarketData`` slice, so they are safe to call inside a walk-forward backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.market_data import MarketData


def zscore(s: pd.Series, clip: float = 3.0) -> pd.Series:
    """Cross-sectional z-score, winsorised to keep one outlier (e.g. TRY) from
    dominating the book."""
    s = s.dropna()
    std = s.std(ddof=0)
    if std < 1e-12:
        return pd.Series(0.0, index=s.index)
    return ((s - s.mean()) / std).clip(-clip, clip)


def carry_signal(md: MarketData) -> pd.Series:
    """Rank by long-EM carry (local rate - USD funding)."""
    carry = pd.Series(
        {c: md.local_rates[c] - md.usd_rate for c in md.codes}, dtype=float
    )
    return zscore(carry)


def momentum_signal(md: MarketData, lookback: int = 63, skip: int = 5) -> pd.Series:
    """Trailing total return of the long-EM position, skipping the last few days
    to avoid short-term reversal."""
    em = md.emlong_returns()
    if len(em) < lookback + skip:
        window = em
    else:
        window = em.iloc[-(lookback + skip) : len(em) - skip] if skip else em.iloc[-lookback:]
    mom = (1.0 + window).prod() - 1.0
    return zscore(mom)


def value_signal(md: MarketData, lookback: int = 252) -> pd.Series:
    """Mean-reversion: how far USD/CCY sits above its own long-run average.

    A high USD/CCY relative to history means the EM currency is historically
    *cheap*, so value wants to be long it (expecting reversion).
    """
    logs = np.log(md.spot)
    window = logs.tail(lookback)
    dev = logs.iloc[-1] - window.mean()
    return zscore(dev)


def all_signals(
    md: MarketData,
    momentum_lookback: int = 63,
    value_lookback: int = 252,
) -> dict[str, pd.Series]:
    """Compute the standard factor set in one call."""
    return {
        "carry": carry_signal(md),
        "momentum": momentum_signal(md, lookback=momentum_lookback),
        "value": value_signal(md, lookback=value_lookback),
    }
