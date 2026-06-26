"""Risk metrics: volatility, VaR, expected shortfall, drawdown, Sharpe.

VaR and ES are returned as **positive loss magnitudes** (0.023 == a 2.3% loss).
No SciPy dependency — the Gaussian quantile comes from the stdlib's
``statistics.NormalDist``.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

_TRADING_DAYS = 252


def _as_array(returns: pd.Series | np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(returns, dtype=float)
    return arr[~np.isnan(arr)]


def annualized_vol(returns: pd.Series | np.ndarray, periods_per_year: int = _TRADING_DAYS) -> float:
    arr = _as_array(returns)
    if arr.size < 2:
        return 0.0
    return float(arr.std(ddof=1) * np.sqrt(periods_per_year))


def parametric_var(returns: pd.Series | np.ndarray, confidence: float = 0.99) -> float:
    """Gaussian (variance-covariance) 1-period VaR as a positive loss."""
    arr = _as_array(returns)
    if arr.size < 2:
        return 0.0
    mu, sigma = arr.mean(), arr.std(ddof=1)
    z = NormalDist().inv_cdf(1.0 - confidence)  # negative
    return float(-(mu + z * sigma))


def historical_var(returns: pd.Series | np.ndarray, confidence: float = 0.99) -> float:
    """Empirical VaR from the return distribution, as a positive loss."""
    arr = _as_array(returns)
    if arr.size == 0:
        return 0.0
    return float(-np.quantile(arr, 1.0 - confidence))


def expected_shortfall(returns: pd.Series | np.ndarray, confidence: float = 0.99) -> float:
    """Mean loss in the tail beyond VaR (a.k.a. CVaR), as a positive loss."""
    arr = _as_array(returns)
    if arr.size == 0:
        return 0.0
    threshold = np.quantile(arr, 1.0 - confidence)
    tail = arr[arr <= threshold]
    if tail.size == 0:
        return float(-threshold)
    return float(-tail.mean())


def sharpe(returns: pd.Series | np.ndarray, periods_per_year: int = _TRADING_DAYS) -> float:
    arr = _as_array(returns)
    if arr.size < 2 or arr.std(ddof=1) == 0:
        return 0.0
    return float(arr.mean() / arr.std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series | np.ndarray) -> float:
    """Worst peak-to-trough decline of an equity curve, as a positive fraction."""
    arr = _as_array(equity)
    if arr.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(arr)
    drawdown = arr / running_max - 1.0
    return float(-drawdown.min())
