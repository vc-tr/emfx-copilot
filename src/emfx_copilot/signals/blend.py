"""Blend factor signals into a tradeable book.

The combined score is a weighted average of factor z-scores. Target *weights*
are proportional to the combined score with an inverse-vol tilt (so the book
doesn't load all its risk on the most volatile names) and are normalised to unit
gross. ``to_notional`` turns weights into USD notionals subject to desk limits.
"""

from __future__ import annotations

import pandas as pd


def blend_signals(signals: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    """Weighted average of factor z-scores -> a single score per currency."""
    df = pd.DataFrame(signals).fillna(0.0)
    w = pd.Series(weights, dtype=float)
    w = w[w.index.intersection(df.columns)]
    denom = w.abs().sum() or 1.0
    return (df[w.index] * w).sum(axis=1) / denom


def target_weights(combined: pd.Series, vol: pd.Series) -> pd.Series:
    """Inverse-vol-tilted weights, normalised to unit gross (sum |w| == 1)."""
    vol = vol.reindex(combined.index)
    vol = vol.fillna(vol.median() if vol.notna().any() else 1.0)
    raw = combined / (vol + 1e-9)
    gross = raw.abs().sum()
    if gross < 1e-12:
        return pd.Series(0.0, index=combined.index)
    return raw / gross


def to_notional(
    weights: pd.Series,
    gross_limit_usd: float,
    per_ccy_limit_usd: float,
) -> pd.Series:
    """Scale unit-gross weights to USD notionals and clip per-currency limits.

    Positive notional = long the EM currency (short USD).
    """
    notional = weights * gross_limit_usd
    return notional.clip(-per_ccy_limit_usd, per_ccy_limit_usd)


def top_signals(combined: pd.Series, n: int = 3) -> tuple[list[str], list[str]]:
    """Return (top longs, top shorts) by combined score."""
    ranked = combined.sort_values(ascending=False)
    longs = list(ranked.head(n).index)
    shorts = list(ranked.tail(n).index[::-1])
    return longs, shorts
