"""Carry analytics for long-EM positions.

Carry is the rate differential earned for being long the EM currency and short
USD funding. It is the single most important driver of EM FX total return and
the backbone of the desk's carry book.
"""

from __future__ import annotations


def carry_yield(r_local: float, r_usd: float) -> float:
    """Annualised long-EM carry = local rate - USD funding."""
    return r_local - r_usd


def expected_carry_return(r_local: float, r_usd: float, tenor_years: float) -> float:
    """Carry accrued over a holding period (ignores spot drift)."""
    return carry_yield(r_local, r_usd) * tenor_years


def carry_to_risk(carry: float, annual_vol: float) -> float:
    """Carry-to-vol ratio — a crude ex-ante Sharpe for the position."""
    return carry / annual_vol if annual_vol > 0 else 0.0
