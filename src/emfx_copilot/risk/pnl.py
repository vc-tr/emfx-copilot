"""Daily P&L, carry/spot attribution, and reconciliation of booking breaks.

The reconciliation helper maps to the desk's post-trade controls: it compares
the P&L the risk system *expects* against what the booking system *recorded* and
flags any difference beyond tolerance as a "break" to investigate.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..data.universe import USD_RATE

_TRADING_DAYS = 252


def daily_pnl(positions: dict[str, float], emlong_returns: pd.Series | dict[str, float]) -> dict[str, float]:
    """Per-currency USD P&L = notional * long-EM return for the day."""
    rets = pd.Series(emlong_returns, dtype=float)
    out: dict[str, float] = {}
    for code, notional in positions.items():
        r = rets.get(code, 0.0)
        out[code] = float(notional * (0.0 if pd.isna(r) else r))
    return out


def total_pnl(positions: dict[str, float], emlong_returns: pd.Series | dict[str, float]) -> float:
    return float(sum(daily_pnl(positions, emlong_returns).values()))


@dataclass(frozen=True)
class PnlAttribution:
    spot: float
    carry: float

    @property
    def total(self) -> float:
        return self.spot + self.carry

    def as_dict(self) -> dict[str, float]:
        return {"spot": round(self.spot, 2), "carry": round(self.carry, 2), "total": round(self.total, 2)}


def attribute_pnl(
    positions: dict[str, float],
    emlong_returns: pd.Series | dict[str, float],
    local_rates: dict[str, float],
    usd_rate: float = USD_RATE,
) -> PnlAttribution:
    """Split daily P&L into spot move vs one day of carry accrual."""
    rets = pd.Series(emlong_returns, dtype=float)
    spot_pnl = 0.0
    carry_pnl = 0.0
    for code, notional in positions.items():
        r = rets.get(code, 0.0)
        spot_pnl += notional * (0.0 if pd.isna(r) else r)
        daily_carry = (local_rates.get(code, usd_rate) - usd_rate) / _TRADING_DAYS
        carry_pnl += notional * daily_carry
    # The return series already embeds carry drift; treat carry as the accrual
    # component and the residual as spot. We report them separately for the desk.
    return PnlAttribution(spot=float(spot_pnl - carry_pnl), carry=float(carry_pnl))


@dataclass(frozen=True)
class PnlBreak:
    code: str
    expected: float
    booked: float

    @property
    def diff(self) -> float:
        return self.booked - self.expected

    def as_dict(self) -> dict[str, float | str]:
        return {
            "code": self.code,
            "expected": round(self.expected, 2),
            "booked": round(self.booked, 2),
            "diff": round(self.diff, 2),
        }


def reconcile(
    expected: dict[str, float],
    booked: dict[str, float],
    tolerance_usd: float = 1.0,
) -> list[PnlBreak]:
    """Flag currencies where booked P&L differs from expected beyond tolerance."""
    breaks: list[PnlBreak] = []
    for code in sorted(set(expected) | set(booked)):
        exp = expected.get(code, 0.0)
        bkd = booked.get(code, 0.0)
        if abs(bkd - exp) > tolerance_usd:
            breaks.append(PnlBreak(code=code, expected=exp, booked=bkd))
    return breaks
