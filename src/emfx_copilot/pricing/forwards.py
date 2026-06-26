"""FX forward & NDF pricing via covered interest rate parity (CIP).

All quotes follow market convention: ``local per USD`` (USDKRW, USDBRL, ...).
Short-tenor money-market (simple-interest) accrual is used, which is standard
for FX forwards out to ~1y.

For a quote S (local per USD) with local rate r_l and USD rate r_u over tenor T:

    F = S * (1 + r_l * T) / (1 + r_u * T)

If the local rate exceeds USD (a high-yielder), F > S — the currency trades at a
*forward discount*, i.e. the market prices in depreciation, the mirror image of
its positive carry.
"""

from __future__ import annotations

from dataclasses import dataclass


def cip_forward(spot: float, r_local: float, r_usd: float, tenor_years: float) -> float:
    """Outright forward (local per USD) under covered interest parity."""
    return spot * (1.0 + r_local * tenor_years) / (1.0 + r_usd * tenor_years)


def forward_points(spot: float, r_local: float, r_usd: float, tenor_years: float) -> float:
    """Forward minus spot, in price terms (local per USD)."""
    return cip_forward(spot, r_local, r_usd, tenor_years) - spot


def annualized_forward_premium(
    spot: float, r_local: float, r_usd: float, tenor_years: float
) -> float:
    """(F/S - 1) annualised. Positive => USD forward premium vs the EM ccy."""
    if tenor_years <= 0:
        return 0.0
    return (cip_forward(spot, r_local, r_usd, tenor_years) / spot - 1.0) / tenor_years


def ndf_settlement_usd(
    notional_usd: float,
    contracted_rate: float,
    fixing_rate: float,
    direction: int = 1,
) -> float:
    """Cash settlement (USD) of a non-deliverable forward at fixing.

    NDFs settle in USD against a published fixing. With quotes in local per USD:

        PnL_USD = direction * notional_usd * (fixing - contracted) / fixing

    ``direction``: +1 = long USD / short EM (gains if USD rises, i.e. fixing >
    contracted); -1 = short USD / long EM.
    """
    if fixing_rate <= 0:
        raise ValueError("fixing_rate must be positive")
    if direction not in (1, -1):
        raise ValueError("direction must be +1 (long USD) or -1 (short USD)")
    return direction * notional_usd * (fixing_rate - contracted_rate) / fixing_rate


@dataclass(frozen=True)
class ForwardQuote:
    """A fully-described forward/NDF quote for the desk."""

    code: str
    spot: float
    forward: float
    points: float
    tenor_months: int
    is_ndf: bool
    annualized_premium: float

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "spot": round(self.spot, 6),
            "forward": round(self.forward, 6),
            "points": round(self.points, 6),
            "tenor_months": self.tenor_months,
            "is_ndf": self.is_ndf,
            "annualized_premium": round(self.annualized_premium, 6),
            "instrument": "NDF" if self.is_ndf else "deliverable forward",
        }


def quote_forward(
    code: str,
    spot: float,
    r_local: float,
    r_usd: float,
    tenor_months: int,
    is_ndf: bool,
) -> ForwardQuote:
    t = tenor_months / 12.0
    return ForwardQuote(
        code=code,
        spot=spot,
        forward=cip_forward(spot, r_local, r_usd, t),
        points=forward_points(spot, r_local, r_usd, t),
        tenor_months=tenor_months,
        is_ndf=is_ndf,
        annualized_premium=annualized_forward_premium(spot, r_local, r_usd, t),
    )
