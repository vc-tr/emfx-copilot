from __future__ import annotations

import pytest

from emfx_copilot.pricing.forwards import (
    cip_forward,
    forward_points,
    ndf_settlement_usd,
    quote_forward,
)


def test_cip_forward_monotonic_in_local_rate():
    low = cip_forward(spot=1000.0, r_local=0.03, r_usd=0.05, tenor_years=1.0)
    high = cip_forward(spot=1000.0, r_local=0.10, r_usd=0.05, tenor_years=1.0)
    assert high > low


def test_forward_points_sign_matches_rate_differential():
    # local rate > USD => forward (local per USD) above spot => positive points
    assert forward_points(1000.0, r_local=0.10, r_usd=0.05, tenor_years=1.0) > 0
    # local rate < USD => negative points
    assert forward_points(1000.0, r_local=0.02, r_usd=0.05, tenor_years=1.0) < 0


def test_ndf_settlement_signs():
    # long USD (direction +1): profit when fixing > contracted
    pnl_long = ndf_settlement_usd(1_000_000, contracted_rate=1300, fixing_rate=1350, direction=1)
    assert pnl_long > 0
    # short USD (direction -1): same move is a loss
    pnl_short = ndf_settlement_usd(1_000_000, contracted_rate=1300, fixing_rate=1350, direction=-1)
    assert pnl_short < 0
    assert pnl_long == pytest.approx(-pnl_short)


def test_ndf_validation():
    with pytest.raises(ValueError):
        ndf_settlement_usd(1_000_000, 1300, fixing_rate=0.0, direction=1)
    with pytest.raises(ValueError):
        ndf_settlement_usd(1_000_000, 1300, 1350, direction=2)


def test_quote_forward_flags_ndf():
    q = quote_forward("KRW", spot=1350.0, r_local=0.035, r_usd=0.0525, tenor_months=3, is_ndf=True)
    d = q.as_dict()
    assert d["code"] == "KRW"
    assert d["is_ndf"] is True
    assert d["instrument"] == "NDF"
    assert d["forward"] > 0
