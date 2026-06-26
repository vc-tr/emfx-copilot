from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from emfx_copilot.risk.exposure import PositionBook, RiskLimits, check_limits
from emfx_copilot.risk.metrics import (
    expected_shortfall,
    historical_var,
    max_drawdown,
    parametric_var,
)
from emfx_copilot.risk.pnl import attribute_pnl, daily_pnl, reconcile


def test_var_is_positive_loss():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0, 0.01, 1000)
    assert parametric_var(rets, 0.99) > 0
    assert historical_var(rets, 0.99) > 0
    # ES is at least as severe as VaR
    assert expected_shortfall(rets, 0.99) >= historical_var(rets, 0.99) - 1e-9


def test_max_drawdown_known_series():
    equity = pd.Series([1.0, 1.10, 0.99, 1.20])
    # peak 1.10 -> trough 0.99 => drawdown ~ 0.10
    assert max_drawdown(equity) == pytest.approx(1 - 0.99 / 1.10, abs=1e-9)


def test_position_book_aggregations():
    book = PositionBook({"BRL": 10e6, "KRW": -5e6, "MXN": 5e6})
    assert book.gross_usd() == pytest.approx(20e6)
    assert book.net_usd() == pytest.approx(10e6)
    assert book.concentration() == pytest.approx(0.5)


def test_check_limits_flags_breaches():
    book = PositionBook({"BRL": 30e6, "KRW": 5e6})
    limits = RiskLimits(gross_limit_usd=100e6, per_ccy_limit_usd=25e6, max_concentration=0.4)
    breaches = check_limits(book, limits)
    kinds = {b.kind for b in breaches}
    assert "per_ccy" in kinds  # BRL exceeds 25mm
    assert "concentration" in kinds  # BRL is ~86% of gross


def test_daily_pnl_and_attribution():
    positions = {"BRL": 10e6, "KRW": -5e6}
    rets = {"BRL": 0.01, "KRW": -0.02}
    pnl = daily_pnl(positions, rets)
    assert pnl["BRL"] == pytest.approx(100_000)
    assert pnl["KRW"] == pytest.approx(100_000)
    attr = attribute_pnl(positions, rets, local_rates={"BRL": 0.105, "KRW": 0.035})
    assert attr.total == pytest.approx(sum(pnl.values()), rel=1e-6)


def test_reconcile_detects_break():
    expected = {"BRL": 100.0, "KRW": 50.0}
    booked = {"BRL": 100.0, "KRW": 80.0}
    breaks = reconcile(expected, booked, tolerance_usd=1.0)
    assert len(breaks) == 1
    assert breaks[0].code == "KRW"
    assert breaks[0].diff == pytest.approx(30.0)
