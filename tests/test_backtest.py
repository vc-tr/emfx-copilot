from __future__ import annotations

import math

from emfx_copilot.backtest.engine import run_backtest
from emfx_copilot.config import Settings
from emfx_copilot.data.market_data import MarketData


def test_backtest_runs_and_metrics_are_finite():
    md = MarketData.synthetic(seed=7, n_days=420)
    result = run_backtest(md, Settings(), warmup=252)

    assert len(result.equity) == len(result.returns)
    assert (result.equity > 0).all()  # geometric equity never goes non-positive
    assert result.n_rebalances > 0
    for value in result.summary().values():
        assert isinstance(value, (int, float))
        assert math.isfinite(value)


def test_backtest_is_deterministic():
    md = MarketData.synthetic(seed=7, n_days=420)
    a = run_backtest(md, Settings(), warmup=252).summary()
    b = run_backtest(md, Settings(), warmup=252).summary()
    assert a == b


def test_backtest_requires_enough_history():
    md = MarketData.synthetic(seed=1, n_days=260)
    try:
        run_backtest(md, Settings(), warmup=252)
    except ValueError:
        return
    raise AssertionError("expected ValueError for insufficient history")
