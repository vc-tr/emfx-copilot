"""Walk-forward backtest of the blended carry/momentum/value strategy.

At each weekly rebalance the strategy recomputes factor signals from data **up to
and including that day only** (via ``MarketData.slice_until``), forms inverse-vol
target weights, and holds them until the next rebalance. Daily P&L uses the prior
day's weights against the realised long-EM return, net of turnover costs — so the
backtest is free of look-ahead by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Settings, get_settings
from ..data.market_data import MarketData
from ..risk.metrics import annualized_vol, max_drawdown, sharpe
from ..signals.blend import blend_signals, target_weights
from ..signals.factors import all_signals

_TRADING_DAYS = 252
# Require a meaningful out-of-sample window beyond warmup (~1 trading month);
# a handful of days is not a backtest.
_MIN_EVAL_DAYS = 21


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    sharpe: float
    ann_return: float
    ann_vol: float
    max_drawdown: float
    hit_rate: float
    avg_turnover: float
    n_rebalances: int

    def summary(self) -> dict[str, float | int]:
        return {
            "sharpe": round(self.sharpe, 2),
            "ann_return": round(self.ann_return, 4),
            "ann_vol": round(self.ann_vol, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "hit_rate": round(self.hit_rate, 3),
            "avg_turnover": round(self.avg_turnover, 3),
            "n_rebalances": self.n_rebalances,
            "n_days": len(self.returns),
        }


def _rebalance_dates(dates: pd.DatetimeIndex, rebalance: str) -> set[pd.Timestamp]:
    ser = pd.Series(dates, index=dates)
    weekly_last = ser.resample(rebalance).last().dropna()
    return set(pd.to_datetime(weekly_last.values))


def run_backtest(
    md: MarketData,
    settings: Settings | None = None,
    rebalance: str = "W-FRI",
    warmup: int = 252,
    cost_bps: float | None = None,
) -> BacktestResult:
    settings = settings or get_settings()
    cost_bps = settings.cost_bps if cost_bps is None else cost_bps

    dates = md.spot.index
    if len(dates) - warmup < _MIN_EVAL_DAYS:
        raise ValueError(
            f"Need at least {warmup + _MIN_EVAL_DAYS} days of history to backtest "
            f"(warmup {warmup} + {_MIN_EVAL_DAYS} out-of-sample); got {len(dates)}."
        )

    emlong = md.emlong_returns()
    eval_dates = dates[warmup:]
    rebal_dates = _rebalance_dates(eval_dates, rebalance)

    codes = md.codes
    weights = pd.DataFrame(0.0, index=eval_dates, columns=codes)
    current_w = pd.Series(0.0, index=codes)
    turnovers: list[float] = []

    for date in eval_dates:
        if date in rebal_dates:
            sub = md.slice_until(date)
            combined = blend_signals(all_signals(sub), settings.signal_weights)
            target = target_weights(combined, sub.realized_vol()).reindex(codes).fillna(0.0)
            turnovers.append(float((target - current_w).abs().sum()))
            current_w = target
        weights.loc[date] = current_w.values

    w_lag = weights.shift(1).fillna(0.0)
    gross_ret = (w_lag * emlong.reindex(eval_dates).fillna(0.0)).sum(axis=1)

    cost_series = pd.Series(0.0, index=eval_dates)
    rebal_index = [d for d in eval_dates if d in rebal_dates]
    # Align recorded turnovers to their rebalance dates.
    for d, turn in zip(rebal_index, turnovers, strict=True):
        cost_series.loc[d] = turn * cost_bps / 1e4

    net_ret = gross_ret - cost_series
    equity = (1.0 + net_ret).cumprod()

    n = len(net_ret)
    ann_return = float(equity.iloc[-1] ** (_TRADING_DAYS / n) - 1.0) if n > 0 else 0.0

    return BacktestResult(
        equity=equity,
        returns=net_ret,
        sharpe=sharpe(net_ret),
        ann_return=ann_return,
        ann_vol=annualized_vol(net_ret),
        max_drawdown=max_drawdown(equity),
        hit_rate=float((net_ret > 0).mean()),
        avg_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
        n_rebalances=len(turnovers),
    )
