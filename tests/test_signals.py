from __future__ import annotations

import pytest

from emfx_copilot.signals.blend import blend_signals, target_weights, to_notional, top_signals
from emfx_copilot.signals.factors import all_signals, carry_signal, zscore


def test_zscore_is_standardised():
    import pandas as pd

    z = zscore(pd.Series({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}))
    assert abs(float(z.mean())) < 1e-9


def test_carry_signal_ranks_highest_yielder_top(md):
    carry = carry_signal(md)
    # TRY carries by far the highest policy rate in the universe.
    assert carry.idxmax() == "TRY"


def test_target_weights_unit_gross(md):
    sigs = all_signals(md)
    combined = blend_signals(sigs, {"carry": 0.4, "momentum": 0.4, "value": 0.2})
    weights = target_weights(combined, md.realized_vol())
    assert float(weights.abs().sum()) == pytest.approx(1.0, abs=1e-9)


def test_top_signals_shapes(md):
    combined = blend_signals(all_signals(md), {"carry": 0.5, "momentum": 0.5})
    longs, shorts = top_signals(combined, n=3)
    assert len(longs) == 3 and len(shorts) == 3
    assert set(longs).isdisjoint(shorts)


def test_to_notional_respects_per_ccy_limit(md):
    combined = blend_signals(all_signals(md), {"carry": 1.0})
    weights = target_weights(combined, md.realized_vol())
    notional = to_notional(weights, gross_limit_usd=100e6, per_ccy_limit_usd=10e6)
    assert notional.abs().max() <= 10e6 + 1e-6
