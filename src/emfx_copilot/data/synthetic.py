"""A reproducible synthetic EM FX market.

We generate daily USD/CCY spot paths driven by:

* a global **risk factor** (AR(1) with stochastic volatility) — the risk-on /
  risk-off regime that drives cross-asset EM moves;
* per-currency **beta** to that factor (high-beta: ZAR, BRL, TRY; low: TWD, CNH);
* a small **carry drift** so that high-yielders earn part of their rate
  differential (which gives the carry factor something to find); and
* idiosyncratic vol scaled to each currency's typical realised volatility.

Everything is seeded, so the market — and therefore every signal, backtest, and
briefing — is byte-for-byte reproducible. This stands in for a real market-data
feed; ``MarketData`` documents where a live provider would plug in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .universe import UNIVERSE

# Reference spot levels (local per USD), roughly current order of magnitude.
REF_SPOT: dict[str, float] = {
    "KRW": 1350.0, "TWD": 32.0, "INR": 83.0, "IDR": 16000.0, "PHP": 57.0,
    "VND": 25000.0, "CNH": 7.20, "THB": 36.0, "BRL": 5.40, "MXN": 18.0,
    "CLP": 950.0, "COP": 4000.0, "ZAR": 18.5, "TRY": 32.0, "PLN": 4.0, "HUF": 360.0,
}

# Sensitivity to the global risk factor (risk-off => EM sells off).
BETA: dict[str, float] = {
    "KRW": 1.05, "TWD": 0.70, "INR": 0.55, "IDR": 0.90, "PHP": 0.65,
    "VND": 0.40, "CNH": 0.60, "THB": 0.80, "BRL": 1.40, "MXN": 1.30,
    "CLP": 1.20, "COP": 1.25, "ZAR": 1.55, "TRY": 1.70, "PLN": 1.10, "HUF": 1.15,
}

# Typical annualised volatility of the pair.
ANN_VOL: dict[str, float] = {
    "KRW": 0.095, "TWD": 0.060, "INR": 0.055, "IDR": 0.075, "PHP": 0.060,
    "VND": 0.035, "CNH": 0.050, "THB": 0.080, "BRL": 0.160, "MXN": 0.140,
    "CLP": 0.150, "COP": 0.155, "ZAR": 0.175, "TRY": 0.220, "PLN": 0.110, "HUF": 0.120,
}

_TRADING_DAYS = 252


def generate_spot(
    seed: int = 7,
    n_days: int = 520,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return a wide DataFrame of USD/CCY spot (index=business days, cols=codes)."""
    rng = np.random.default_rng(seed)
    end = (end or pd.Timestamp.today()).normalize()
    dates = pd.bdate_range(end=end, periods=n_days)

    # Global risk factor with stochastic volatility (log-AR(1) vol).
    rf = np.zeros(n_days)
    log_vol = 0.0
    for t in range(1, n_days):
        log_vol = 0.95 * log_vol + 0.20 * rng.standard_normal()
        shock = np.exp(0.5 * log_vol) * rng.standard_normal()
        rf[t] = 0.93 * rf[t - 1] + shock
    rf = (rf - rf.mean()) / (rf.std() + 1e-9)  # standardise

    columns: dict[str, np.ndarray] = {}
    for c in UNIVERSE:
        s0 = REF_SPOT[c.code]
        beta = BETA[c.code]
        sig_d = ANN_VOL[c.code] / np.sqrt(_TRADING_DAYS)
        carry = c.policy_rate - 0.0525  # vs USD funding

        # "EM-long" daily return = being long the EM ccy (short USD/CCY).
        # Capture ~half the carry as drift; load on the risk factor; add noise.
        mu_emlong = (carry / _TRADING_DAYS) * 0.5
        idio = rng.standard_normal(n_days)
        emlong_ret = mu_emlong + beta * 0.0045 * rf + sig_d * idio

        # USD/CCY log-return is the negative of the EM-long return.
        usdccy_logret = -emlong_ret
        usdccy_logret[0] = 0.0
        columns[c.code] = s0 * np.exp(np.cumsum(usdccy_logret))

    return pd.DataFrame(columns, index=dates)
