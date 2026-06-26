"""Risk-on / risk-off regime detection for EM FX.

Features are built from an equally-weighted EM-long index: its trailing return,
its realised volatility, and the cross-sectional dispersion of moves (broad,
correlated selloffs are the signature of risk-off). A 2-state Gaussian Mixture
labels each day; the higher-volatility / lower-return cluster is mapped to
"risk-off". If scikit-learn is unavailable the detector degrades gracefully to a
transparent rule based on the same features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.market_data import MarketData

_TRADING_DAYS = 252


@dataclass(frozen=True)
class RegimeResult:
    label: str  # "risk-on" | "risk-off"
    risk_off_prob: float
    index_return_20d: float
    index_vol: float
    method: str  # "gmm" | "rule"

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "risk_off_prob": round(self.risk_off_prob, 3),
            "index_return_20d": round(self.index_return_20d, 4),
            "index_vol": round(self.index_vol, 4),
            "method": self.method,
        }


def _features(md: MarketData, vol_window: int = 20) -> pd.DataFrame:
    em = md.emlong_returns()
    index = em.mean(axis=1)  # equally-weighted EM-long index daily return
    feat = pd.DataFrame(
        {
            "ret": index.rolling(vol_window).mean(),
            "vol": index.rolling(vol_window).std(ddof=0) * np.sqrt(_TRADING_DAYS),
            "dispersion": em.std(axis=1, ddof=0),
        }
    ).dropna()
    return feat


class RegimeDetector:
    """2-state regime model over EM-index return / vol / dispersion features."""

    def __init__(self, vol_window: int = 20, random_state: int = 0) -> None:
        self.vol_window = vol_window
        self.random_state = random_state

    def detect(self, md: MarketData) -> RegimeResult:
        feat = _features(md, self.vol_window)
        if feat.empty:
            return RegimeResult("risk-on", 0.5, 0.0, 0.0, "rule")

        index_ret = float(feat["ret"].iloc[-1])
        index_vol = float(feat["vol"].iloc[-1])

        gmm_result = self._gmm(feat, index_ret, index_vol)
        if gmm_result is not None:
            return gmm_result
        return self._rule(feat, index_ret, index_vol)

    def _gmm(
        self, feat: pd.DataFrame, index_ret: float, index_vol: float
    ) -> RegimeResult | None:
        try:
            from sklearn.mixture import GaussianMixture
            from sklearn.preprocessing import StandardScaler
        except ImportError:  # pragma: no cover - sklearn is a core dep
            return None

        x = StandardScaler().fit_transform(feat.values)
        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=self.random_state)
        labels = gmm.fit_predict(x)

        # Map the cluster with the lower mean return to "risk-off".
        means = [feat["ret"].values[labels == k].mean() for k in range(2)]
        risk_off_cluster = int(np.argmin(means))
        proba = gmm.predict_proba(x[-1:])[0]
        risk_off_prob = float(proba[risk_off_cluster])
        label = "risk-off" if risk_off_prob >= 0.5 else "risk-on"
        return RegimeResult(label, risk_off_prob, index_ret, index_vol, "gmm")

    def _rule(self, feat: pd.DataFrame, index_ret: float, index_vol: float) -> RegimeResult:
        vol_pctile = float((feat["vol"] <= index_vol).mean())
        risk_off = index_ret < 0 and vol_pctile > 0.5
        # Probability proxy blends elevated vol and negative drift.
        prob = float(np.clip(0.5 * vol_pctile + 0.5 * (index_ret < 0), 0.0, 1.0))
        return RegimeResult("risk-off" if risk_off else "risk-on", prob, index_ret, index_vol, "rule")


def detect_regime(md: MarketData, vol_window: int = 20) -> RegimeResult:
    """Convenience wrapper: fit + classify the latest regime."""
    return RegimeDetector(vol_window=vol_window).detect(md)
