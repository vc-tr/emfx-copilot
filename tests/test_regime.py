from __future__ import annotations

from emfx_copilot.regime.detector import RegimeDetector, detect_regime


def test_detect_regime_returns_valid_result(md):
    r = detect_regime(md)
    assert r.label in {"risk-on", "risk-off"}
    assert 0.0 <= r.risk_off_prob <= 1.0
    assert r.method in {"gmm", "rule"}


def test_regime_as_dict(md):
    d = RegimeDetector().detect(md).as_dict()
    assert set(d) == {"label", "risk_off_prob", "index_return_20d", "index_vol", "method"}


def test_regime_is_deterministic(md):
    a = detect_regime(md)
    b = detect_regime(md)
    assert a.label == b.label
    assert a.risk_off_prob == b.risk_off_prob
