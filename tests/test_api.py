from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from emfx_copilot.api.app import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_signals_and_risk_and_regime(client):
    for path in ("/signals", "/risk", "/regime", "/market/snapshot"):
        r = client.get(path)
        assert r.status_code == 200
        assert isinstance(r.json(), dict)


def test_pretrade_endpoint(client):
    r = client.post(
        "/pretrade",
        json={"ccy": "BRL", "side": "long_em", "notional_usd": 10_000_000, "tenor_months": 3},
    )
    assert r.status_code == 200
    assert r.json()["verdict"] in {"GO", "CAUTION"}


def test_pretrade_bad_side_is_400(client):
    r = client.post("/pretrade", json={"ccy": "BRL", "side": "buy", "notional_usd": 1_000_000})
    assert r.status_code == 400


def test_ask_endpoint(client):
    r = client.post("/ask", json={"question": "What's the regime?"})
    assert r.status_code == 200
    assert r.json()["answer"].strip()
