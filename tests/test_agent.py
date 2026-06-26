from __future__ import annotations

import pytest

from emfx_copilot.agent.briefing import build_briefing
from emfx_copilot.agent.copilot import Copilot
from emfx_copilot.agent.tools import TOOLS, pretrade, run_tool


@pytest.mark.parametrize("spec", TOOLS)
def test_each_tool_runs_without_error(ctx, spec):
    name = spec["name"]
    payload = {}
    if name == "price_forward":
        payload = {"ccy": "BRL", "tenor_months": 3}
    elif name == "score_news":
        payload = {"text": "hawkish hike inflation"}
    out = run_tool(name, payload, ctx)
    assert isinstance(out, dict)
    assert "error" not in out


def test_copilot_answers_and_uses_tools(ctx):
    answer = Copilot(ctx).ask("What's the regime and how should we price a 3m BRL NDF?")
    assert answer.text.strip()
    # The mock plans the core desk tools; BRL mention pulls in price_forward.
    assert "detect_regime" in answer.tools_used
    assert "compute_signals" in answer.tools_used
    assert "assess_risk" in answer.tools_used
    assert "price_forward" in answer.tools_used


def test_pretrade_verdict(ctx):
    r = pretrade(ctx, ccy="MXN", side="long_em", notional_usd=10e6, tenor_months=3)
    assert r["verdict"] in {"GO", "CAUTION"}
    assert r["code"] == "MXN"
    assert "marginal_var_usd_1d" in r


def test_pretrade_rejects_bad_side(ctx):
    r = pretrade(ctx, ccy="MXN", side="buy", notional_usd=10e6)
    assert "error" in r


def test_build_briefing(ctx):
    b = build_briefing(ctx, headlines=["EM FX rallies on risk-on optimism and inflows"])
    md = b.to_markdown()
    assert "EM FX Desk Briefing" in md
    assert b.regime["label"] in {"risk-on", "risk-off"}
    assert b.sentiment["n"] == 1
