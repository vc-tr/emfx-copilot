"""The co-pilot's tool surface.

Each tool is a small, typed capability the agent can call to ground its answers
in live desk data: market snapshot, forward/NDF pricing, factor signals, the
risk book, the regime read, and news sentiment. ``TOOLS`` holds the Anthropic
tool specs; ``run_tool`` dispatches a call to the right Python function bound to
a ``CopilotContext``. All tools return JSON-serialisable dicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import pandas as pd

from ..config import Settings, get_settings
from ..data.market_data import MarketData
from ..data.universe import CODES, get_currency, is_ndf
from ..nlp.llm import BaseLLM
from ..nlp.sentiment import score_text
from ..pricing.forwards import quote_forward
from ..regime.detector import detect_regime
from ..risk.exposure import PositionBook, RiskLimits, check_limits
from ..risk.metrics import expected_shortfall, parametric_var
from ..signals.blend import blend_signals, target_weights, to_notional, top_signals
from ..signals.factors import all_signals


@dataclass
class CopilotContext:
    """Everything the tools need: the market, the LLM, settings, and a book."""

    market: MarketData
    llm: BaseLLM
    settings: Settings
    book: PositionBook | None = None

    @classmethod
    def default(cls, llm: BaseLLM | None = None, settings: Settings | None = None) -> CopilotContext:
        from ..nlp.llm import get_llm

        settings = settings or get_settings()
        market = MarketData.synthetic(seed=settings.data_seed, n_days=settings.history_days)
        return cls(market=market, llm=llm or get_llm(settings), settings=settings)


# --- individual tools -------------------------------------------------------
def tool_market_snapshot(ctx: CopilotContext) -> dict[str, Any]:
    snap = ctx.market.latest_snapshot()
    rows = {
        c: {
            "spot": round(snap.spot[c], 4),
            "rate": round(snap.local_rates[c], 4),
            "carry": round(snap.carry(c), 4),
            "vol": round(snap.realized_vol.get(c, 0.0), 4),
            "ndf": is_ndf(c),
        }
        for c in snap.spot
    }
    return {"asof": str(snap.asof.date()), "usd_rate": snap.usd_rate, "currencies": rows}


def tool_price_forward(ctx: CopilotContext, ccy: str, tenor_months: int = 3) -> dict[str, Any]:
    cur = get_currency(ccy)
    snap = ctx.market.latest_snapshot()
    quote = quote_forward(
        code=cur.code,
        spot=snap.spot[cur.code],
        r_local=snap.local_rates[cur.code],
        r_usd=snap.usd_rate,
        tenor_months=int(tenor_months),
        is_ndf=not cur.deliverable,
    )
    return quote.as_dict()


def _combined_and_weights(ctx: CopilotContext) -> tuple[pd.Series, pd.Series, pd.Series]:
    sigs = all_signals(ctx.market)
    combined = blend_signals(sigs, ctx.settings.signal_weights)
    vol = ctx.market.realized_vol()
    weights = target_weights(combined, vol)
    return combined, weights, vol


def tool_compute_signals(ctx: CopilotContext) -> dict[str, Any]:
    sigs = all_signals(ctx.market)
    combined = blend_signals(sigs, ctx.settings.signal_weights)
    weights = target_weights(combined, ctx.market.realized_vol())
    longs, shorts = top_signals(combined, n=3)
    table = {
        c: {
            "carry": round(float(sigs["carry"].get(c, 0.0)), 2),
            "momentum": round(float(sigs["momentum"].get(c, 0.0)), 2),
            "value": round(float(sigs["value"].get(c, 0.0)), 2),
            "combined": round(float(combined.get(c, 0.0)), 3),
            "weight": round(float(weights.get(c, 0.0)), 3),
        }
        for c in combined.index
    }
    return {
        "weights_basis": "unit gross (sum |w| = 1)",
        "top_longs": longs,
        "top_shorts": shorts,
        "factors": table,
    }


def tool_assess_risk(ctx: CopilotContext) -> dict[str, Any]:
    combined, weights, _ = _combined_and_weights(ctx)
    notional = to_notional(weights, ctx.settings.gross_limit_usd, ctx.settings.per_ccy_limit_usd)
    book = ctx.book or PositionBook(positions={c: float(notional.get(c, 0.0)) for c in combined.index})

    gross = book.gross_usd()
    # Portfolio historical returns from current weights (no lookahead concern for risk).
    w = pd.Series(book.positions, dtype=float)
    w_unit = w / gross if gross > 0 else w
    port_ret = (ctx.market.emlong_returns() * w_unit).sum(axis=1)

    conf = ctx.settings.var_confidence
    var_pct = parametric_var(port_ret, conf)
    es_pct = expected_shortfall(port_ret, conf)

    limits = RiskLimits(
        gross_limit_usd=ctx.settings.gross_limit_usd,
        per_ccy_limit_usd=ctx.settings.per_ccy_limit_usd,
        max_concentration=ctx.settings.max_concentration,
    )
    breaches = [b.as_dict() for b in check_limits(book, limits)]

    return {
        "confidence": conf,
        "gross_usd": round(gross, 2),
        "net_usd": round(book.net_usd(), 2),
        "concentration": round(book.concentration(), 3),
        "var_pct_1d": round(var_pct, 5),
        "es_pct_1d": round(es_pct, 5),
        "var_usd_1d_99": round(var_pct * gross, 2),
        "es_usd_1d_99": round(es_pct * gross, 2),
        "by_region": {k: round(v, 2) for k, v in book.by_region().items()},
        "breaches": breaches,
    }


def tool_detect_regime(ctx: CopilotContext) -> dict[str, Any]:
    return detect_regime(ctx.market).as_dict()


def tool_score_news(ctx: CopilotContext, text: str) -> dict[str, Any]:
    return score_text(text, ctx.llm).as_dict()


# --- registry ---------------------------------------------------------------
_DISPATCH = {
    "get_market_snapshot": lambda ctx, **kw: tool_market_snapshot(ctx),
    "price_forward": lambda ctx, **kw: tool_price_forward(ctx, **kw),
    "compute_signals": lambda ctx, **kw: tool_compute_signals(ctx),
    "assess_risk": lambda ctx, **kw: tool_assess_risk(ctx),
    "detect_regime": lambda ctx, **kw: tool_detect_regime(ctx),
    "score_news": lambda ctx, **kw: tool_score_news(ctx, **kw),
}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_market_snapshot",
        "description": "Current spot, local rate, long-EM carry, realised vol, and NDF flag for every currency on the desk.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "price_forward",
        "description": "Price an outright forward / NDF for one currency via covered interest parity. Returns the outright, forward points, and annualised premium.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ccy": {"type": "string", "description": f"Currency code, one of: {', '.join(CODES)}"},
                "tenor_months": {"type": "integer", "description": "Tenor in months (e.g. 1, 3, 6, 12)"},
            },
            "required": ["ccy"],
        },
    },
    {
        "name": "compute_signals",
        "description": "Cross-sectional carry/momentum/value factor scores, the blended target book (unit gross), and the top longs/shorts.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "assess_risk",
        "description": "Risk of the model target book (or the desk book if set): gross/net, concentration, 1-day VaR & expected shortfall (% and USD), regional exposure, and any limit breaches.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "detect_regime",
        "description": "Current EM risk-on / risk-off regime from the EM-index return, volatility, and cross-sectional dispersion.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "score_news",
        "description": "Score market text for central-bank tone (hawkish/dovish) and risk sentiment (risk-on/off).",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Headline or statement to score"}},
            "required": ["text"],
        },
    },
]


def run_tool(name: str, tool_input: dict[str, Any] | None, ctx: CopilotContext) -> dict[str, Any]:
    if name not in _DISPATCH:
        return {"error": f"unknown tool {name!r}"}
    try:
        return _DISPATCH[name](ctx, **(tool_input or {}))
    except Exception as exc:  # surface tool errors to the agent rather than crashing the loop
        return {"error": f"{type(exc).__name__}: {exc}"}


# --- pre-trade analysis (used by the API/CLI; not exposed as an agent tool) --
def pretrade(
    ctx: CopilotContext,
    ccy: str,
    side: str,
    notional_usd: float,
    tenor_months: int = 3,
) -> dict[str, Any]:
    """Pre-trade check for a single ticket: pricing, signal alignment, expected
    carry over the tenor, a marginal 1-day VaR, and a go / caution verdict."""
    side = side.lower()
    if side not in ("long_em", "short_em"):
        return {"error": "side must be 'long_em' (buy EM / sell USD) or 'short_em'"}

    cur = get_currency(ccy)
    snap = ctx.market.latest_snapshot()
    want_long = side == "long_em"

    quote = quote_forward(
        code=cur.code,
        spot=snap.spot[cur.code],
        r_local=snap.local_rates[cur.code],
        r_usd=snap.usd_rate,
        tenor_months=int(tenor_months),
        is_ndf=not cur.deliverable,
    )

    combined = blend_signals(all_signals(ctx.market), ctx.settings.signal_weights)
    score = float(combined.get(cur.code, 0.0))
    aligned = ((score > 0) == want_long) and abs(score) > 0.05

    carry = snap.carry(cur.code)
    expected_carry_usd = (carry if want_long else -carry) * (tenor_months / 12.0) * notional_usd

    vol_ann = snap.realized_vol.get(cur.code, 0.0)
    daily_vol = vol_ann / (252 ** 0.5)
    z = NormalDist().inv_cdf(ctx.settings.var_confidence)
    var_usd = notional_usd * daily_vol * z

    regime = detect_regime(ctx.market)
    notes: list[str] = ["signal-aligned" if aligned else "against the model signal"]
    risk_off_long = regime.label == "risk-off" and want_long
    if risk_off_long:
        notes.append("risk-off regime — size down high-beta longs and tighten stops")
    if not cur.deliverable:
        notes.append("NDF: settles in USD vs fixing; no physical delivery")

    go = aligned and not (risk_off_long and vol_ann > 0.12)

    return {
        "code": cur.code,
        "side": side,
        "notional_usd": round(notional_usd, 2),
        "instrument": "NDF" if not cur.deliverable else "deliverable forward",
        "forward": quote.forward,
        "forward_points": round(quote.points, 6),
        "tenor_months": int(tenor_months),
        "signal_score": round(score, 3),
        "expected_carry_usd": round(expected_carry_usd, 2),
        "marginal_var_usd_1d": round(var_usd, 2),
        "regime": regime.label,
        "verdict": "GO" if go else "CAUTION",
        "notes": notes,
    }
