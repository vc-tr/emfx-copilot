"""FastAPI service exposing the desk co-pilot.

Run with::

    uvicorn emfx_copilot.api.app:app --reload
    # or: emfx serve

Endpoints cover the desk workflow: market snapshot, signals, risk, regime,
pre-trade analysis, the morning briefing, and a free-form agent ``/ask``.
Requires the ``api`` extra: ``pip install 'emfx-copilot[api]'``.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..agent.briefing import build_briefing
from ..agent.copilot import Copilot
from ..agent.tools import (
    CopilotContext,
    pretrade,
    tool_assess_risk,
    tool_compute_signals,
    tool_detect_regime,
    tool_market_snapshot,
)
from ..config import get_settings


class PretradeRequest(BaseModel):
    ccy: str = Field(..., examples=["BRL"])
    side: str = Field(..., examples=["long_em"], description="long_em (buy EM / sell USD) or short_em")
    notional_usd: float = Field(..., examples=[10_000_000])
    tenor_months: int = Field(3, examples=[3])


class BriefingRequest(BaseModel):
    headlines: list[str] = Field(default_factory=list)


class AskRequest(BaseModel):
    question: str = Field(..., examples=["What's the regime and where should the book lean?"])


def create_app() -> FastAPI:
    settings = get_settings()
    ctx = CopilotContext.default(settings=settings)

    app = FastAPI(
        title="EM FX Desk Co-pilot",
        version="0.1.0",
        description="AI + quant analytics for a Currencies & Emerging Markets trading desk.",
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "llm": ctx.llm.name, "asof": str(ctx.market.spot.index[-1].date())}

    @app.get("/market/snapshot")
    def market_snapshot() -> dict[str, Any]:
        return tool_market_snapshot(ctx)

    @app.get("/signals")
    def signals() -> dict[str, Any]:
        return tool_compute_signals(ctx)

    @app.get("/risk")
    def risk() -> dict[str, Any]:
        return tool_assess_risk(ctx)

    @app.get("/regime")
    def regime() -> dict[str, Any]:
        return tool_detect_regime(ctx)

    @app.post("/pretrade")
    def pretrade_endpoint(req: PretradeRequest) -> dict[str, Any]:
        result = pretrade(ctx, req.ccy, req.side, req.notional_usd, req.tenor_months)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result

    @app.post("/briefing")
    def briefing(req: BriefingRequest) -> dict[str, Any]:
        b = build_briefing(ctx, req.headlines)
        return {"asof": b.asof, "markdown": b.to_markdown(), "regime": b.regime, "risk": b.risk}

    @app.post("/ask")
    def ask(req: AskRequest) -> dict[str, Any]:
        answer = Copilot(ctx).ask(req.question)
        return {"answer": answer.text, "tools_used": answer.tools_used}

    return app


app = create_app()
