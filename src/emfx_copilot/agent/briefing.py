"""Pre-trade / morning desk briefing.

Assembles the desk state (regime, factor book, risk, news sentiment) into a
structured object and a markdown narrative. With a real LLM the narrative is
written by Claude from the structured context; offline it uses a deterministic
template so the briefing always renders.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..nlp.prompts import BRIEFING_SYSTEM, briefing_user
from ..nlp.sentiment import aggregate, score_text
from .tools import (
    CopilotContext,
    tool_assess_risk,
    tool_compute_signals,
    tool_detect_regime,
)


@dataclass
class Briefing:
    asof: str
    regime: dict[str, Any]
    signals: dict[str, Any]
    risk: dict[str, Any]
    sentiment: dict[str, Any]
    narrative: str
    headlines: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        longs = ", ".join(self.signals.get("top_longs", []))
        shorts = ", ".join(self.signals.get("top_shorts", []))
        var = self.risk.get("var_usd_1d_99", 0.0)
        gross = self.risk.get("gross_usd", 0.0)
        breaches = self.risk.get("breaches", [])
        lines = [
            f"# EM FX Desk Briefing — {self.asof}",
            "",
            f"**Regime:** {self.regime.get('label')} "
            f"(risk-off prob {self.regime.get('risk_off_prob')}, "
            f"method: {self.regime.get('method')})",
            "",
            f"**Book lean:** long {longs or 'n/a'} / short {shorts or 'n/a'}",
            "",
            f"**Risk:** 1-day {int(self.risk.get('confidence', 0.99) * 100)}% VaR "
            f"${var:,.0f} on ${gross:,.0f} gross; "
            f"concentration {self.risk.get('concentration')}",
        ]
        if breaches:
            lines.append("")
            lines.append(f"**⚠ Limit breaches:** {len(breaches)} — {breaches}")
        lines += [
            "",
            f"**Sentiment:** {self.sentiment.get('summary', 'n/a')}",
            "",
            "## Narrative",
            self.narrative,
        ]
        return "\n".join(lines)


def build_briefing(ctx: CopilotContext, headlines: list[str] | None = None) -> Briefing:
    headlines = headlines or []
    regime = tool_detect_regime(ctx)
    signals = tool_compute_signals(ctx)
    risk = tool_assess_risk(ctx)
    sentiment = aggregate([score_text(h, ctx.llm) for h in headlines])
    asof = ctx.market.latest_snapshot().asof.date().isoformat()

    context = {"asof": asof, "regime": regime, "signals": signals, "risk": risk, "sentiment": sentiment}
    narrative = _narrative(ctx, context)

    return Briefing(
        asof=asof,
        regime=regime,
        signals=signals,
        risk=risk,
        sentiment=sentiment,
        narrative=narrative,
        headlines=headlines,
    )


def _narrative(ctx: CopilotContext, context: dict[str, Any]) -> str:
    if ctx.llm.name != "mock":
        try:
            resp = ctx.llm.complete(
                system=BRIEFING_SYSTEM,
                messages=[{"role": "user", "content": briefing_user(json.dumps(context, default=str))}],
                max_tokens=ctx.settings.llm_max_tokens,
            )
            if resp.text.strip():
                return resp.text.strip()
        except Exception:
            pass  # fall through to the template
    return _template_narrative(context)


def _template_narrative(context: dict[str, Any]) -> str:
    regime = context["regime"]
    signals = context["signals"]
    risk = context["risk"]
    sentiment = context["sentiment"]
    longs = ", ".join(signals.get("top_longs", [])) or "n/a"
    shorts = ", ".join(signals.get("top_shorts", [])) or "n/a"
    risk_word = "defensive" if regime.get("label") == "risk-off" else "constructive"
    return (
        f"The desk opens {risk_word}: the regime model reads {regime.get('label')} "
        f"(risk-off probability {regime.get('risk_off_prob')}). The blended carry/momentum/value "
        f"book leans long {longs} and short {shorts}. Risk sits at a 1-day "
        f"{int(risk.get('confidence', 0.99) * 100)}% VaR of ${risk.get('var_usd_1d_99', 0):,.0f} "
        f"on ${risk.get('gross_usd', 0):,.0f} gross, with single-name concentration at "
        f"{risk.get('concentration')}. News backdrop: {sentiment.get('summary', 'no headlines')}. "
        f"{'Trim high-beta longs and keep powder dry.' if regime.get('label') == 'risk-off' else 'Carry remains the core engine; add on dips with stops.'}"
    )
