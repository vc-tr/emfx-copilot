"""End-to-end offline demo of the EM FX desk co-pilot.

Runs the full pipeline against the deterministic synthetic market with the mock
LLM — no API key, no network — and prints a desk's-eye view: regime, factor
signals, a pre-trade ticket, the risk book, a backtest, and the morning briefing
plus an agentic Q&A.
"""

from __future__ import annotations

from rich.console import Console
from rich.rule import Rule

from .agent.briefing import build_briefing
from .agent.copilot import Copilot
from .agent.tools import (
    CopilotContext,
    pretrade,
    tool_assess_risk,
    tool_compute_signals,
    tool_detect_regime,
)
from .backtest.engine import run_backtest
from .config import get_settings

console = Console()


def run_demo() -> None:
    settings = get_settings()
    ctx = CopilotContext.default(settings=settings)
    console.print(Rule("EM FX Desk Co-pilot — offline demo"))
    console.print(f"Market as of [bold]{ctx.market.spot.index[-1].date()}[/], LLM backend: [bold]{ctx.llm.name}[/]\n")

    regime = tool_detect_regime(ctx)
    console.print(f"[bold]Regime:[/] {regime['label']} (risk-off prob {regime['risk_off_prob']}, "
                  f"method {regime['method']})")

    sig = tool_compute_signals(ctx)
    console.print(f"[bold]Signals:[/] long {', '.join(sig['top_longs'])} / short {', '.join(sig['top_shorts'])}")

    pt = pretrade(ctx, ccy="BRL", side="long_em", notional_usd=15_000_000, tenor_months=3)
    console.print(f"[bold]Pre-trade[/] (BRL long_em $15mm 3m): {pt['verdict']} — "
                  f"fwd {pt['forward']}, carry ${pt['expected_carry_usd']:,.0f}, "
                  f"VaR ${pt['marginal_var_usd_1d']:,.0f}")

    risk = tool_assess_risk(ctx)
    console.print(f"[bold]Risk:[/] gross ${risk['gross_usd']:,.0f}, "
                  f"1d 99% VaR ${risk['var_usd_1d_99']:,.0f}, breaches {risk['breaches'] or 'none'}")

    bt = run_backtest(ctx.market, settings).summary()
    console.print(f"[bold]Backtest:[/] Sharpe {bt['sharpe']}, ann.return {bt['ann_return']}, "
                  f"maxDD {bt['max_drawdown']}, turnover {bt['avg_turnover']}")

    console.print(Rule("Morning briefing"))
    headlines = [
        "Bank of Korea holds but signals a hawkish tilt as inflation expectations firm",
        "EM Asia FX rallies on broad risk-on optimism and equity inflows",
        "Lira slumps anew as intervention fails to stem outflows; risk-off grips the region",
    ]
    console.print(build_briefing(ctx, headlines).to_markdown())

    console.print(Rule("Co-pilot Q&A"))
    answer = Copilot(ctx).ask("What's the regime, where should the book lean, and what's our risk?")
    console.print(answer.text)
    console.print(f"[dim]tools used: {', '.join(answer.tools_used)}[/]")


if __name__ == "__main__":  # pragma: no cover
    run_demo()
