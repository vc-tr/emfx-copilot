"""``emfx`` command-line interface.

A desk-facing CLI over the same analytics the API serves: regime, signals,
risk, forward pricing, pre-trade checks, the morning briefing, the agent, and
the backtest. Runs offline with the mock LLM by default.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .agent.briefing import build_briefing
from .agent.copilot import Copilot
from .agent.tools import (
    CopilotContext,
    pretrade,
    tool_assess_risk,
    tool_compute_signals,
    tool_detect_regime,
    tool_price_forward,
)
from .backtest.engine import run_backtest
from .config import get_settings
from .evals.sentiment_evals import run_sentiment_eval

app = typer.Typer(add_completion=False, help="EM FX desk co-pilot — AI + quant analytics.")
console = Console()


def _ctx() -> CopilotContext:
    return CopilotContext.default()


@app.command()
def regime() -> None:
    """Show the current risk-on / risk-off regime."""
    r = tool_detect_regime(_ctx())
    console.print(Panel.fit(
        f"[bold]{r['label'].upper()}[/]  (risk-off prob {r['risk_off_prob']}, method {r['method']})\n"
        f"20d EM index return {r['index_return_20d']} | index vol {r['index_vol']}",
        title="Regime",
    ))


@app.command()
def signals() -> None:
    """Show factor scores and the blended target book."""
    s = tool_compute_signals(_ctx())
    table = Table(title="Factor signals (z-scores) & target weights")
    for col in ("ccy", "carry", "momentum", "value", "combined", "weight"):
        table.add_column(col, justify="right" if col != "ccy" else "left")
    for ccy, row in sorted(s["factors"].items(), key=lambda kv: kv[1]["combined"], reverse=True):
        table.add_row(ccy, str(row["carry"]), str(row["momentum"]), str(row["value"]),
                      str(row["combined"]), str(row["weight"]))
    console.print(table)
    console.print(f"[green]Top longs:[/] {', '.join(s['top_longs'])}   "
                  f"[red]Top shorts:[/] {', '.join(s['top_shorts'])}")


@app.command()
def risk() -> None:
    """Show risk of the model target book."""
    r = tool_assess_risk(_ctx())
    console.print(Panel.fit(
        f"Gross ${r['gross_usd']:,.0f} | Net ${r['net_usd']:,.0f} | "
        f"Concentration {r['concentration']}\n"
        f"1d {int(r['confidence']*100)}% VaR ${r['var_usd_1d_99']:,.0f} | "
        f"ES ${r['es_usd_1d_99']:,.0f}\n"
        f"By region: {r['by_region']}\n"
        f"Breaches: {r['breaches'] or 'none'}",
        title="Risk",
    ))


@app.command()
def price(
    ccy: str = typer.Argument(..., help="Currency code, e.g. BRL"),
    tenor_months: int = typer.Option(3, help="Tenor in months"),
) -> None:
    """Price a forward / NDF via covered interest parity."""
    q = tool_price_forward(_ctx(), ccy=ccy, tenor_months=tenor_months)
    console.print(Panel.fit(
        f"{q['code']} {q['tenor_months']}m {q['instrument']}\n"
        f"spot {q['spot']} → forward {q['forward']} (points {q['points']})\n"
        f"annualised premium {q['annualized_premium']}",
        title="Forward / NDF",
    ))


@app.command()
def trade(
    ccy: str = typer.Argument(..., help="Currency code, e.g. MXN"),
    side: str = typer.Argument(..., help="long_em (buy EM/sell USD) or short_em"),
    notional_usd: float = typer.Option(10_000_000, help="USD notional"),
    tenor_months: int = typer.Option(3, help="Tenor in months"),
) -> None:
    """Run a pre-trade check on a single ticket."""
    r = pretrade(_ctx(), ccy=ccy, side=side, notional_usd=notional_usd, tenor_months=tenor_months)
    if "error" in r:
        console.print(f"[red]{r['error']}[/]")
        raise typer.Exit(1)
    color = "green" if r["verdict"] == "GO" else "yellow"
    console.print(Panel.fit(
        f"[{color}][bold]{r['verdict']}[/][/]  {r['code']} {r['side']} ${r['notional_usd']:,.0f}\n"
        f"{r['instrument']} fwd {r['forward']} | signal {r['signal_score']} | regime {r['regime']}\n"
        f"expected carry ${r['expected_carry_usd']:,.0f} over {r['tenor_months']}m | "
        f"marginal 1d VaR ${r['marginal_var_usd_1d']:,.0f}\n"
        f"notes: {'; '.join(r['notes'])}",
        title="Pre-trade",
    ))


@app.command()
def briefing() -> None:
    """Generate the morning desk briefing (uses bundled sample headlines)."""
    from importlib.resources import files  # noqa: F401  (kept for future packaged data)

    headlines = [
        "Bank of Korea holds but signals a hawkish tilt as inflation expectations firm",
        "EM Asia FX rallies on broad risk-on optimism and equity inflows",
        "Lira slumps anew as intervention fails to stem outflows; risk-off grips the region",
    ]
    b = build_briefing(_ctx(), headlines)
    console.print(b.to_markdown())


@app.command()
def ask(question: str = typer.Argument(..., help="Free-form desk question")) -> None:
    """Ask the agentic co-pilot a desk question."""
    answer = Copilot(_ctx()).ask(question)
    console.print(Panel(answer.text, title="Co-pilot"))
    console.print(f"[dim]tools used: {', '.join(answer.tools_used) or 'none'}[/]")


@app.command()
def backtest(
    seed: int = typer.Option(7, help="Synthetic market seed"),
    days: int = typer.Option(520, help="History length in trading days"),
) -> None:
    """Backtest the blended strategy on the synthetic market."""
    from .data.market_data import MarketData

    md = MarketData.synthetic(seed=seed, n_days=days)
    result = run_backtest(md, get_settings())
    table = Table(title="Backtest summary")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for k, v in result.summary().items():
        table.add_row(k, str(v))
    console.print(table)


@app.command()
def evals() -> None:
    """Run the sentiment-scorer eval set."""
    from .nlp.llm import get_llm

    report = run_sentiment_eval(get_llm())
    console.print(Panel.fit(
        f"n={report.n}\nhawkish/dovish accuracy {report.hawkish_accuracy}\n"
        f"risk-sentiment accuracy {report.risk_accuracy}",
        title="Sentiment eval",
    ))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Serve the HTTP API (requires the 'api' extra)."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]Install the API extra:[/] pip install 'emfx-copilot[api]'")
        raise typer.Exit(1) from None
    uvicorn.run("emfx_copilot.api.app:app", host=host, port=port, reload=False)


@app.command()
def demo() -> None:
    """Run the full offline demo (regime → signals → pre-trade → risk → briefing)."""
    from .demo import run_demo

    run_demo()


if __name__ == "__main__":  # pragma: no cover
    app()
