# Architecture

`emfx-copilot` is a desk-shaped analytics stack for a **Currencies & Emerging
Markets (EM FX)** trading desk: a deterministic quant core (pricing, signals,
risk, regime) with an **LLM/agent layer** on top that turns the numbers into a
desk read. Everything runs **offline** against a seeded synthetic market, and
the language layer defaults to a deterministic mock so the whole thing is
reproducible and testable with no API key and no data feed.

## Layered view

```
        ┌──────────────────────────────────────────────────────────────┐
 access │   CLI (Typer + Rich)       FastAPI service        Python lib   │
        │   emfx regime/signals/…    /signals /risk /…       import emfx │
        └───────────────┬────────────────┬───────────────────┬──────────┘
                        │                │                   │
        ┌───────────────▼────────────────▼───────────────────▼──────────┐
  agent │   Copilot (manual tool-use loop)  ·  Briefing  ·  Pre-trade    │
        │   tools: detect_regime · compute_signals · assess_risk ·       │
        │          price_forward · score_news                            │
        └──────┬──────────┬──────────┬──────────┬──────────┬─────────────┘
               │          │          │          │          │
        ┌──────▼───┐ ┌────▼─────┐ ┌──▼──────┐ ┌─▼───────┐ ┌▼─────────┐
  core  │ pricing  │ │ signals  │ │  risk   │ │ regime  │ │   nlp    │
        │ CIP fwd, │ │ carry,   │ │ VaR/ES, │ │ GMM     │ │ LLM +    │
        │ NDF,     │ │ momentum,│ │ exposure│ │ risk-on │ │ hawkish/ │
        │ carry    │ │ value,   │ │ /limits,│ │ /off    │ │ dovish + │
        │          │ │ blend    │ │ P&L attr│ │ detector│ │ risk tone│
        └──────┬───┘ └────┬─────┘ └──┬──────┘ └─┬───────┘ └┬─────────┘
               └──────────┴────┬─────┴──────────┴──────────┘
                               │
                        ┌──────▼────────┐        ┌───────────────┐
                  data  │  MarketData   │◀───────│   universe    │
                        │  (synthetic,  │        │ EM ccys, NDF  │
                        │  seeded)      │        │ flags, rates  │
                        └───────────────┘        └───────────────┘
                          MarketData.from_provider() = live-feed hook

        backtest/  — walk-forward strategy backtest over the core signals
        evals/     — eval harness treating the sentiment scorer as an artifact
```

## Data flow — one "desk read"

1. **Market.** `data/` builds a `MarketData` snapshot: spot histories, policy
   rates, realized vol, carry and NDF flags for the EM universe (`data/universe.py`).
   The bundled market is synthetic and seeded (`data/synthetic.py`) so results
   are reproducible.
2. **Quant core.** From that snapshot:
   - `pricing/` prices outright forwards and **NDFs** via covered interest parity
     and computes carry.
   - `signals/` builds cross-sectional carry / momentum / value factors, z-scores
     them, and blends to inverse-vol target weights.
   - `risk/` computes parametric & historical VaR / expected shortfall, aggregates
     exposure by currency and region against limits, and attributes P&L.
   - `regime/` fits a Gaussian-mixture detector to classify the risk-on / risk-off
     state (with a transparent rule-based fallback).
3. **Language layer.** `nlp/` scores central-bank statements and news for
   hawkish/dovish tilt and risk sentiment. `nlp/llm.py` is provider-agnostic: a
   deterministic `MockLLM` (default, offline) or `AnthropicLLM` (real Claude).
4. **Agent.** `agent/` exposes the core as a small set of tools and runs a manual
   tool-use loop (`agent/copilot.py`) that plans, calls tools, and composes an
   answer. `agent/briefing.py` assembles the morning briefing; `agent.pretrade`
   runs a single-ticket go/no-go check (pricing + signal alignment + marginal VaR).
5. **Surfaces.** The same analytics are reachable three ways: the **CLI**
   (`cli.py`), the **FastAPI** service (`api/app.py`), and as a **library**.

## The agent loop

We run the tool-use loop ourselves rather than delegating to the SDK's runner so
that every tool call is recorded into a trace (for transparency/audit) and the
loop behaves identically against the real Claude backend and the offline mock.
Each iteration:

1. Call the model with the desk system prompt, the running message list, the tool
   specs, and adaptive thinking enabled.
2. If the model returns tool calls, execute each grounded tool against the live
   `CopilotContext`, append the results, and loop.
3. If the model returns text with no tool calls (or we hit the iteration cap),
   return the answer plus the tool trace.

The tool specs follow the Anthropic Messages API shape, so the `MockLLM` can plan
a sensible tool sequence and compose a desk-style answer from the results — which
is what makes the agent end-to-end testable offline.

## The synthetic market

Spot paths are generated from a shared risk factor with stochastic volatility,
per-currency betas to that factor, a carry-driven drift, and idiosyncratic noise;
policy rates and reference spots are indicative. This is deliberately *not* a
data vendor — it is a seeded, reproducible stand-in so the analytics, demo, and
tests are deterministic. The seam for real data is `MarketData.from_provider()`:
implement that hook against a live feed and the entire stack above is unchanged.

## Extending it

- **Real LLM** — set `EMFX_LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY`; the
  sentiment scorer, briefing prose, and agent all route through Claude via the
  official SDK (`nlp/llm.py`). Defaults to `claude-opus-4-8`.
- **Live market data** — implement `MarketData.from_provider()`; nothing
  downstream changes.
- **New signals / tools** — add a factor in `signals/factors.py` or a tool in
  `agent/tools.py`; both are picked up by the blend / agent automatically.

## Mapping to the desk's responsibilities

| Desk responsibility | Where it lives |
| --- | --- |
| Deliver timely market insights | `agent/briefing.py` — morning briefing |
| Monitor drivers, risk-on/off, cross-asset moves | `regime/` + `nlp/sentiment.py` |
| Pre-trade analysis | `agent.pretrade` — pricing + signal + marginal VaR + go/no-go |
| Product pricing, models, quant frameworks | `pricing/` (CIP forwards & NDFs), `signals/` |
| Manage risk, liquidity, exposure | `risk/` — VaR/ES, exposure & limits, concentration |
| P&L reconciliation, investigate breaks | `risk/pnl.py` — attribution + break reconciliation |
| Anticipate market movements | `backtest/` — walk-forward strategy backtest |

## Disclaimer

Synthetic data; indicative rates. Not investment advice and not a live trading
system — a portfolio / study project built around EM FX desk workflows.
