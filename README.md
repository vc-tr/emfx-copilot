# emfx-copilot

**An AI co-pilot for a Currencies & Emerging Markets (EM FX) trading desk.**

`emfx-copilot` fuses an LLM/agent market-intelligence layer with a quant
signal/pricing/risk engine — the toolkit a front-office EM FX desk leans on for
pre-trade analysis, market monitoring, risk and exposure, and client-ready
commentary. It runs **fully offline** against a deterministic synthetic market
with a mock LLM (no API key, no data feed), and swaps in **Claude** for the
language layer with one environment variable.

> Built as a focused study project around the workflows of a Currencies &
> Emerging Markets trading desk: NDFs, carry, cross-asset risk regimes, VaR,
> P&L reconciliation, and turning all of it into a timely desk read.

---

## Why this exists

The desk's day is: *monitor market drivers → form a pre-trade view → manage
risk, liquidity and exposure → process trades → keep post-trade P&L accurate.*
`emfx-copilot` implements a working slice of each, and wires an agent on top that
can answer desk questions and write the morning briefing.

| Desk responsibility | Where it lives |
| --- | --- |
| Deliver timely market insights | `agent/briefing.py` — LLM-written morning briefing |
| Monitor market drivers, risk-on/off, cross-asset moves | `regime/` — GMM risk-regime detector; `nlp/sentiment.py` — central-bank tone & risk sentiment |
| Pre-trade analysis | `agent.pretrade` — pricing + signal alignment + marginal VaR + go/no-go |
| Product pricing, models, quant frameworks | `pricing/` — covered-interest forwards & **NDFs**; `signals/` — carry/momentum/value |
| Manage risk, liquidity, exposure | `risk/` — VaR/ES, exposure & limits, regional concentration |
| P&L reconciliation, investigate breaks | `risk/pnl.py` — attribution + break reconciliation |
| Anticipate market movements | `backtest/` — walk-forward strategy backtest |

---

## The stack (and what each piece demonstrates)

- **LLM agent (tool use)** — a manual Claude tool-use loop (`agent/copilot.py`)
  with six grounded tools; adaptive thinking; a deterministic offline mock so
  the agent is testable.
- **NLP** — central-bank hawkish/dovish + risk-on/off scoring, with a small
  **eval harness** (`evals/`) treating the scorer as a testable artifact.
- **Quant** — cross-sectional carry/momentum/value factors, inverse-vol sizing,
  and a no-lookahead **walk-forward backtest**.
- **Pricing & risk** — CIP forwards/NDFs, parametric & historical VaR, expected
  shortfall, exposure limits, P&L attribution and break reconciliation.
- **ML** — a Gaussian-mixture risk-regime classifier (`scikit-learn`) with a
  transparent rule-based fallback.
- **Engineering** — typed `src/` layout, `pydantic-settings` config, **FastAPI**
  service, **Typer + Rich** CLI, `pytest` suite, `ruff` lint, GitHub Actions
  CI, and a Dockerfile.

---

## Quickstart

```bash
# 1. Install (uv recommended; plain pip works too)
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,api]"

# 2. Run the full offline demo (no API key needed)
emfx demo
```

Individual commands:

```bash
emfx regime                       # risk-on / risk-off read
emfx signals                      # factor z-scores + target book
emfx risk                         # VaR / exposure / limit breaches
emfx price BRL --tenor-months 3   # CIP forward / NDF
emfx trade MXN long_em --notional-usd 10000000   # pre-trade check
emfx briefing                     # morning desk briefing
emfx ask "What's the regime and where should the book lean?"
emfx backtest                     # walk-forward strategy backtest
emfx evals                        # sentiment-scorer eval set
emfx serve                        # FastAPI on :8000 (needs the api extra)
```

## Use it as a library

```python
from emfx_copilot.agent.tools import CopilotContext
from emfx_copilot.agent.copilot import Copilot
from emfx_copilot.agent.briefing import build_briefing

ctx = CopilotContext.default()                     # synthetic market + mock LLM
print(Copilot(ctx).ask("Price a 3m KRW NDF and tell me the regime.").text)
print(build_briefing(ctx, headlines=["EM Asia FX rallies on risk-on optimism"]).to_markdown())
```

## HTTP API

```bash
emfx serve   # or: uvicorn emfx_copilot.api.app:app --reload
```

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | GET | liveness + LLM backend |
| `/market/snapshot` | GET | spot, rates, carry, vol, NDF flags |
| `/signals` | GET | factor scores + target book |
| `/risk` | GET | VaR/ES, exposure, breaches |
| `/regime` | GET | risk-on/off regime |
| `/pretrade` | POST | single-ticket pre-trade check |
| `/briefing` | POST | morning briefing (markdown) |
| `/ask` | POST | agentic desk Q&A |

---

## Enabling Claude (real LLM)

The language layer is provider-agnostic and defaults to a deterministic mock.
To use Claude for sentiment, briefings, and the agent:

```bash
pip install -e ".[llm]"
export EMFX_LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
# optional: export EMFX_LLM_MODEL=claude-opus-4-8
emfx briefing
```

Calls go through the official `anthropic` SDK (`nlp/llm.py`) using the Messages
API with adaptive thinking and a manual tool-use loop. See `.env.example`.

---

## Data & disclaimer

The bundled market is **synthetic and reproducible** (seeded) — spot paths driven
by a risk factor with stochastic vol, per-currency betas, carry drift, and
idiosyncratic noise. Policy rates and reference spots are *indicative*. There is
a documented `MarketData.from_provider` hook where a live feed would plug in.
Nothing here is investment advice or a live trading system; it is a portfolio /
study project.

See [`docs/architecture.md`](docs/architecture.md) for the design and data flow.

## Development

```bash
uv pip install -e ".[dev,api]"
pytest            # test suite
ruff check .      # lint
```

## License

MIT — see [`LICENSE`](LICENSE).
