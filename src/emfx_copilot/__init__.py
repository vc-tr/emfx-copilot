"""emfx-copilot — an AI co-pilot for a Currencies & Emerging Markets (EM FX) trading desk.

The package fuses three layers that mirror a front-office EM FX desk:

* ``nlp`` / ``agent``  — LLM-driven market intelligence: central-bank tone and
  risk-on/off sentiment scoring, plus a tool-using agent that answers desk
  questions and writes pre-trade briefings.
* ``signals``          — quant cross-sectional factors (carry, momentum, value)
  blended into target positions.
* ``pricing`` / ``risk`` / ``regime`` — NDF & forward pricing (covered interest
  parity), VaR / exposure / P&L reconciliation, and risk-regime detection.

Everything runs offline against a deterministic synthetic market with a mock
LLM, so the repo is reproducible without any API key or market-data feed.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
