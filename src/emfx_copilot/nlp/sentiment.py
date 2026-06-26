"""Central-bank tone & risk-sentiment scoring.

``score_text`` uses the configured LLM when a real provider is available and
falls back to a transparent lexicon model otherwise — so sentiment scoring (and
the briefing that consumes it) works fully offline and is deterministic in
tests. The lexicon is intentionally small and auditable; the LLM path is what
you'd lean on in production for nuance.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .llm import BaseLLM
from .prompts import SENTIMENT_SYSTEM, sentiment_user

HAWKISH = {
    "hike", "hikes", "tighten", "tightening", "restrictive", "inflation", "overheating",
    "raise", "higher", "hawkish", "upside", "tighter", "elevated",
}
DOVISH = {
    "cut", "cuts", "ease", "easing", "accommodative", "dovish", "slowdown", "lower",
    "stimulus", "soften", "softening", "downside", "patient", "pause",
}
RISK_ON = {
    "rally", "rallied", "optimism", "optimistic", "inflows", "risk-on", "recovery",
    "stabilize", "stabilizes", "resilient", "upbeat", "relief", "carry",
}
RISK_OFF = {
    "selloff", "sell-off", "crisis", "default", "contagion", "outflows", "risk-off",
    "aversion", "volatility", "plunge", "slump", "intervention", "downgrade", "war",
}

_TOKEN = re.compile(r"[a-z\-]+")


@dataclass
class SentimentScore:
    hawkish_dovish: float  # -1 dovish .. +1 hawkish
    risk_sentiment: float  # -1 risk-off .. +1 risk-on
    summary: str = ""
    drivers: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "hawkish_dovish": round(self.hawkish_dovish, 3),
            "risk_sentiment": round(self.risk_sentiment, 3),
            "summary": self.summary,
            "drivers": self.drivers,
        }


def _polarity(pos: int, neg: int) -> float:
    total = pos + neg
    return (pos - neg) / total if total else 0.0


def lexicon_score(text: str) -> SentimentScore:
    tokens = _TOKEN.findall(text.lower())
    bag = set(tokens)
    haw = [t for t in tokens if t in HAWKISH]
    dov = [t for t in tokens if t in DOVISH]
    on = [t for t in tokens if t in RISK_ON]
    off = [t for t in tokens if t in RISK_OFF]

    hawkish_dovish = _polarity(len(haw), len(dov))
    risk_sentiment = _polarity(len(on), len(off))

    tone = "hawkish" if hawkish_dovish > 0.1 else "dovish" if hawkish_dovish < -0.1 else "neutral"
    risk = "risk-on" if risk_sentiment > 0.1 else "risk-off" if risk_sentiment < -0.1 else "balanced"
    summary = f"{tone.capitalize()} tone, {risk} backdrop."
    drivers = sorted(set(haw + dov + on + off) & bag)
    return SentimentScore(hawkish_dovish, risk_sentiment, summary, drivers)


def score_text(text: str, llm: BaseLLM | None = None) -> SentimentScore:
    """Score a piece of text. Uses the LLM for a real provider; lexicon otherwise."""
    if llm is None or llm.name == "mock":
        return lexicon_score(text)

    try:
        resp = llm.complete(system=SENTIMENT_SYSTEM, messages=[{"role": "user", "content": sentiment_user(text)}])
        data = json.loads(resp.text)
        return SentimentScore(
            hawkish_dovish=float(data["hawkish_dovish"]),
            risk_sentiment=float(data["risk_sentiment"]),
            summary=str(data.get("summary", "")),
            drivers=list(data.get("drivers", [])),
        )
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        # Robust fallback if the model returns unexpected output.
        return lexicon_score(text)


def aggregate(scores: list[SentimentScore]) -> dict[str, object]:
    """Aggregate a set of headline scores into a desk-level read."""
    if not scores:
        return {"hawkish_dovish": 0.0, "risk_sentiment": 0.0, "n": 0, "summary": "No news."}
    haw = sum(s.hawkish_dovish for s in scores) / len(scores)
    risk = sum(s.risk_sentiment for s in scores) / len(scores)
    tone = "hawkish" if haw > 0.1 else "dovish" if haw < -0.1 else "neutral"
    backdrop = "risk-on" if risk > 0.1 else "risk-off" if risk < -0.1 else "mixed"
    return {
        "hawkish_dovish": round(haw, 3),
        "risk_sentiment": round(risk, 3),
        "n": len(scores),
        "summary": f"Aggregate tone {tone}, {backdrop} backdrop across {len(scores)} headlines.",
    }
