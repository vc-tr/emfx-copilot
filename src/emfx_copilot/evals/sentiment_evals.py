"""A tiny labelled eval set for the sentiment scorer.

Treating prompts/scorers as testable artifacts is core to shipping LLM systems.
This module measures directional accuracy (sign of the hawkish/dovish and
risk-on/off scores) against hand-labelled headlines. It runs against whatever
backend is configured — the offline lexicon by default, or Claude if enabled —
so the same harness doubles as a regression test and a model comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..nlp.llm import BaseLLM
from ..nlp.sentiment import score_text

# (text, expected hawkish/dovish sign, expected risk sign)
# sign in {-1, 0, 1}; 0 means "neutral / not directionally labelled"
CASES: list[tuple[str, int, int]] = [
    ("Central bank signals further rate hikes as inflation stays elevated", 1, 0),
    ("Policymakers turn dovish and flag rate cuts amid a sharp slowdown", -1, 0),
    ("EM currencies rally on optimism and strong portfolio inflows", 0, 1),
    ("Risk-off selloff and contagion fears trigger heavy EM outflows", 0, -1),
    ("Hawkish surprise: the bank raises rates and warns of overheating", 1, 0),
    ("Accommodative stance as the committee eases policy to support growth", -1, 0),
    ("Relief rally as volatility fades and the currency stabilizes", 0, 1),
    ("Crisis deepens; default and downgrade risk spark a plunge", 0, -1),
    ("Bank stays patient, signals a pause and softer guidance", -1, 0),
    ("Tighter financial conditions as the central bank turns restrictive", 1, 0),
    ("Carry trades resume as upbeat data fuels a recovery", 0, 1),
    ("Sharp slump and aversion grip markets as the war escalates", 0, -1),
]


@dataclass
class EvalReport:
    n: int
    hawkish_accuracy: float
    risk_accuracy: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "hawkish_accuracy": round(self.hawkish_accuracy, 3),
            "risk_accuracy": round(self.risk_accuracy, 3),
        }


def _sign(x: float, thresh: float = 0.1) -> int:
    return 1 if x > thresh else -1 if x < -thresh else 0


def run_sentiment_eval(llm: BaseLLM | None = None) -> EvalReport:
    haw_hits = 0
    haw_total = 0
    risk_hits = 0
    risk_total = 0

    for text, exp_haw, exp_risk in CASES:
        score = score_text(text, llm)
        if exp_haw != 0:
            haw_total += 1
            haw_hits += int(_sign(score.hawkish_dovish) == exp_haw)
        if exp_risk != 0:
            risk_total += 1
            risk_hits += int(_sign(score.risk_sentiment) == exp_risk)

    return EvalReport(
        n=len(CASES),
        hawkish_accuracy=haw_hits / haw_total if haw_total else 0.0,
        risk_accuracy=risk_hits / risk_total if risk_total else 0.0,
    )
