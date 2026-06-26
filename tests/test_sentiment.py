from __future__ import annotations

from emfx_copilot.evals.sentiment_evals import run_sentiment_eval
from emfx_copilot.nlp.llm import MockLLM
from emfx_copilot.nlp.sentiment import aggregate, lexicon_score, score_text


def test_lexicon_hawkish_vs_dovish():
    hawk = lexicon_score("The central bank will hike rates to fight inflation; tighter policy ahead")
    dove = lexicon_score("Policymakers ease and cut rates amid a slowdown; accommodative stance")
    assert hawk.hawkish_dovish > 0
    assert dove.hawkish_dovish < 0


def test_lexicon_risk_sentiment():
    risk_off = lexicon_score("Crisis and contagion spark a selloff with heavy outflows")
    risk_on = lexicon_score("Optimism fuels a rally and inflows; markets stabilize")
    assert risk_off.risk_sentiment < 0
    assert risk_on.risk_sentiment > 0


def test_score_text_uses_lexicon_for_mock():
    s = score_text("hawkish hike inflation", MockLLM())
    assert s.hawkish_dovish > 0


def test_aggregate_summary():
    scores = [lexicon_score("rally optimism inflows"), lexicon_score("selloff crisis outflows")]
    agg = aggregate(scores)
    assert agg["n"] == 2
    assert "backdrop" in agg["summary"]


def test_sentiment_eval_accuracy():
    report = run_sentiment_eval(MockLLM())
    # The lexicon should nail these clearly-labelled directional cases.
    assert report.hawkish_accuracy >= 0.8
    assert report.risk_accuracy >= 0.8
