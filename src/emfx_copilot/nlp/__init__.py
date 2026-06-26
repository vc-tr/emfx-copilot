"""NLP: LLM backends and market-text sentiment scoring."""

from __future__ import annotations

from .llm import AnthropicLLM, BaseLLM, LLMResponse, MockLLM, ToolCall, get_llm
from .sentiment import SentimentScore, aggregate, lexicon_score, score_text

__all__ = [
    "BaseLLM",
    "MockLLM",
    "AnthropicLLM",
    "LLMResponse",
    "ToolCall",
    "get_llm",
    "SentimentScore",
    "score_text",
    "lexicon_score",
    "aggregate",
]
