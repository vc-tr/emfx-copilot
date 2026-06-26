"""LLM abstraction with a real Claude backend and a deterministic offline mock.

``BaseLLM.complete`` is a thin, provider-agnostic surface that supports both a
single completion (sentiment scoring, briefing prose) and a tool-using agent
turn. Messages and tool specs follow the Anthropic Messages API shape so the
agent loop is identical across backends:

* ``AnthropicLLM`` calls Claude through the official ``anthropic`` SDK
  (manual tool-use loop, adaptive thinking, effort control).
* ``MockLLM`` is fully offline and deterministic — it plans a sensible set of
  tool calls and then composes a desk-style answer from the tool results, so
  the agent, demo, and tests all run with no API key.

``get_llm`` returns the configured backend, falling back to the mock whenever
the Anthropic SDK or an API key is missing.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings, get_settings

logger = logging.getLogger(__name__)

# A type alias for the loosely-typed message list shared across backends.
Message = dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    # Provider-native assistant content to append back into the conversation
    # (raw SDK blocks for Anthropic; generic dict blocks for the mock).
    assistant_content: Any = None


class BaseLLM:
    name: str = "base"

    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        thinking: bool = False,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Anthropic (Claude) backend
# ---------------------------------------------------------------------------
class AnthropicLLM(BaseLLM):
    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        api_key: str | None = None,
        max_tokens: int = 4096,
        effort: str = "medium",
    ) -> None:
        import anthropic  # imported lazily so the package works without the extra

        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort

    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        thinking: bool = False,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "system": system,
            "messages": messages,
            "output_config": {"effort": self.effort},
        }
        if tools:
            kwargs["tools"] = tools
        if thinking:
            # Adaptive thinking is the only supported on-mode for Opus 4.7/4.8.
            kwargs["thinking"] = {"type": "adaptive"}

        resp = self._client.messages.create(**kwargs)

        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        tool_calls = [
            ToolCall(id=b.id, name=b.name, input=dict(b.input))
            for b in resp.content
            if getattr(b, "type", None) == "tool_use"
        ]
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=resp.stop_reason or "end_turn",
            assistant_content=resp.content,  # pass thinking/tool_use blocks back unchanged
        )


# ---------------------------------------------------------------------------
# Deterministic offline backend
# ---------------------------------------------------------------------------
class MockLLM(BaseLLM):
    """Offline, deterministic stand-in used for the demo and tests.

    For agent turns it plans a standard tool set on the first call, then on the
    follow-up (once tool results are present) composes a desk-style answer from
    those results. This is *not* a language model — it is a scripted harness so
    the end-to-end flow exercises every tool and produces sensible output with
    no network access.
    """

    name = "mock"

    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        thinking: bool = False,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if not tools:
            return LLMResponse(
                text="(mock) No tools available; provide a real LLM provider for free-form replies.",
                assistant_content=[{"type": "text", "text": "(mock) acknowledged"}],
            )

        results = self._collect_results(messages)
        if results:
            answer = self._compose_answer(self._first_question(messages), results)
            return LLMResponse(text=answer, assistant_content=[{"type": "text", "text": answer}])

        plan = self._plan_tools(self._first_question(messages), tools)
        assistant_content = [
            {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input} for tc in plan
        ]
        return LLMResponse(text="", tool_calls=plan, stop_reason="tool_use", assistant_content=assistant_content)

    # --- helpers -----------------------------------------------------------
    @staticmethod
    def _first_question(messages: list[Message]) -> str:
        for m in messages:
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                return m["content"]
        return ""

    @staticmethod
    def _plan_tools(question: str, tools: list[dict[str, Any]]) -> list[ToolCall]:
        from ..data.universe import CODES

        available = {t["name"] for t in tools}
        plan: list[ToolCall] = []

        def add(name: str, payload: dict[str, Any]) -> None:
            if name in available:
                plan.append(ToolCall(id=f"mock_{name}", name=name, input=payload))

        add("detect_regime", {})
        add("compute_signals", {})
        add("assess_risk", {})

        mentioned = [c for c in CODES if c.lower() in question.lower()]
        if mentioned:
            add("price_forward", {"ccy": mentioned[0], "tenor_months": 3})

        return plan or [ToolCall(id="mock_snapshot", name=next(iter(available)), input={})]

    @staticmethod
    def _collect_results(messages: list[Message]) -> dict[str, Any]:
        """Map tool name -> parsed result, using the assistant tool_use blocks
        to recover names for each tool_result id."""
        id_to_name: dict[str, str] = {}
        for m in messages:
            content = m.get("content")
            if m.get("role") == "assistant" and isinstance(content, list):
                for block in content:
                    btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
                    if btype == "tool_use":
                        bid = block["id"] if isinstance(block, dict) else block.id
                        bname = block["name"] if isinstance(block, dict) else block.name
                        id_to_name[bid] = bname

        results: dict[str, Any] = {}
        for m in messages:
            content = m.get("content")
            if m.get("role") == "user" and isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        name = id_to_name.get(block.get("tool_use_id", ""), block.get("tool_use_id", "?"))
                        try:
                            results[name] = json.loads(block.get("content", "{}"))
                        except (json.JSONDecodeError, TypeError):
                            results[name] = block.get("content")
        return results

    @staticmethod
    def _compose_answer(question: str, results: dict[str, Any]) -> str:
        lines: list[str] = []
        regime = results.get("detect_regime")
        if isinstance(regime, dict):
            lines.append(
                f"Regime: {regime.get('label', '?')} "
                f"(risk-off probability {regime.get('risk_off_prob', '?')}, "
                f"20d EM index return {regime.get('index_return_20d', '?')})."
            )

        signals = results.get("compute_signals")
        if isinstance(signals, dict):
            longs = ", ".join(signals.get("top_longs", [])) or "n/a"
            shorts = ", ".join(signals.get("top_shorts", [])) or "n/a"
            lines.append(f"Signal book leans long {longs}; short {shorts}.")

        risk = results.get("assess_risk")
        if isinstance(risk, dict):
            var = risk.get("var_usd_1d_99")
            gross = risk.get("gross_usd")
            lines.append(
                f"Risk: 1-day 99% VaR ${var:,.0f} on ${gross:,.0f} gross."
                if isinstance(var, (int, float)) and isinstance(gross, (int, float))
                else "Risk metrics computed."
            )
            breaches = risk.get("breaches") or []
            if breaches:
                lines.append(f"Limit alerts: {len(breaches)} breach(es) flagged.")

        fwd = results.get("price_forward")
        if isinstance(fwd, dict) and "code" in fwd:
            lines.append(
                f"{fwd['code']} {fwd.get('tenor_months', '?')}m "
                f"{fwd.get('instrument', 'forward')}: {fwd.get('forward')} "
                f"(points {fwd.get('points')}, ann. premium {fwd.get('annualized_premium')})."
            )

        if not lines:
            lines.append("Tools returned no actionable data.")

        header = "Desk read"
        if question:
            header = f'Desk read for "{question.strip()[:80]}"'
        return header + ":\n- " + "\n- ".join(lines)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_llm(settings: Settings | None = None) -> BaseLLM:
    settings = settings or get_settings()
    if settings.llm_provider.lower() == "anthropic":
        api_key = settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        try:
            import anthropic  # noqa: F401
        except ImportError:
            logger.warning("anthropic provider requested but SDK not installed; using MockLLM. "
                           "Install with: pip install 'emfx-copilot[llm]'")
            return MockLLM()
        if not api_key:
            logger.warning("anthropic provider requested but no API key found; using MockLLM.")
            return MockLLM()
        return AnthropicLLM(
            model=settings.llm_model,
            api_key=api_key,
            max_tokens=settings.llm_max_tokens,
            effort=settings.llm_effort,
        )
    return MockLLM()
