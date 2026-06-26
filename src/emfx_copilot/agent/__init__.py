"""Agent: the desk co-pilot, its tool surface, and the briefing generator."""

from __future__ import annotations

from .briefing import Briefing, build_briefing
from .copilot import Copilot, CopilotAnswer, ToolInvocation
from .tools import TOOLS, CopilotContext, pretrade, run_tool

__all__ = [
    "Copilot",
    "CopilotAnswer",
    "ToolInvocation",
    "CopilotContext",
    "TOOLS",
    "run_tool",
    "pretrade",
    "Briefing",
    "build_briefing",
]
