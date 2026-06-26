"""The desk co-pilot: a manual tool-use agent loop.

We run the loop ourselves (rather than the SDK's tool runner) so we keep full
control: every tool call is logged into a trace for transparency/audit, and the
loop works identically against the real Claude backend and the offline mock.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..nlp.prompts import DESK_SYSTEM
from .tools import TOOLS, CopilotContext, run_tool


@dataclass
class ToolInvocation:
    name: str
    input: dict[str, Any]
    output: dict[str, Any]


@dataclass
class CopilotAnswer:
    text: str
    trace: list[ToolInvocation] = field(default_factory=list)

    @property
    def tools_used(self) -> list[str]:
        return [t.name for t in self.trace]


class Copilot:
    """Answers desk questions by planning and executing tool calls."""

    def __init__(self, ctx: CopilotContext, max_iters: int = 5) -> None:
        self.ctx = ctx
        self.max_iters = max_iters

    def ask(self, question: str) -> CopilotAnswer:
        system = DESK_SYSTEM
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        trace: list[ToolInvocation] = []
        last_text = ""

        for _ in range(self.max_iters):
            resp = self.ctx.llm.complete(
                system=system,
                messages=messages,
                tools=TOOLS,
                thinking=True,
                max_tokens=self.ctx.settings.llm_max_tokens,
            )
            last_text = resp.text or last_text
            messages.append({"role": "assistant", "content": resp.assistant_content})

            if not resp.tool_calls:
                return CopilotAnswer(text=resp.text, trace=trace)

            results = []
            for call in resp.tool_calls:
                output = run_tool(call.name, call.input, self.ctx)
                trace.append(ToolInvocation(name=call.name, input=call.input, output=output))
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": json.dumps(output, default=str),
                    }
                )
            messages.append({"role": "user", "content": results})

        return CopilotAnswer(text=last_text or "(reached max tool iterations)", trace=trace)
