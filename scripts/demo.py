#!/usr/bin/env python3
"""Thin wrapper so the demo can be run as a script: ``python scripts/demo.py``.

Equivalent to ``emfx demo``.
"""

from __future__ import annotations

from emfx_copilot.demo import run_demo

if __name__ == "__main__":
    run_demo()
