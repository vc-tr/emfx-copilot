"""HTTP API (FastAPI). Import lazily — requires the ``api`` extra."""

from __future__ import annotations

__all__ = ["create_app"]


def create_app():  # pragma: no cover - thin lazy wrapper
    from .app import create_app as _create_app

    return _create_app()
