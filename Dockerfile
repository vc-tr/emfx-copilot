# syntax=docker/dockerfile:1

# EM-FX Desk Co-pilot — container image for the FastAPI service.
# Runs fully offline with the deterministic mock LLM (no API key required).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    EMFX_LLM_PROVIDER=mock

WORKDIR /app

# uv for fast, reproducible installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install the package (+ api extra) from source.
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system --no-cache ".[api]"

# Drop privileges.
RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status == 200 else 1)"

CMD ["uvicorn", "emfx_copilot.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
