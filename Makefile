.PHONY: install test lint fmt demo serve clean

# Set up a local environment (uv recommended; falls back to venv + pip).
install:
	uv venv && uv pip install -e ".[dev,api,llm]"

test:
	pytest -q

lint:
	ruff check .

fmt:
	ruff check --fix . && ruff format .

# Full offline walkthrough: regime -> signals -> pre-trade -> risk -> briefing.
demo:
	emfx demo

# Serve the analytics over HTTP on :8000.
serve:
	emfx serve

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__ mlruns
