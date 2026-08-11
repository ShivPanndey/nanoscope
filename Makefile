.PHONY: setup test lint typecheck bench run check clean

setup:
	uv sync --all-groups

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run pytest -m "not slow"

# Everything CI runs, so a green local check means a green pipeline.
check: lint typecheck test

bench:
	PYTORCH_ENABLE_MPS_FALLBACK=0 uv run nanoscope bench

run:
	uv run nanoscope --help

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
