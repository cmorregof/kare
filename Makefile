.PHONY: install test lint format typecheck check ingest status

install:
	uv sync --group dev

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy app/

check: lint typecheck test

ingest:
	uv run kare ingerir

status:
	uv run kare estado
