# Simple task runner. Everything goes through uv, so `make` is optional sugar.
.DEFAULT_GOAL := help
PORT ?= 8000
WORKERS ?= 2
CHROMA_PORT ?= 8001

.PHONY: help install test test-unit test-integration run run-workers chroma ingest docker-build up down tarball clean

help: ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

install: ## Create .venv from uv.lock
	uv sync

test: ## Run all tests (offline)
	uv run pytest

test-unit: ## Unit tests only
	uv run pytest tests/unit

test-integration: ## Integration tests (TestClient, embedded + local Chroma server)
	uv run pytest tests/integration

run: ## Dev server, embedded Chroma, single process, auto-reload
	uv run uvicorn rag_app.main:app --reload --port $(PORT)

chroma: ## Local Chroma server for multi-worker mode (bundled chroma CLI)
	uv run chroma run --path storage/chroma-server --port $(CHROMA_PORT)

run-workers: ## Multi-worker API against the Chroma server started by `make chroma`
	CHROMA_HOST=127.0.0.1 CHROMA_PORT=$(CHROMA_PORT) JOB_STORE_BACKEND=sqlite \
	uv run uvicorn rag_app.main:app --workers $(WORKERS) --port $(PORT)

ingest: ## Bulk ingest ./data (stop `make run` first when using embedded Chroma)
	uv run python scripts/ingest_dir.py data/

docker-build: ## Build the image
	docker build -t rag-production:latest .

up: ## docker compose: api + chroma server
	docker compose up --build -d

down: ## Stop compose stack
	docker compose down

tarball: ## Source tarball without venv, secrets or data
	cd .. && tar --exclude='rag-production/.venv' --exclude='rag-production/.env' \
	  --exclude='rag-production/storage' --exclude='rag-production/.git' \
	  --exclude='__pycache__' --exclude='.pytest_cache' -czf rag-production.tar.gz rag-production

clean: ## Remove caches (keeps storage/)
	find . -name __pycache__ -type d -prune -exec rm -rf {} + ; rm -rf .pytest_cache
