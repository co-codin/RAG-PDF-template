# syntax=docker/dockerfile:1.7

# ---- Builder: install deps + the rag_app package into /app/.venv with uv --------
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

# Optional extras, e.g. `--build-arg UV_EXTRAS=postgres` for the Postgres checkpointer.
ARG UV_EXTRAS=""

WORKDIR /app

# 1) Dependencies only (cached layer; invalidated only when the lockfile changes).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --locked --no-install-project --no-dev ${UV_EXTRAS:+--extra $UV_EXTRAS}

# 2) The project itself, installed non-editable (runtime needs only .venv).
COPY pyproject.toml uv.lock .python-version README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable ${UV_EXTRAS:+--extra $UV_EXTRAS}


# ---- Runtime: slim, non-root, no build tooling ---------------------------------------
FROM python:3.13-slim-bookworm AS runtime

RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app \
 && mkdir -p /data && chown app:app /data

WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app data ./data

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/data \
    LOG_JSON=true \
    # uvicorn reads WEB_CONCURRENCY as its worker count. Keep 1 with embedded Chroma;
    # raise it when CHROMA_HOST points at a Chroma server.
    WEB_CONCURRENCY=1

USER app
EXPOSE 8000
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "rag_app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
