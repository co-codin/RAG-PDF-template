"""FastAPI application factory.

Run:  uv run uvicorn rag_app.main:app            (or --factory rag_app.main:create_app)
The app is stateless apart from the per-process Container; all durable state lives in
the vector store, checkpointer and job store, so workers/replicas can be added freely
once those are shared backends (Chroma server, Postgres, Redis/SQLite).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_app import __version__
from rag_app.api.dependencies import require_api_key
from rag_app.api.middleware import request_context_middleware
from rag_app.api.routers import documents, health, ingest, query, sessions
from rag_app.container import Container, build_container
from rag_app.core.config import Settings, get_settings
from rag_app.core.errors import register_exception_handlers
from rag_app.core.logging import setup_logging

logger = logging.getLogger("rag_app")


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owns = container is None
        app.state.container = container or build_container(settings)
        if settings.chroma_mode == "embedded":
            logger.info("embedded_chroma_single_process",
                        extra={"hint": "use CHROMA_HOST (Chroma server) before running >1 worker"})
        logger.info("startup", extra={"version": __version__, "auth_enabled": settings.api_key is not None})
        try:
            yield
        finally:
            if owns:
                app.state.container.close()
            logger.info("shutdown")

    app = FastAPI(title="rag-production", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                           allow_methods=["*"], allow_headers=["*"])
    app.middleware("http")(request_context_middleware)
    register_exception_handlers(app)

    app.include_router(health.router)  # public
    protected = [Depends(require_api_key)]
    for r in (ingest.router, query.router, documents.router, sessions.router):
        app.include_router(r, dependencies=protected)
    return app


app = create_app()
