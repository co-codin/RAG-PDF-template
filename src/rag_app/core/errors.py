"""Application error hierarchy + FastAPI exception handlers.

Lower layers raise ``AppError`` subclasses (no HTTP knowledge); the API maps them
to status codes and a consistent JSON body: {"error", "detail", "request_id"}.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from rag_app.core.logging import request_id_var

logger = logging.getLogger(__name__)


class AppError(Exception):
    status_code = 500
    code = "app_error"


class IngestionError(AppError, ValueError):
    """User-facing ingestion problem: bad file type, unparsable file, fetch failure."""
    status_code = 400
    code = "ingestion_error"


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ConfigurationError(AppError):
    status_code = 500
    code = "configuration_error"


def error_response(status: int, error: str, detail: str | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={
        "error": error, "detail": detail, "request_id": request_id_var.get()})


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def _http(_: Request, exc: HTTPException):
        return error_response(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        return error_response(422, "validation_error", str(exc.errors())[:2000])

    @app.exception_handler(AppError)
    async def _app(_: Request, exc: AppError):
        if exc.status_code >= 500:
            logger.error("app_error", exc_info=exc)
        return error_response(exc.status_code, exc.code, str(exc))
