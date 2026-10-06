"""Request context middleware: request id, timing headers, access log, last-resort 500."""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse

from rag_app.core.logging import request_id_var

logger = logging.getLogger("rag_app.access")


async def request_context_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
    token = request_id_var.set(rid)
    t0 = time.perf_counter()
    status_code = 500
    try:
        try:
            response = await call_next(request)
        except Exception:
            # Handled here (not in Starlette's outer ServerErrorMiddleware) so the
            # response and log line still carry the request id.
            logger.exception("unhandled_error")
            response = JSONResponse(status_code=500, content={
                "error": "internal_error", "detail": "An unexpected error occurred", "request_id": rid})
        status_code = response.status_code
        response.headers["X-Request-ID"] = rid
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
        return response
    finally:
        logger.info("request", extra={
            "method": request.method, "path": request.url.path, "status": status_code,
            "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
            "client": request.client.host if request.client else None})
        request_id_var.reset(token)
