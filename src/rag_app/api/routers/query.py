"""Question answering: blocking JSON and Server-Sent Events streaming."""

import json
import logging
import time
import uuid
from collections.abc import Iterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from rag_app.api.dependencies import QueryServiceDep
from rag_app.api.schemas import QueryRequest, QueryResponse
from rag_app.core.logging import request_id_var

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, svc: QueryServiceDep) -> QueryResponse:
    session_id = body.session_id or uuid.uuid4().hex
    t0 = time.perf_counter()
    result = svc.query(body.question, session_id)
    latency = round((time.perf_counter() - t0) * 1000, 1)
    logger.info("query_answered", extra={"session_id": session_id, "latency_ms": latency,
                                         "sources": len(result.sources)})
    return QueryResponse(**result.model_dump(), session_id=session_id, latency_ms=latency)


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@router.post("/query/stream")
def query_stream(body: QueryRequest, svc: QueryServiceDep) -> StreamingResponse:
    session_id = body.session_id or uuid.uuid4().hex
    request_id = request_id_var.get()

    def events() -> Iterator[str]:
        request_id_var.set(request_id)  # middleware context is gone once the body streams
        t0 = time.perf_counter()
        yield sse("meta", {"session_id": session_id, "request_id": request_id})
        try:
            for event, payload in svc.stream(body.question, session_id):
                if event == "done":
                    payload["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                    logger.info("stream_completed", extra={"session_id": session_id,
                                                           "latency_ms": payload["latency_ms"]})
                yield sse(event, payload)
        except Exception as exc:  # the stream already started: report in-band
            logger.exception("stream_failed", extra={"session_id": session_id})
            yield sse("error", {"error": type(exc).__name__, "detail": str(exc)[:500]})

    return StreamingResponse(events(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Session-ID": session_id})
