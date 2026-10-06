from typing import Annotated

from fastapi import APIRouter, Path

from rag_app.api.dependencies import QueryServiceDep
from rag_app.api.schemas import SESSION_ID_PATTERN, HistoryResponse

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("/{session_id}/history", response_model=HistoryResponse)
def session_history(session_id: Annotated[str, Path(pattern=SESSION_ID_PATTERN)],
                    svc: QueryServiceDep) -> HistoryResponse:
    return HistoryResponse(session_id=session_id, messages=svc.history(session_id))
