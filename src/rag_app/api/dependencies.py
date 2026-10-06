"""FastAPI dependency injection: services come from the per-process Container stored
on ``app.state`` (built in the lifespan), so routers never construct backends."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from rag_app.container import Container
from rag_app.core.config import Settings
from rag_app.core.security import verify_api_key
from rag_app.services import DocumentService, IngestionService, QueryService


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_ingestion_service(c: Annotated[Container, Depends(get_container)]) -> IngestionService:
    return c.ingestion


def get_query_service(c: Annotated[Container, Depends(get_container)]) -> QueryService:
    return c.query


def get_document_service(c: Annotated[Container, Depends(get_container)]) -> DocumentService:
    return c.documents


def require_api_key(request: Request, settings: Annotated[Settings, Depends(get_settings)]) -> None:
    if not verify_api_key(settings.api_key, request.headers.get(settings.api_key_header)):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail=f"Missing or invalid {settings.api_key_header} header")


SettingsDep = Annotated[Settings, Depends(get_settings)]
ContainerDep = Annotated[Container, Depends(get_container)]
IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]
QueryServiceDep = Annotated[QueryService, Depends(get_query_service)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
