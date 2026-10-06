from fastapi import APIRouter

from rag_app import __version__
from rag_app.api.dependencies import ContainerDep, SettingsDep
from rag_app.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(settings: SettingsDep, container: ContainerDep) -> HealthResponse:
    ok = container.vector_store.healthcheck()
    return HealthResponse(
        status="ok" if ok else "degraded", version=__version__, environment=settings.environment,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model if settings.llm_provider != "fake" else "fake",
        embeddings_provider=settings.embeddings_provider,
        vector_store=f"{settings.vector_store_backend}:{settings.chroma_mode}", vector_store_ok=ok,
        checkpointer=container.checkpointer.backend, job_store=settings.job_store_backend,
        chunks=container.documents.chunk_count() if ok else 0)
