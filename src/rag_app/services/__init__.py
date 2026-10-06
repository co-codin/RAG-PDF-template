"""Use-cases. Framework-agnostic: no FastAPI imports; depend on Protocols, return domain models."""

from rag_app.services.document_service import DocumentService
from rag_app.services.ingestion_service import IngestionService
from rag_app.services.query_service import QueryService

__all__ = ["DocumentService", "IngestionService", "QueryService"]
