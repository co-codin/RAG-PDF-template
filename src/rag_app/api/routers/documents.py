from fastapi import APIRouter

from rag_app.api.dependencies import DocumentServiceDep
from rag_app.api.schemas import DeleteResponse, DocumentsResponse

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=DocumentsResponse)
def list_documents(svc: DocumentServiceDep) -> DocumentsResponse:
    docs = svc.list()
    return DocumentsResponse(documents=docs, total_chunks=sum(d.chunks for d in docs))


@router.delete("/{source:path}", response_model=DeleteResponse)
def delete_document(source: str, svc: DocumentServiceDep) -> DeleteResponse:
    return DeleteResponse(source=source, deleted_chunks=svc.delete(source))  # 404 via NotFoundError
