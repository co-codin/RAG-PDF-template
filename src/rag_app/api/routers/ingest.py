"""Ingestion endpoints. Work runs as a background job; clients poll the job URL.

FastAPI BackgroundTasks run in the same process after the response is sent. For heavy
or bursty ingestion move ``run_*_job`` to a queue worker (Celery/RQ/Arq) - the service
API stays the same.
"""

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile, status

from rag_app.api.dependencies import IngestionServiceDep
from rag_app.api.schemas import JobAccepted, JobResponse, UrlIngestRequest
from rag_app.core.errors import IngestionError, NotFoundError
from rag_app.domain.models import IngestResult

router = APIRouter(prefix="/ingest", tags=["ingestion"])


def _accepted(request: Request, job, accepted: list[str], rejected: list[IngestResult]) -> JobAccepted:
    return JobAccepted(job_id=job.id, status=job.status, accepted=accepted, rejected=rejected,
                       status_url=str(request.url_for("get_ingest_job", job_id=job.id).path))


@router.post("/files", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
def ingest_files(request: Request, background: BackgroundTasks, svc: IngestionServiceDep,
                 files: list[UploadFile] = File(...)) -> JobAccepted:
    max_bytes = svc.settings.max_upload_mb * 1024 * 1024
    payloads: list[tuple[str, bytes]] = []
    rejected: list[IngestResult] = []
    for f in files:
        name = svc.safe_filename(f.filename)
        data = f.file.read(max_bytes + 1)
        try:
            svc.validate_upload(name, len(data))
            payloads.append((name, data))
        except IngestionError as exc:
            rejected.append(IngestResult(source=name, status="error", error=str(exc)))
    if not payloads:
        raise HTTPException(status_code=400, detail="; ".join(r.error or "" for r in rejected))
    job = svc.create_job("files", [n for n, _ in payloads], rejected)
    background.add_task(svc.run_files_job, job.id, payloads)
    return _accepted(request, job, [n for n, _ in payloads], rejected)


@router.post("/url", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
def ingest_url(request: Request, body: UrlIngestRequest, background: BackgroundTasks,
               svc: IngestionServiceDep) -> JobAccepted:
    svc.validate_url(body.url)  # fail fast (400) on invalid / private URLs
    job = svc.create_job("url", [body.url])
    background.add_task(svc.run_url_job, job.id, body.url)
    return _accepted(request, job, [body.url], [])


@router.get("/jobs/{job_id}", response_model=JobResponse, name="get_ingest_job")
def get_ingest_job(job_id: str, svc: IngestionServiceDep) -> JobResponse:
    job = svc.get_job(job_id)
    if job is None:
        raise NotFoundError(f"Unknown job '{job_id}'")
    return JobResponse(**job.model_dump())
