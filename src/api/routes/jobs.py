from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.dependencies import TenantContext, get_tenant_context
from src.api.schemas import JobCreateRequest, JobResponse
from src.db.models import ProcessingJob, Video
from src.db.session import get_db


router = APIRouter(prefix="/jobs", tags=["jobs"])


def _job_response(job: ProcessingJob) -> JobResponse:
    return JobResponse(
        id=job.id,
        video_id=job.video_id,
        status=job.status,
        processed_frames=job.processed_frames,
        estimated_total_frames=job.estimated_total_frames,
        progress_percent=job.progress_percent,
        attempt_count=job.attempt_count,
        error_code=job.error_code,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@router.post("", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
def create_job(
    payload: JobCreateRequest,
    context: TenantContext = Depends(get_tenant_context),
    session: Session = Depends(get_db),
) -> JobResponse:
    video = session.scalar(
        select(Video).where(
            Video.id == payload.video_id,
            Video.tenant_id == context.tenant_id,
        )
    )
    if video is None:
        raise HTTPException(status_code=404, detail="video_not_found")
    if video.status != "ready":
        raise HTTPException(status_code=409, detail="video_not_ready")

    job = ProcessingJob(
        id=uuid4(),
        tenant_id=context.tenant_id,
        video_id=video.id,
        requested_by_user_id=context.user_id,
        status="created",
        options=payload.options.model_dump(mode="json"),
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    # Phase 1A records a request only; it intentionally has no dispatcher.
    return _job_response(job)


@router.get("/{job_id}", response_model=JobResponse)
def get_job(
    job_id: UUID,
    context: TenantContext = Depends(get_tenant_context),
    session: Session = Depends(get_db),
) -> JobResponse:
    job = session.scalar(
        select(ProcessingJob).where(
            ProcessingJob.id == job_id,
            ProcessingJob.tenant_id == context.tenant_id,
        )
    )
    if job is None:
        raise HTTPException(status_code=404, detail="job_not_found")
    return _job_response(job)
