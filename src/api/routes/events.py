from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from src.api.dependencies import TenantContext, get_tenant_context
from src.api.schemas import (
    EventCandidateResponse,
    EventReviewRequest,
    EventReviewResponse,
    JobEventsResponse,
)
from src.db.models import EventCandidate, EventReview, ProcessingJob
from src.db.session import get_db
from src.schemas import ReviewStatus


router = APIRouter(tags=["events"])


@router.get("/jobs/{job_id}/events", response_model=JobEventsResponse)
def list_job_events(
    job_id: UUID,
    context: TenantContext = Depends(get_tenant_context),
    session: Session = Depends(get_db),
) -> JobEventsResponse:
    job = session.scalar(
        select(ProcessingJob.id).where(
            ProcessingJob.id == job_id,
            ProcessingJob.tenant_id == context.tenant_id,
        )
    )
    if job is None:
        raise HTTPException(status_code=404, detail="job_not_found")

    rows = session.execute(
        select(EventCandidate, EventReview.status)
        .outerjoin(
            EventReview,
            (EventReview.event_id == EventCandidate.id)
            & (EventReview.tenant_id == EventCandidate.tenant_id),
        )
        .where(
            EventCandidate.job_id == job_id,
            EventCandidate.tenant_id == context.tenant_id,
        )
        .order_by(EventCandidate.video_time_seconds, EventCandidate.id)
    ).all()
    items = [
        EventCandidateResponse(
            id=event.id,
            job_id=event.job_id,
            video_id=event.video_id,
            event_type=event.event_type,
            track_id=event.track_id,
            frame_index=event.frame_index,
            video_time_seconds=event.video_time_seconds,
            confidence=event.confidence,
            bbox=event.bbox,
            detector_notes=event.detector_notes,
            detected_at=event.detected_at,
            review_status=review_status or ReviewStatus.PENDING_REVIEW,
        )
        for event, review_status in rows
    ]
    return JobEventsResponse(items=items)


@router.post("/events/{event_id}/review", response_model=EventReviewResponse)
def review_event(
    event_id: UUID,
    payload: EventReviewRequest,
    context: TenantContext = Depends(get_tenant_context),
    session: Session = Depends(get_db),
) -> EventReviewResponse:
    event = session.scalar(
        select(EventCandidate).where(
            EventCandidate.id == event_id,
            EventCandidate.tenant_id == context.tenant_id,
        )
    )
    if event is None:
        raise HTTPException(status_code=404, detail="event_not_found")

    # Initialize absent review state idempotently; candidate data remains untouched.
    session.execute(
        pg_insert(EventReview)
        .values(
            id=uuid4(),
            event_id=event_id,
            tenant_id=context.tenant_id,
            status=ReviewStatus.PENDING_REVIEW.value,
            revision=0,
        )
        .on_conflict_do_nothing(index_elements=[EventReview.event_id])
    )
    review = session.scalar(
        select(EventReview)
        .where(
            EventReview.event_id == event_id,
            EventReview.tenant_id == context.tenant_id,
        )
        .with_for_update()
    )
    if review is None:
        session.rollback()
        raise HTTPException(status_code=404, detail="event_not_found")
    if review.revision != payload.expected_revision:
        session.rollback()
        raise HTTPException(status_code=409, detail="review_revision_conflict")

    updated = session.execute(
        update(EventReview)
        .where(
            EventReview.event_id == event_id,
            EventReview.tenant_id == context.tenant_id,
            EventReview.revision == payload.expected_revision,
        )
        .values(
            status=payload.status.value,
            notes=payload.notes,
            reviewer_user_id=context.user_id,
            revision=EventReview.revision + 1,
        )
        .returning(EventReview)
    ).scalar_one_or_none()
    if updated is None:
        session.rollback()
        raise HTTPException(status_code=409, detail="review_revision_conflict")
    session.commit()
    session.refresh(updated)
    return EventReviewResponse(
        event_id=updated.event_id,
        status=updated.status,
        notes=updated.notes,
        reviewer_user_id=updated.reviewer_user_id,
        revision=updated.revision,
        updated_at=updated.updated_at,
    )
