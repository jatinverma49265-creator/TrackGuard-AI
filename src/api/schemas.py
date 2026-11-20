"""HTTP-specific Pydantic contracts; identity and storage keys are server-owned."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import TrackBoundary
from src.schemas import EventType, ReviewStatus

MAX_VIDEO_UPLOAD_BYTES = 50 * 1024 * 1024


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VideoCreateRequest(RequestModel):
    original_filename: str = Field(min_length=1, max_length=255)
    declared_content_type: Literal["video/mp4", "video/quicktime", "video/x-msvideo", "video/x-matroska"]
    size_bytes: int = Field(gt=0, le=MAX_VIDEO_UPLOAD_BYTES)

    @field_validator("original_filename")
    @classmethod
    def reject_path_separators(cls, value: str) -> str:
        if "/" in value or "\\" in value or value in {".", ".."}:
            raise ValueError("Filename must be a name, not a path")
        if not value.strip():
            raise ValueError("Filename cannot be blank")
        return value


class VideoResponse(BaseModel):
    id: UUID
    original_filename: str
    declared_content_type: str
    size_bytes: int
    status: str
    created_at: datetime


class VideoUploadResponse(BaseModel):
    video_id: UUID
    upload_url: str
    upload_method: Literal["PUT"] = "PUT"
    required_headers: dict[str, str]
    expires_at: datetime


class VideoListResponse(BaseModel):
    items: list[VideoResponse]
    limit: int
    offset: int


class JobOptions(RequestModel):
    confidence_threshold: float = Field(default=0.35, ge=0, le=1)
    iou_threshold: float = Field(default=0.45, ge=0, le=1)
    track_boundary: TrackBoundary | None = None
    boundary_hysteresis_pixels: float = Field(default=5.0, ge=0)
    evidence_pre_seconds: float = Field(default=3.0, ge=0)
    evidence_post_seconds: float = Field(default=3.0, ge=0)
    enable_tracking: bool = True


class JobCreateRequest(RequestModel):
    video_id: UUID
    options: JobOptions = Field(default_factory=JobOptions)


class JobResponse(BaseModel):
    id: UUID
    video_id: UUID
    status: str
    processed_frames: int
    estimated_total_frames: int | None
    progress_percent: int | None
    attempt_count: int
    error_code: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class EventCandidateResponse(BaseModel):
    id: UUID
    job_id: UUID
    video_id: UUID
    event_type: EventType
    track_id: int | None
    frame_index: int | None
    video_time_seconds: float | None
    confidence: float | None
    bbox: dict | None
    detector_notes: str | None
    detected_at: datetime
    review_status: ReviewStatus


class JobEventsResponse(BaseModel):
    items: list[EventCandidateResponse]


class EventReviewRequest(RequestModel):
    status: ReviewStatus
    notes: str | None = Field(default=None, max_length=1000)
    expected_revision: int = Field(ge=0)


class EventReviewResponse(BaseModel):
    event_id: UUID
    status: ReviewStatus
    notes: str | None
    reviewer_user_id: UUID
    revision: int
    updated_at: datetime


class MeResponse(BaseModel):
    user_id: UUID
    email: str | None
    tenant_id: UUID


class HealthResponse(BaseModel):
    status: str
