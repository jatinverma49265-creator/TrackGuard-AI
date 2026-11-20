"""SQLAlchemy models for tenant-scoped TrackGuard records."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    JSON,
    String,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    email: Mapped[str | None] = mapped_column(String(320))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")


class Tenant(Base, TimestampMixin):
    __tablename__ = "tenants"
    __table_args__ = (
        UniqueConstraint("owner_user_id", name="uq_tenants_owner_user"),
        UniqueConstraint("id", "owner_user_id", name="uq_tenants_id_owner"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    owner_user_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(160), nullable=False)


class Video(Base, TimestampMixin):
    __tablename__ = "videos"
    __table_args__ = (
        UniqueConstraint("id", "tenant_id", name="uq_videos_id_tenant"),
        ForeignKeyConstraint(
            ["tenant_id", "created_by_user_id"],
            ["tenants.id", "tenants.owner_user_id"],
            ondelete="RESTRICT",
            name="fk_videos_creator_tenant_owner",
        ),
        UniqueConstraint("storage_key", name="uq_videos_storage_key"),
        CheckConstraint("size_bytes > 0 AND size_bytes <= 524288000", name="ck_videos_size"),
        CheckConstraint(
            "status IN ('pending_upload', 'ready', 'failed', 'deleted')",
            name="ck_videos_status",
        ),
        Index("ix_videos_tenant_created", "tenant_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    tenant_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False
    )
    created_by_user_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    declared_content_type: Mapped[str] = mapped_column(String(127), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    checksum_sha256: Mapped[str | None] = mapped_column(String(64))
    storage_key: Mapped[str | None] = mapped_column(String(1024))
    upload_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(32), nullable=False, server_default="pending_upload")
    duration_seconds: Mapped[float | None]
    width: Mapped[int | None]
    height: Mapped[int | None]
    source_fps: Mapped[float | None]


class ProcessingJob(Base, TimestampMixin):
    __tablename__ = "processing_jobs"
    __table_args__ = (
        UniqueConstraint("id", "tenant_id", name="uq_jobs_id_tenant"),
        UniqueConstraint("id", "video_id", "tenant_id", name="uq_jobs_id_video_tenant"),
        UniqueConstraint("tenant_id", "idempotency_key", name="uq_jobs_tenant_idempotency"),
        ForeignKeyConstraint(
            ["video_id", "tenant_id"],
            ["videos.id", "videos.tenant_id"],
            ondelete="RESTRICT",
            name="fk_jobs_video_tenant",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "requested_by_user_id"],
            ["tenants.id", "tenants.owner_user_id"],
            ondelete="RESTRICT",
            name="fk_jobs_requester_tenant_owner",
        ),
        CheckConstraint(
            "status IN ('created', 'queued', 'running', 'finalizing', 'succeeded', 'failed', 'cancelled')",
            name="ck_jobs_status",
        ),
        CheckConstraint("progress_percent IS NULL OR progress_percent BETWEEN 0 AND 100", name="ck_jobs_progress"),
        CheckConstraint("attempt_count >= 0", name="ck_jobs_attempt_count"),
        Index("ix_jobs_tenant_created", "tenant_id", "created_at"),
        Index("ix_jobs_tenant_status_created", "tenant_id", "status", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    tenant_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False
    )
    video_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    requested_by_user_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, server_default="created")
    options: Mapped[dict] = mapped_column(JSON, nullable=False)
    progress_percent: Mapped[int | None] = mapped_column(Integer)
    processed_frames: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    estimated_total_frames: Mapped[int | None] = mapped_column(Integer)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    idempotency_key: Mapped[str | None] = mapped_column(String(128))
    error_code: Mapped[str | None] = mapped_column(String(80))
    error_message: Mapped[str | None] = mapped_column(String(500))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class EventCandidate(Base):
    __tablename__ = "event_candidates"
    __table_args__ = (
        UniqueConstraint("id", "tenant_id", name="uq_events_id_tenant"),
        UniqueConstraint("id", "job_id", "tenant_id", name="uq_events_id_job_tenant"),
        ForeignKeyConstraint(
            ["job_id", "video_id", "tenant_id"],
            ["processing_jobs.id", "processing_jobs.video_id", "processing_jobs.tenant_id"],
            ondelete="CASCADE",
            name="fk_events_job_video_tenant",
        ),
        CheckConstraint("frame_index IS NULL OR frame_index >= 0", name="ck_events_frame_index"),
        CheckConstraint("video_time_seconds IS NULL OR video_time_seconds >= 0", name="ck_events_video_time"),
        CheckConstraint("confidence IS NULL OR confidence BETWEEN 0 AND 1", name="ck_events_confidence"),
        Index("ix_events_tenant_job_time", "tenant_id", "job_id", "video_time_seconds"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    tenant_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False
    )
    job_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    video_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    event_type: Mapped[str] = mapped_column(String(80), nullable=False)
    track_id: Mapped[int | None] = mapped_column(Integer)
    frame_index: Mapped[int | None] = mapped_column(Integer)
    video_time_seconds: Mapped[float | None]
    confidence: Mapped[float | None]
    bbox: Mapped[dict | None] = mapped_column(JSON)
    detector_notes: Mapped[str | None] = mapped_column(String(2000))
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class EvidenceAsset(Base):
    __tablename__ = "evidence_assets"
    __table_args__ = (
        ForeignKeyConstraint(
            ["job_id", "tenant_id"],
            ["processing_jobs.id", "processing_jobs.tenant_id"],
            ondelete="CASCADE",
            name="fk_assets_job_tenant",
        ),
        ForeignKeyConstraint(
            ["event_id", "job_id", "tenant_id"],
            ["event_candidates.id", "event_candidates.job_id", "event_candidates.tenant_id"],
            ondelete="CASCADE",
            name="fk_assets_event_job_tenant",
        ),
        UniqueConstraint("storage_key", name="uq_evidence_storage_key"),
        CheckConstraint("size_bytes IS NULL OR size_bytes >= 0", name="ck_assets_size"),
        Index("ix_assets_tenant_event", "tenant_id", "event_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    tenant_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False
    )
    job_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    event_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True))
    asset_type: Mapped[str] = mapped_column(String(40), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    content_type: Mapped[str] = mapped_column(String(127), nullable=False)
    size_bytes: Mapped[int | None] = mapped_column(Integer)
    checksum_sha256: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class EventReview(Base, TimestampMixin):
    __tablename__ = "event_reviews"
    __table_args__ = (
        UniqueConstraint("event_id", name="uq_reviews_event"),
        ForeignKeyConstraint(
            ["event_id", "tenant_id"],
            ["event_candidates.id", "event_candidates.tenant_id"],
            ondelete="CASCADE",
            name="fk_reviews_event_tenant",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "reviewer_user_id"],
            ["tenants.id", "tenants.owner_user_id"],
            ondelete="RESTRICT",
            name="fk_reviews_reviewer_tenant_owner",
        ),
        CheckConstraint(
            "status IN ('pending_review', 'under_review', 'reviewed_accepted', 'reviewed_rejected')",
            name="ck_reviews_status",
        ),
        CheckConstraint("revision >= 0", name="ck_reviews_revision"),
        Index("ix_reviews_tenant_status_updated", "tenant_id", "status", "updated_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    event_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    tenant_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False
    )
    reviewer_user_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="RESTRICT")
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, server_default="pending_review")
    notes: Mapped[str | None] = mapped_column(String(1000))
    revision: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")


class ProcessingLog(Base):
    __tablename__ = "processing_logs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["job_id", "tenant_id"],
            ["processing_jobs.id", "processing_jobs.tenant_id"],
            ondelete="CASCADE",
            name="fk_logs_job_tenant",
        ),
        Index("ix_logs_tenant_job_created", "tenant_id", "job_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    tenant_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False
    )
    job_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    code: Mapped[str] = mapped_column(String(80), nullable=False)
    message: Mapped[str] = mapped_column(String(1000), nullable=False)
    details: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
