"""Create tenant-scoped backend foundation tables."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0001_backend_foundation"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("email", sa.String(320), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "tenants",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("owner_user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("name", sa.String(160), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("owner_user_id", name="uq_tenants_owner_user"),
        sa.UniqueConstraint("id", "owner_user_id", name="uq_tenants_id_owner"),
    )
    op.create_table(
        "videos",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("tenant_id", sa.Uuid(), sa.ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("created_by_user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("original_filename", sa.String(255), nullable=False),
        sa.Column("declared_content_type", sa.String(127), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("checksum_sha256", sa.String(64)),
        sa.Column("storage_key", sa.String(1024)),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending_upload"),
        sa.Column("duration_seconds", sa.Float()),
        sa.Column("width", sa.Integer()),
        sa.Column("height", sa.Integer()),
        sa.Column("source_fps", sa.Float()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("id", "tenant_id", name="uq_videos_id_tenant"),
        sa.ForeignKeyConstraint(
            ["tenant_id", "created_by_user_id"], ["tenants.id", "tenants.owner_user_id"],
            ondelete="RESTRICT", name="fk_videos_creator_tenant_owner",
        ),
        sa.UniqueConstraint("storage_key", name="uq_videos_storage_key"),
        sa.CheckConstraint("size_bytes > 0 AND size_bytes <= 524288000", name="ck_videos_size"),
        sa.CheckConstraint("status IN ('pending_upload', 'ready', 'failed', 'deleted')", name="ck_videos_status"),
    )
    op.create_index("ix_videos_tenant_created", "videos", ["tenant_id", "created_at"])

    op.create_table(
        "processing_jobs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("tenant_id", sa.Uuid(), sa.ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("video_id", sa.Uuid(), nullable=False),
        sa.Column("requested_by_user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="created"),
        sa.Column("options", sa.JSON(), nullable=False),
        sa.Column("progress_percent", sa.Integer()),
        sa.Column("processed_frames", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("estimated_total_frames", sa.Integer()),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("idempotency_key", sa.String(128)),
        sa.Column("error_code", sa.String(80)),
        sa.Column("error_message", sa.String(500)),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(
            ["video_id", "tenant_id"], ["videos.id", "videos.tenant_id"],
            ondelete="RESTRICT", name="fk_jobs_video_tenant",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "requested_by_user_id"], ["tenants.id", "tenants.owner_user_id"],
            ondelete="RESTRICT", name="fk_jobs_requester_tenant_owner",
        ),
        sa.UniqueConstraint("id", "tenant_id", name="uq_jobs_id_tenant"),
        sa.UniqueConstraint("id", "video_id", "tenant_id", name="uq_jobs_id_video_tenant"),
        sa.UniqueConstraint("tenant_id", "idempotency_key", name="uq_jobs_tenant_idempotency"),
        sa.CheckConstraint(
            "status IN ('created', 'queued', 'running', 'finalizing', 'succeeded', 'failed', 'cancelled')",
            name="ck_jobs_status",
        ),
        sa.CheckConstraint("progress_percent IS NULL OR progress_percent BETWEEN 0 AND 100", name="ck_jobs_progress"),
        sa.CheckConstraint("attempt_count >= 0", name="ck_jobs_attempt_count"),
    )
    op.create_index("ix_jobs_tenant_created", "processing_jobs", ["tenant_id", "created_at"])
    op.create_index("ix_jobs_tenant_status_created", "processing_jobs", ["tenant_id", "status", "created_at"])

    op.create_table(
        "event_candidates",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("tenant_id", sa.Uuid(), sa.ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("video_id", sa.Uuid(), nullable=False),
        sa.Column("event_type", sa.String(80), nullable=False),
        sa.Column("track_id", sa.Integer()),
        sa.Column("frame_index", sa.Integer()),
        sa.Column("video_time_seconds", sa.Float()),
        sa.Column("confidence", sa.Float()),
        sa.Column("bbox", sa.JSON()),
        sa.Column("detector_notes", sa.String(2000)),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(
            ["job_id", "video_id", "tenant_id"],
            ["processing_jobs.id", "processing_jobs.video_id", "processing_jobs.tenant_id"],
            ondelete="CASCADE", name="fk_events_job_video_tenant",
        ),
        sa.UniqueConstraint("id", "tenant_id", name="uq_events_id_tenant"),
        sa.UniqueConstraint("id", "job_id", "tenant_id", name="uq_events_id_job_tenant"),
        sa.CheckConstraint("frame_index IS NULL OR frame_index >= 0", name="ck_events_frame_index"),
        sa.CheckConstraint("video_time_seconds IS NULL OR video_time_seconds >= 0", name="ck_events_video_time"),
        sa.CheckConstraint("confidence IS NULL OR confidence BETWEEN 0 AND 1", name="ck_events_confidence"),
    )
    op.create_index("ix_events_tenant_job_time", "event_candidates", ["tenant_id", "job_id", "video_time_seconds"])

    op.create_table(
        "evidence_assets",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("tenant_id", sa.Uuid(), sa.ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("event_id", sa.Uuid()),
        sa.Column("asset_type", sa.String(40), nullable=False),
        sa.Column("storage_key", sa.String(1024), nullable=False),
        sa.Column("content_type", sa.String(127), nullable=False),
        sa.Column("size_bytes", sa.Integer()),
        sa.Column("checksum_sha256", sa.String(64)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(
            ["job_id", "tenant_id"], ["processing_jobs.id", "processing_jobs.tenant_id"],
            ondelete="CASCADE", name="fk_assets_job_tenant",
        ),
        sa.ForeignKeyConstraint(
            ["event_id", "job_id", "tenant_id"],
            ["event_candidates.id", "event_candidates.job_id", "event_candidates.tenant_id"],
            ondelete="CASCADE", name="fk_assets_event_job_tenant",
        ),
        sa.UniqueConstraint("storage_key", name="uq_evidence_storage_key"),
        sa.CheckConstraint("size_bytes IS NULL OR size_bytes >= 0", name="ck_assets_size"),
    )
    op.create_index("ix_assets_tenant_event", "evidence_assets", ["tenant_id", "event_id"])

    op.create_table(
        "event_reviews",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("event_id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), sa.ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("reviewer_user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="RESTRICT")),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending_review"),
        sa.Column("notes", sa.String(1000)),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(
            ["event_id", "tenant_id"], ["event_candidates.id", "event_candidates.tenant_id"],
            ondelete="CASCADE", name="fk_reviews_event_tenant",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "reviewer_user_id"], ["tenants.id", "tenants.owner_user_id"],
            ondelete="RESTRICT", name="fk_reviews_reviewer_tenant_owner",
        ),
        sa.UniqueConstraint("event_id", name="uq_reviews_event"),
        sa.CheckConstraint(
            "status IN ('pending_review', 'under_review', 'reviewed_accepted', 'reviewed_rejected')",
            name="ck_reviews_status",
        ),
        sa.CheckConstraint("revision >= 0", name="ck_reviews_revision"),
    )
    op.create_index("ix_reviews_tenant_status_updated", "event_reviews", ["tenant_id", "status", "updated_at"])

    op.create_table(
        "processing_logs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("tenant_id", sa.Uuid(), sa.ForeignKey("tenants.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("severity", sa.String(16), nullable=False),
        sa.Column("code", sa.String(80), nullable=False),
        sa.Column("message", sa.String(1000), nullable=False),
        sa.Column("details", sa.JSON()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(
            ["job_id", "tenant_id"], ["processing_jobs.id", "processing_jobs.tenant_id"],
            ondelete="CASCADE", name="fk_logs_job_tenant",
        ),
    )
    op.create_index("ix_logs_tenant_job_created", "processing_logs", ["tenant_id", "job_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_logs_tenant_job_created", table_name="processing_logs")
    op.drop_table("processing_logs")
    op.drop_index("ix_reviews_tenant_status_updated", table_name="event_reviews")
    op.drop_table("event_reviews")
    op.drop_index("ix_assets_tenant_event", table_name="evidence_assets")
    op.drop_table("evidence_assets")
    op.drop_index("ix_events_tenant_job_time", table_name="event_candidates")
    op.drop_table("event_candidates")
    op.drop_index("ix_jobs_tenant_status_created", table_name="processing_jobs")
    op.drop_index("ix_jobs_tenant_created", table_name="processing_jobs")
    op.drop_table("processing_jobs")
    op.drop_index("ix_videos_tenant_created", table_name="videos")
    op.drop_table("videos")
    op.drop_table("tenants")
    op.drop_table("users")
