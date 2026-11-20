import re
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from src.api.dependencies import TenantContext, get_storage_client, get_tenant_context
from src.api.schemas import (
    MAX_VIDEO_UPLOAD_BYTES,
    VideoCreateRequest,
    VideoListResponse,
    VideoResponse,
    VideoUploadResponse,
)
from src.db.models import Video
from src.db.session import get_db
from src.api.storage import (
    SIGNED_UPLOAD_URL_LIFETIME_SECONDS,
    StorageMetadataInvalid,
    StorageUnavailable,
    SupabaseStorageClient,
)


router = APIRouter(prefix="/videos", tags=["videos"])


def _video_response(video: Video) -> VideoResponse:
    return VideoResponse(
        id=video.id,
        original_filename=video.original_filename,
        declared_content_type=video.declared_content_type,
        size_bytes=video.size_bytes,
        status=video.status,
        created_at=video.created_at,
    )


@router.post("", response_model=VideoUploadResponse, status_code=status.HTTP_201_CREATED)
def create_video(
    payload: VideoCreateRequest,
    context: TenantContext = Depends(get_tenant_context),
    session: Session = Depends(get_db),
    storage: SupabaseStorageClient = Depends(get_storage_client),
) -> VideoUploadResponse:
    if payload.size_bytes > MAX_VIDEO_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="video_too_large")

    video_id = uuid4()
    object_key = f"tenants/{context.tenant_id}/videos/{video_id}/{uuid4().hex}"
    video = Video(
        id=video_id,
        tenant_id=context.tenant_id,
        created_by_user_id=context.user_id,
        original_filename=payload.original_filename,
        declared_content_type=payload.declared_content_type,
        size_bytes=payload.size_bytes,
        storage_key=object_key,
        status="pending_upload",
    )
    session.add(video)
    session.commit()
    session.refresh(video)

    try:
        upload_url = storage.create_signed_upload_url(object_key)
    except StorageUnavailable as exc:
        session.execute(
            update(Video)
            .where(
                Video.id == video_id,
                Video.tenant_id == context.tenant_id,
                Video.status == "pending_upload",
            )
            .values(status="failed")
        )
        session.commit()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="storage_upload_initialization_failed",
        ) from exc

    expires_at = datetime.now(timezone.utc) + timedelta(seconds=SIGNED_UPLOAD_URL_LIFETIME_SECONDS)
    video.upload_expires_at = expires_at
    session.commit()
    return VideoUploadResponse(
        video_id=video.id,
        upload_url=upload_url,
        required_headers={"Content-Type": payload.declared_content_type},
        expires_at=expires_at,
    )


def _valid_server_object_key(video: Video, tenant_id: UUID) -> bool:
    if not video.storage_key:
        return False
    parts = video.storage_key.split("/")
    return (
        len(parts) == 5
        and parts[:4] == ["tenants", str(tenant_id), "videos", str(video.id)]
        and re.fullmatch(r"[0-9a-f]{32}", parts[4]) is not None
    )


def _mark_video_failed(session: Session, video_id: UUID, tenant_id: UUID) -> None:
    session.execute(
        update(Video)
        .where(
            Video.id == video_id,
            Video.tenant_id == tenant_id,
            Video.status == "pending_upload",
        )
        .values(status="failed")
    )
    session.commit()


@router.post("/{video_id}/complete", response_model=VideoResponse)
def complete_video_upload(
    video_id: UUID,
    context: TenantContext = Depends(get_tenant_context),
    session: Session = Depends(get_db),
    storage: SupabaseStorageClient = Depends(get_storage_client),
) -> VideoResponse:
    video = session.scalar(
        select(Video).where(
            Video.id == video_id,
            Video.tenant_id == context.tenant_id,
        )
    )
    if video is None:
        raise HTTPException(status_code=404, detail="video_not_found")
    if video.status != "pending_upload" or not _valid_server_object_key(video, context.tenant_id):
        raise HTTPException(status_code=409, detail="video_not_pending_upload")

    expires_at = video.upload_expires_at
    if expires_at is None:
        _mark_video_failed(session, video_id, context.tenant_id)
        raise HTTPException(status_code=410, detail="upload_url_expired")
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at <= datetime.now(timezone.utc):
        _mark_video_failed(session, video_id, context.tenant_id)
        raise HTTPException(status_code=410, detail="upload_url_expired")

    try:
        object_info = storage.get_object_info(video.storage_key)
    except StorageMetadataInvalid as exc:
        _mark_video_failed(session, video_id, context.tenant_id)
        raise HTTPException(status_code=422, detail="uploaded_object_metadata_invalid") from exc
    except StorageUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="storage_verification_failed",
        ) from exc

    if object_info is None:
        raise HTTPException(status_code=409, detail="uploaded_object_not_found")
    if (
        object_info.size_bytes <= 0
        or object_info.size_bytes > MAX_VIDEO_UPLOAD_BYTES
        or object_info.size_bytes != video.size_bytes
        or object_info.content_type != video.declared_content_type.lower()
    ):
        _mark_video_failed(session, video_id, context.tenant_id)
        raise HTTPException(status_code=422, detail="uploaded_object_metadata_mismatch")

    result = session.execute(
        update(Video)
        .where(
            Video.id == video_id,
            Video.tenant_id == context.tenant_id,
            Video.status == "pending_upload",
            Video.upload_expires_at == video.upload_expires_at,
        )
        .values(status="ready", size_bytes=object_info.size_bytes)
    )
    if result.rowcount != 1:
        session.rollback()
        raise HTTPException(status_code=409, detail="video_state_changed")
    session.commit()
    session.refresh(video)
    return _video_response(video)


@router.get("", response_model=VideoListResponse)
def list_videos(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    context: TenantContext = Depends(get_tenant_context),
    session: Session = Depends(get_db),
) -> VideoListResponse:
    rows = session.scalars(
        select(Video)
        .where(Video.tenant_id == context.tenant_id)
        .order_by(Video.created_at.desc(), Video.id.desc())
        .offset(offset)
        .limit(limit)
    ).all()
    return VideoListResponse(items=[_video_response(row) for row in rows], limit=limit, offset=offset)
