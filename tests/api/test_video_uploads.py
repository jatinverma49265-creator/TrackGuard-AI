from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4
from urllib.parse import unquote, urlsplit

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from src.api.dependencies import TenantContext, get_db, get_storage_client, get_tenant_context
from src.api.main import create_app
from src.api.settings import Settings
from src.api.storage import SupabaseStorageClient, VIDEO_BUCKET
from src.db.base import Base
from src.db.models import Tenant, User, Video


class MockStorageAPI:
    def __init__(self):
        self.sign_status = 200
        self.info_status = 200
        self.object_size = 4096
        self.object_content_type = "video/mp4"
        self.requests = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if "/object/upload/sign/" in request.url.path:
            if self.sign_status >= 400:
                return httpx.Response(
                    self.sign_status,
                    json={"message": "mock storage detail must not reach the client"},
                )
            suffix = request.url.path.split(f"/object/upload/sign/{VIDEO_BUCKET}/", 1)[1]
            return httpx.Response(
                200,
                json={"url": f"/object/upload/sign/{VIDEO_BUCKET}/{suffix}?token=mock-upload-token"},
            )
        if "/object/info/" in request.url.path:
            if self.info_status >= 400:
                return httpx.Response(self.info_status, json={"message": "mock info failure"})
            return httpx.Response(
                200,
                json={
                    "metadata": {
                        "size": self.object_size,
                        "mimetype": self.object_content_type,
                    }
                },
            )
        return httpx.Response(404, json={"message": "unexpected storage request"})


@pytest.fixture
def upload_api():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    user_id, tenant_id = uuid4(), uuid4()
    other_user_id, other_tenant_id = uuid4(), uuid4()
    with Session(engine) as session:
        session.add_all(
            [
                User(id=user_id, email="owner@example.test"),
                Tenant(id=tenant_id, owner_user_id=user_id, name="Owner workspace"),
                User(id=other_user_id, email="other@example.test"),
                Tenant(id=other_tenant_id, owner_user_id=other_user_id, name="Other workspace"),
            ]
        )
        session.commit()

    storage_api = MockStorageAPI()
    http_client = httpx.Client(transport=httpx.MockTransport(storage_api.handle))
    storage_client = SupabaseStorageClient(
        Settings(
            supabase_url="https://storage-test.supabase.co",
            supabase_storage_secret_key="server-only-test-secret",
        ),
        http_client=http_client,
    )
    app = create_app()

    def override_db():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_tenant_context] = lambda: TenantContext(
        user_id=user_id, tenant_id=tenant_id, email="owner@example.test"
    )
    app.dependency_overrides[get_storage_client] = lambda: storage_client

    with TestClient(app) as client:
        yield {
            "client": client,
            "engine": engine,
            "app": app,
            "storage_api": storage_api,
            "http_client": http_client,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "other_tenant_id": other_tenant_id,
            "other_user_id": other_user_id,
        }

    http_client.close()
    Base.metadata.drop_all(engine)
    engine.dispose()


def create_upload(client: TestClient, *, content_type="video/mp4", size=4096):
    return client.post(
        "/videos",
        json={
            "original_filename": "race.mp4",
            "declared_content_type": content_type,
            "size_bytes": size,
        },
    )


def read_video(engine, video_id):
    with Session(engine) as session:
        return session.scalar(select(Video).where(Video.id == UUID(str(video_id))))


def test_rejects_invalid_mime_and_files_over_50_mib(upload_api):
    client = upload_api["client"]
    invalid_mime = create_upload(client, content_type="application/pdf")
    oversized = create_upload(client, size=50 * 1024 * 1024 + 1)

    assert invalid_mime.status_code == 422
    assert oversized.status_code == 422
    assert upload_api["storage_api"].requests == []


def test_missing_bearer_is_unauthorized(upload_api):
    app = upload_api["app"]
    app.dependency_overrides.pop(get_tenant_context)
    response = upload_api["client"].post(
        "/videos",
        json={
            "original_filename": "race.mp4",
            "declared_content_type": "video/mp4",
            "size_bytes": 4096,
        },
    )
    assert response.status_code == 401


def test_initiation_returns_scoped_signed_url_without_storage_secret(upload_api):
    response = create_upload(upload_api["client"])
    assert response.status_code == 201
    data = response.json()
    assert data["upload_method"] == "PUT"
    assert data["required_headers"] == {"Content-Type": "video/mp4"}
    assert "mock-upload-token" in data["upload_url"]
    assert "server-only-test-secret" not in response.text

    signed_path = unquote(urlsplit(data["upload_url"]).path)
    prefix = f"/storage/v1/object/upload/sign/{VIDEO_BUCKET}/tenants/{upload_api['tenant_id']}/videos/{data['video_id']}/"
    assert signed_path.startswith(prefix)
    assert len(signed_path.removeprefix(prefix)) == 32

    video = read_video(upload_api["engine"], data["video_id"])
    assert video is not None
    assert video.status == "pending_upload"
    assert video.storage_key == signed_path.split(f"/{VIDEO_BUCKET}/", 1)[1]
    assert video.upload_expires_at is not None

    sign_request = upload_api["storage_api"].requests[0]
    assert sign_request.headers["apikey"] == "server-only-test-secret"
    assert "authorization" not in sign_request.headers


def test_failed_storage_signing_marks_video_failed_and_hides_provider_error(upload_api):
    upload_api["storage_api"].sign_status = 503
    response = create_upload(upload_api["client"])

    assert response.status_code == 502
    assert response.json() == {"detail": "storage_upload_initialization_failed"}
    assert "mock storage detail" not in response.text
    assert "server-only-test-secret" not in response.text
    with Session(upload_api["engine"]) as session:
        video = session.scalar(select(Video))
        assert video.status == "failed"


def test_completion_requires_object_and_ignores_client_success_claim(upload_api):
    started = create_upload(upload_api["client"])
    video_id = started.json()["video_id"]
    upload_api["storage_api"].info_status = 404

    response = upload_api["client"].post(
        f"/videos/{video_id}/complete",
        json={"success": True},
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "uploaded_object_not_found"}
    assert read_video(upload_api["engine"], video_id).status == "pending_upload"


def test_storage_verification_failure_does_not_mark_video_ready(upload_api):
    started = create_upload(upload_api["client"])
    video_id = started.json()["video_id"]
    upload_api["storage_api"].info_status = 503

    response = upload_api["client"].post(f"/videos/{video_id}/complete")

    assert response.status_code == 502
    assert response.json() == {"detail": "storage_verification_failed"}
    assert "mock info failure" not in response.text
    assert read_video(upload_api["engine"], video_id).status == "pending_upload"


def test_completion_checks_actual_size_and_content_type(upload_api):
    started = create_upload(upload_api["client"])
    video_id = started.json()["video_id"]
    upload_api["storage_api"].object_size = 4095
    upload_api["storage_api"].object_content_type = "video/quicktime"

    response = upload_api["client"].post(f"/videos/{video_id}/complete")

    assert response.status_code == 422
    assert response.json() == {"detail": "uploaded_object_metadata_mismatch"}
    assert read_video(upload_api["engine"], video_id).status == "failed"


def test_successful_completion_marks_ready_and_repeated_completion_is_rejected(upload_api):
    started = create_upload(upload_api["client"])
    video_id = started.json()["video_id"]

    completed = upload_api["client"].post(f"/videos/{video_id}/complete")
    repeated = upload_api["client"].post(f"/videos/{video_id}/complete")

    assert completed.status_code == 200
    assert completed.json()["status"] == "ready"
    assert completed.json()["size_bytes"] == upload_api["storage_api"].object_size
    assert repeated.status_code == 409
    assert repeated.json() == {"detail": "video_not_pending_upload"}
    assert len([r for r in upload_api["storage_api"].requests if "/object/info/" in r.url.path]) == 1


def test_expired_upload_cannot_be_completed(upload_api):
    started = create_upload(upload_api["client"])
    video_id = started.json()["video_id"]
    with Session(upload_api["engine"]) as session:
        video = session.get(Video, UUID(str(video_id)))
        video.upload_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()

    response = upload_api["client"].post(f"/videos/{video_id}/complete")

    assert response.status_code == 410
    assert response.json() == {"detail": "upload_url_expired"}
    assert read_video(upload_api["engine"], video_id).status == "failed"
    assert len([r for r in upload_api["storage_api"].requests if "/object/info/" in r.url.path]) == 0


def test_other_tenant_cannot_complete_video(upload_api):
    video_id = uuid4()
    other_tenant_id = upload_api["other_tenant_id"]
    other_user_id = upload_api["other_user_id"]
    key = f"tenants/{other_tenant_id}/videos/{video_id}/{uuid4().hex}"
    with Session(upload_api["engine"]) as session:
        session.add(
            Video(
                id=video_id,
                tenant_id=other_tenant_id,
                created_by_user_id=other_user_id,
                original_filename="private.mp4",
                declared_content_type="video/mp4",
                size_bytes=4096,
                storage_key=key,
                status="pending_upload",
                upload_expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            )
        )
        session.commit()
    storage_requests_before = len(upload_api["storage_api"].requests)

    response = upload_api["client"].post(f"/videos/{video_id}/complete")

    assert response.status_code == 404
    assert response.json() == {"detail": "video_not_found"}
    assert len(upload_api["storage_api"].requests) == storage_requests_before
