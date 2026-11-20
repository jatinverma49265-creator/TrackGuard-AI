from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.api.schemas import JobCreateRequest, VideoCreateRequest
from src.db.models import EventCandidate, EventReview


def test_video_request_validates_size_and_rejects_paths_and_extra_identity():
    request = VideoCreateRequest(
        original_filename="race.mp4",
        declared_content_type="video/mp4",
        size_bytes=1024,
    )
    assert request.size_bytes == 1024
    for filename in ("../race.mp4", "folder\\race.mp4", " "):
        with pytest.raises(ValidationError):
            VideoCreateRequest(
                original_filename=filename,
                declared_content_type="video/mp4",
                size_bytes=1024,
            )
    with pytest.raises(ValidationError):
        VideoCreateRequest(
            original_filename="race.mp4",
            declared_content_type="video/mp4",
            size_bytes=501 * 1024 * 1024,
            user_id=str(uuid4()),
        )


def test_job_contract_does_not_accept_client_identity_or_local_paths():
    request = JobCreateRequest.model_validate({"video_id": str(uuid4())})
    assert request.options.enable_tracking is True
    with pytest.raises(ValidationError):
        JobCreateRequest.model_validate(
            {"video_id": str(uuid4()), "tenant_id": str(uuid4()), "local_video_path": "C:\\video.mp4"}
        )


def test_review_state_is_separate_from_detector_candidate_model():
    assert "status" not in EventCandidate.__table__.columns
    assert "review_status" not in EventCandidate.__table__.columns
    assert "status" in EventReview.__table__.columns
    assert any(constraint.name == "fk_reviews_event_tenant" for constraint in EventReview.__table__.constraints)

