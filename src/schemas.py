from datetime import datetime
from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_serializer, field_validator

from src.config import TrackGuardConfig


class ReviewStatus(str, Enum):
    """Review workflow state for an event candidate.
    
    All detections enter as candidates and remain unconfirmed until
    adjudicated by a human official.
    """
    PENDING_REVIEW = "pending_review"
    UNDER_REVIEW = "under_review"
    REVIEWED_ACCEPTED = "reviewed_accepted"
    REVIEWED_REJECTED = "reviewed_rejected"


class EventType(str, Enum):
    """Categorization of detected trackside candidate observations."""
    TRACK_LIMIT_CANDIDATE = "track_limit_candidate"
    INCIDENT_INDICATOR_CANDIDATE = "incident_indicator_candidate"
    STATIONARY_VEHICLE_CANDIDATE = "stationary_vehicle_candidate"
    TRAJECTORY_DEVIATION_CANDIDATE = "trajectory_deviation_candidate"
    UNCLASSIFIED_CANDIDATE = "unclassified_candidate"


class BoundingBox(BaseModel):
    """Axis-aligned bounding box coordinates [x1, y1, x2, y2]."""
    x1: float = Field(ge=0.0, description="Top-left X coordinate")
    y1: float = Field(ge=0.0, description="Top-left Y coordinate")
    x2: float = Field(ge=0.0, description="Bottom-right X coordinate")
    y2: float = Field(ge=0.0, description="Bottom-right Y coordinate")

    @field_validator("x2")
    @classmethod
    def validate_x2_greater_than_x1(cls, v: float, info) -> float:
        if "x1" in info.data and v < info.data["x1"]:
            raise ValueError(f"x2 ({v}) must be greater than or equal to x1 ({info.data['x1']})")
        return v

    @field_validator("y2")
    @classmethod
    def validate_y2_greater_than_y1(cls, v: float, info) -> float:
        if "y1" in info.data and v < info.data["y1"]:
            raise ValueError(f"y2 ({v}) must be greater than or equal to y1 ({info.data['y1']})")
        return v


class TrackEventRecord(BaseModel):
    """Neutral event record representing a candidate trackside incident or boundary event.
    
    This schema intentionally labels all events as candidates rather than confirmed
    violations to preserve evidentiary impartiality pending manual review.
    """
    event_id: str = Field(
        description="Unique identifier for the event candidate."
    )
    event_type: EventType = Field(
        description="Category of the candidate observation."
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="Optional wall-clock timestamp when known; omitted for video-relative events."
    )
    track_id: Optional[int] = Field(
        default=None,
        ge=0,
        description="Optional tracking ID associated with the vehicle involved."
    )
    frame_index: Optional[int] = Field(
        default=None,
        ge=0,
        description="Video frame index corresponding to the peak or trigger moment."
    )
    video_time_seconds: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Time offset in seconds from the start of the source video."
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Detector or heuristic confidence score associated with candidate flag."
    )
    bbox: Optional[BoundingBox] = Field(
        default=None,
        description="Bounding box of the vehicle at candidate trigger time."
    )
    evidence_reference: Optional[str] = Field(
        default=None,
        description="Reference path or identifier for associated video clip or keyframe image."
    )
    evidence_frame_reference: Optional[str] = Field(
        default=None,
        description="Local path to the annotated event frame, when evidence capture succeeds."
    )
    evidence_clip_reference: Optional[str] = Field(
        default=None,
        description="Local path to the event video clip, when evidence capture succeeds."
    )
    evidence_audio_status: Optional[Literal["preserved", "source_has_no_audio", "unavailable"]] = Field(
        default=None,
        description="Whether clip audio was preserved, absent at source, or unavailable to capture."
    )
    review_status: ReviewStatus = Field(
        default=ReviewStatus.PENDING_REVIEW,
        description="Current status in the review workflow. Defaults to 'pending_review'."
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional observational notes or review audit remarks."
    )


class EventReviewEntry(BaseModel):
    """Session review data stored separately from detector event records."""
    status: ReviewStatus
    notes: Optional[str] = Field(default=None, max_length=1000)


class EventReviewData(BaseModel):
    """Run-scoped human review decisions and optional reviewer notes."""
    run_id: str = Field(min_length=1)
    revision: int = Field(default=0, ge=0)
    reviews: dict[str, EventReviewEntry] = Field(default_factory=dict)


class ProcessingJobInput(BaseModel):
    """Serializable, worker-neutral request for one video processing run.

    Video identifiers are deliberately references rather than filesystem paths;
    an execution adapter resolves them to the worker's local scratch space.
    """
    job_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    config: TrackGuardConfig = Field(default_factory=TrackGuardConfig)
    max_frames: Optional[int] = Field(default=None, gt=0)
    enable_tracking: bool = True

    @field_serializer("config")
    def serialize_config(self, config: TrackGuardConfig) -> dict:
        # evidence_dir is an execution-local scratch path, not part of a job
        # message. Deserialization uses the config's default until the adapter
        # binds it to the job's isolated workspace.
        return config.model_dump(mode="json", exclude={"evidence_dir"})


class ProcessingProgress(BaseModel):
    """Progress with actual completed frames and a metadata-derived estimate."""
    processed_frames: int = Field(
        ge=0,
        description="Actual number of frames fully processed and written so far.",
    )
    total_frames: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Estimated total frame count from video metadata; it may differ from "
            "the number of frames that can actually be decoded."
        ),
    )


class ProcessingJobResult(BaseModel):
    """Serializable processing output without worker-local filesystem paths."""
    job_id: str = Field(min_length=1)
    output_reference: str = Field(min_length=1)
    total_frames: int = Field(ge=0)
    vehicle_detections_count: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0.0)
    processing_fps: float = Field(ge=0.0)
    video_width: int = Field(gt=0)
    video_height: int = Field(gt=0)
    source_fps: float = Field(gt=0.0)
    audio_status: Literal[
        "unknown", "preserved", "source_has_no_audio", "unavailable"
    ]
    event_candidates: list[TrackEventRecord] = Field(default_factory=list)
