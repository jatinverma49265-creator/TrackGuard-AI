from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


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
    timestamp: datetime = Field(
        description="Timestamp when the candidate event occurred or was logged."
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
    review_status: ReviewStatus = Field(
        default=ReviewStatus.PENDING_REVIEW,
        description="Current status in the review workflow. Defaults to 'pending_review'."
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional observational notes or review audit remarks."
    )
