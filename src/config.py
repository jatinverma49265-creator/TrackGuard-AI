from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class TrackBoundary(BaseModel):
    """Finite track-boundary segment with endpoints normalized to [0, 1]."""
    start_x: float = Field(ge=0.0, le=1.0)
    start_y: float = Field(ge=0.0, le=1.0)
    end_x: float = Field(ge=0.0, le=1.0)
    end_y: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_distinct_endpoints(self):
        if self.start_x == self.end_x and self.start_y == self.end_y:
            raise ValueError("Track boundary endpoints must be different")
        return self


class TrackGuardConfig(BaseModel):
    """Configuration settings for TrackGuard AI processing pipeline.
    
    Optimized for resource-constrained hardware such as Windows laptops
    with NVIDIA RTX 2050 (4 GB VRAM).
    """
    device: Literal["cuda", "cpu", "auto"] = Field(
        default="auto",
        description="Execution device: 'cuda', 'cpu', or 'auto' for detection."
    )
    enable_fp16: bool = Field(
        default=True,
        description="Enable FP16 half-precision inference to reduce VRAM pressure on 4GB GPUs."
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Inference batch size. Kept minimal for fixed-camera real-time processing."
    )
    input_width: int = Field(
        default=640,
        ge=320,
        le=1920,
        description="Model input width in pixels."
    )
    input_height: int = Field(
        default=640,
        ge=320,
        le=1920,
        description="Model input height in pixels."
    )
    confidence_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum detection confidence score to retain a candidate box."
    )
    iou_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Intersection-over-Union threshold for NMS and association."
    )
    evidence_dir: str = Field(
        default="evidence",
        description="Directory path where incident evidence clips and keyframes are archived."
    )
    max_event_buffer_seconds: float = Field(
        default=10.0,
        gt=0.0,
        le=60.0,
        description="Maximum total evidence clip duration around a detected candidate event."
    )
    evidence_pre_seconds: float = Field(
        default=3.0,
        ge=0.0,
        description="Seconds of video to retain before a candidate event."
    )
    evidence_post_seconds: float = Field(
        default=3.0,
        ge=0.0,
        description="Seconds of video to retain after a candidate event."
    )
    track_limits_margin_pixels: float = Field(
        default=0.0,
        ge=0.0,
        description="Buffer margin in pixels applied to track-limit boundary polygons."
    )
    track_boundary: Optional[TrackBoundary] = Field(
        default=None,
        description="Optional finite boundary segment with normalized frame coordinates."
    )
    boundary_hysteresis_pixels: float = Field(
        default=5.0,
        ge=0.0,
        description="Stable-side distance from a configured boundary required to confirm a crossing."
    )

    @field_validator("confidence_threshold", "iou_threshold")
    @classmethod
    def validate_thresholds(cls, value: float) -> float:
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {value}")
        return value

    @model_validator(mode="after")
    def validate_evidence_window(self):
        if self.evidence_pre_seconds + self.evidence_post_seconds > self.max_event_buffer_seconds:
            raise ValueError("Evidence pre/post duration must not exceed max_event_buffer_seconds")
        return self
