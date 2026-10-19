from typing import Literal
from pydantic import BaseModel, Field, field_validator


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
        description="Duration of sliding video buffer retained around detected candidate events."
    )
    track_limits_margin_pixels: float = Field(
        default=0.0,
        ge=0.0,
        description="Buffer margin in pixels applied to track-limit boundary polygons."
    )

    @field_validator("confidence_threshold", "iou_threshold")
    @classmethod
    def validate_thresholds(cls, value: float) -> float:
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {value}")
        return value
