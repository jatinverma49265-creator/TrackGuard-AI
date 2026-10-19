import time
from pathlib import Path
from typing import Any, Dict, Optional, Set
import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from src.config import TrackBoundary, TrackGuardConfig
from src.events import BoundaryEventDetector
from src.schemas import TrackEventRecord

# COCO vehicle class identifiers
COCO_VEHICLE_CLASSES: Dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
VEHICLE_CLASS_IDS: Set[int] = set(COCO_VEHICLE_CLASSES.keys())


class VideoProcessingSummary:
    """Summary metrics of a video detection run."""
    def __init__(
        self,
        input_path: str,
        output_path: str,
        total_frames: int,
        vehicle_detections_count: int,
        elapsed_seconds: float,
        processing_fps: float,
        video_width: int,
        video_height: int,
        source_fps: float,
        event_candidates: Optional[list[TrackEventRecord]] = None,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.total_frames = total_frames
        self.vehicle_detections_count = vehicle_detections_count
        self.elapsed_seconds = elapsed_seconds
        self.processing_fps = processing_fps
        self.video_width = video_width
        self.video_height = video_height
        self.source_fps = source_fps
        self.event_candidates = event_candidates or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "total_frames": self.total_frames,
            "vehicle_detections_count": self.vehicle_detections_count,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "processing_fps": round(self.processing_fps, 2),
            "video_width": self.video_width,
            "video_height": self.video_height,
            "source_fps": round(self.source_fps, 2),
            "event_candidates": [event.model_dump(mode="json") for event in self.event_candidates],
        }


def resolve_device(requested_device: str) -> str:
    """Resolves execution device based on hardware availability."""
    if requested_device == "cpu":
        return "cpu"
    if requested_device == "cuda":
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    # 'auto' mode
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def annotate_frame(
    frame: np.ndarray,
    detections: list,
    frame_idx: int,
    current_fps: float,
    boundary_line: Optional[tuple[tuple[int, int], tuple[int, int]]] = None,
) -> np.ndarray:
    """Draws bounding boxes, labels, and telemetry overlay on a single frame."""
    annotated = frame.copy()

    if boundary_line is not None:
        cv2.line(annotated, boundary_line[0], boundary_line[1], (0, 0, 255), 2)

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det[:6]
        track_id = det[6] if len(det) > 6 else None
        cls_id = int(cls_id)
        label = COCO_VEHICLE_CLASSES.get(cls_id, "vehicle")
        track_label = f" ID:{track_id}" if track_id is not None else ""
        caption = f"{label}{track_label} {conf:.2f}"

        # Bounding box
        cv2.rectangle(
            annotated,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2,
        )

        # Label background and text
        (txt_w, txt_h), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        bg_y1 = max(0, int(y1) - txt_h - 4)
        cv2.rectangle(
            annotated,
            (int(x1), bg_y1),
            (int(x1) + txt_w, int(y1)),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            annotated,
            caption,
            (int(x1), int(y1) - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # Telemetry HUD overlay
    hud_text = f"Frame: {frame_idx} | Proc FPS: {current_fps:.1f} | Vehicles: {len(detections)}"
    cv2.putText(
        annotated,
        hud_text,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        hud_text,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 165, 255),
        1,
        cv2.LINE_AA,
    )

    return annotated


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    config: Optional[TrackGuardConfig] = None,
    model: Optional[Any] = None,
    max_frames: Optional[int] = None,
    enable_tracking: bool = True,
) -> VideoProcessingSummary:
    """Detects and annotates vehicles, optionally tracking them with ByteTrack.
    
    Args:
        input_path: Path to the input video file.
        output_path: Destination path for the annotated video.
        config: Optional TrackGuardConfig for inference parameters.
        model: Optional pre-instantiated YOLO model or compatible mock.
        max_frames: Optional limit on frames to process (useful for short tests/benchmarks).
        enable_tracking: Use ByteTrack to assign persistent IDs across video frames.

    Returns:
        VideoProcessingSummary containing execution metrics.
    """
    in_path = Path(input_path)
    out_path = Path(output_path)

    if not in_path.is_file():
        raise FileNotFoundError(f"Input video file not found: {in_path}")

    cfg = config or TrackGuardConfig()
    device = resolve_device(cfg.device)

    # Initialize model if not passed in
    if model is None:
        from ultralytics import YOLO
        default_weight = Path("weights/yolov8n.pt")
        if not default_weight.exists():
            raise FileNotFoundError(f"Model weight file not found at {default_weight}")
        model = YOLO(str(default_weight))

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {in_path}")

    writer: Optional[cv2.VideoWriter] = None

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if source_fps <= 0 or np.isnan(source_fps):
            source_fps = 25.0

        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid video dimensions: {width}x{height}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, source_fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to initialize video output writer at {out_path}")

        frames_processed = 0
        total_vehicle_detections = 0
        event_candidates: list[TrackEventRecord] = []
        boundary_detector = None
        boundary_line = None
        if cfg.track_boundary is not None:
            x_scale = max(width - 1, 1)
            y_scale = max(height - 1, 1)
            boundary_line = (
                (int(round(cfg.track_boundary.start_x * x_scale)), int(round(cfg.track_boundary.start_y * y_scale))),
                (int(round(cfg.track_boundary.end_x * x_scale)), int(round(cfg.track_boundary.end_y * y_scale))),
            )
            boundary_detector = BoundaryEventDetector(
                cfg.track_boundary,
                width,
                height,
                cfg.boundary_hysteresis_pixels,
            )
        start_time = time.perf_counter()

        while True:
            if max_frames is not None and frames_processed >= max_frames:
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                break

            inference = model.track if enable_tracking else model
            inference_kwargs = {
                "conf": cfg.confidence_threshold,
                "iou": cfg.iou_threshold,
                "classes": list(VEHICLE_CLASS_IDS),
                "device": device,
                "verbose": False,
            }
            if enable_tracking:
                # Ultralytics 8.4.182 creates fresh trackers with persist=False.
                # Persist from frame two onward so state spans this video only.
                inference_kwargs.update(
                    tracker="bytetrack.yaml",
                    persist=frames_processed > 0,
                )
            results = inference(
                frame,
                **inference_kwargs,
            )

            # Parse vehicle detections
            frame_detections = []
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    if cls_id in VEHICLE_CLASS_IDS:
                        xyxy = box.xyxy[0].tolist()
                        conf = float(box.conf[0].item())
                        track_id = None
                        if getattr(box, "id", None) is not None:
                            track_id = int(box.id[0].item())
                        frame_detections.append(
                            (xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, cls_id, track_id)
                        )
                        if boundary_detector is not None:
                            event = boundary_detector.update(
                                track_id=track_id,
                                bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                                frame_index=frames_processed + 1,
                                video_time_seconds=frames_processed / source_fps,
                                confidence=conf,
                            )
                            if event is not None:
                                event_candidates.append(event)

            total_vehicle_detections += len(frame_detections)
            frames_processed += 1

            elapsed = max(time.perf_counter() - start_time, 1e-5)
            curr_fps = frames_processed / elapsed

            annotated = annotate_frame(
                frame,
                frame_detections,
                frames_processed,
                curr_fps,
                boundary_line=boundary_line,
            )
            writer.write(annotated)

        total_elapsed = max(time.perf_counter() - start_time, 1e-5)
        overall_fps = frames_processed / total_elapsed

        return VideoProcessingSummary(
            input_path=str(in_path),
            output_path=str(out_path),
            total_frames=frames_processed,
            vehicle_detections_count=total_vehicle_detections,
            elapsed_seconds=total_elapsed,
            processing_fps=overall_fps,
            video_width=width,
            video_height=height,
            source_fps=source_fps,
            event_candidates=event_candidates,
        )

    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TrackGuard AI - Vehicle Detection and Boundary Candidates")
    parser.add_argument("--input", "-i", required=True, help="Path to input video file")
    parser.add_argument("--output", "-o", default="runs/detected_output.mp4", help="Path to output annotated video file")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (default: 0.35)")
    parser.add_argument("--device", default="auto", help="Inference device: 'cpu', 'cuda', or 'auto'")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument(
        "--boundary",
        type=float,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Optional boundary endpoints in normalized 0..1 frame coordinates",
    )
    parser.add_argument(
        "--boundary-hysteresis-pixels",
        type=float,
        default=5.0,
        help="Stable-side margin for boundary crossing candidates (default: 5 pixels)",
    )
    parser.add_argument(
        "--no-track",
        dest="enable_tracking",
        action="store_false",
        help="Run detection only without ByteTrack IDs",
    )

    args = parser.parse_args()
    boundary = None
    if args.boundary is not None:
        boundary = TrackBoundary(
            start_x=args.boundary[0],
            start_y=args.boundary[1],
            end_x=args.boundary[2],
            end_y=args.boundary[3],
        )
    cfg = TrackGuardConfig(
        confidence_threshold=args.conf,
        device=args.device,
        track_boundary=boundary,
        boundary_hysteresis_pixels=args.boundary_hysteresis_pixels,
    )
    summary = process_video(
        input_path=args.input,
        output_path=args.output,
        config=cfg,
        max_frames=args.max_frames,
        enable_tracking=args.enable_tracking,
    )
    print("\n[TrackGuard AI] Vehicle Detection Complete:")
    for k, v in summary.to_dict().items():
        print(f"  {k}: {v}")
