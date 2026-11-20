import time
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set
import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from src.config import TrackBoundary, TrackGuardConfig
from src.evidence import EvidenceCapture
from src.events import BoundaryEventDetector
from src.schemas import ProcessingProgress, TrackEventRecord

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
        audio_status: str = "unknown",
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
        self.audio_status = audio_status

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
            "audio_status": self.audio_status,
            "event_candidates": [event.model_dump(mode="json") for event in self.event_candidates],
        }


class ProcessingCancelledError(RuntimeError):
    """Raised when a processing run is cancelled at a frame boundary."""


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
    unique_tracks: int = 0,
    candidate_count: int = 0,
    mean_confidence: float = 0.0,
    tracking_enabled: bool = True,
    boundary_configured: bool = False,
) -> np.ndarray:
    """Draw detections and a compact trackside review summary."""
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

    # Keep the live processing readout small and legible.
    hud_text = f"FRAME {frame_idx:05d}   {current_fps:.1f} PROC FPS"
    hud_width = min(310, max(190, annotated.shape[1] - 32))
    hud_layer = annotated.copy()
    cv2.rectangle(hud_layer, (16, 16), (16 + hud_width, 49), (18, 25, 35), -1)
    cv2.addWeighted(hud_layer, 0.78, annotated, 0.22, 0, annotated)
    cv2.putText(annotated, hud_text, (27, 39), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (115, 220, 255), 1, cv2.LINE_AA)

    # Compact, translucent analysis panel in the upper-right corner.
    panel_width = min(360, max(180, annotated.shape[1] - 32))
    panel_height = min(224, max(126, annotated.shape[0] - 32))
    right = annotated.shape[1] - 16
    left = max(16, right - panel_width)
    top = 16
    bottom = min(annotated.shape[0] - 16, top + panel_height)
    panel = annotated.copy()
    cv2.rectangle(panel, (left, top), (right, bottom), (18, 25, 35), -1)
    cv2.addWeighted(panel, 0.82, annotated, 0.18, 0, annotated)
    cv2.rectangle(annotated, (left, top), (right, bottom), (74, 92, 112), 1)
    cv2.rectangle(annotated, (left, top), (left + 3, bottom), (0, 190, 255), -1)

    scale = max(0.44, min(0.62, annotated.shape[1] / 1500.0))
    line_height = max(21, int(30 * scale / 0.5))
    x = left + 15
    y = top + 25
    title = "TRACKGUARD  /  LIVE REVIEW"
    cv2.putText(annotated, title, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.88,
                (225, 235, 245), 1, cv2.LINE_AA)
    y += line_height + 2
    cv2.line(annotated, (x, y - 7), (right - 14, y - 7), (74, 92, 112), 1, cv2.LINE_AA)

    rows = [
        ("VEHICLES NOW", str(len(detections)), (90, 220, 160)),
        ("TRACKS SEEN", str(unique_tracks) if tracking_enabled else "tracking off", (120, 195, 255)),
        ("MEAN CONFIDENCE", f"{mean_confidence:.0%}" if detections else "—", (210, 215, 225)),
        ("BOUNDARY CANDIDATES", str(candidate_count), (80, 170, 255)),
    ]
    for label, value, value_color in rows:
        cv2.putText(annotated, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale * 0.64, (155, 170, 185), 1, cv2.LINE_AA)
        value_size = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, scale * 0.78, 1)[0]
        cv2.putText(annotated, value, (right - 14 - value_size[0], y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale * 0.78, value_color, 1, cv2.LINE_AA)
        y += line_height

    footer = "REVIEW REQUIRED" if candidate_count else (
        "NO CANDIDATES" if boundary_configured else "BOUNDARY NOT CONFIGURED"
    )
    cv2.putText(annotated, footer, (x, min(bottom - 9, y + 2)), cv2.FONT_HERSHEY_SIMPLEX,
                scale * 0.56, (80, 190, 255) if candidate_count else (110, 190, 155),
                1, cv2.LINE_AA)

    return annotated


def _get_ffmpeg_path() -> Optional[str]:
    """Find system FFmpeg or the executable bundled by imageio-ffmpeg."""
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError):
        return None


def _mux_source_audio(ffmpeg: str, video_path: Path, source_path: Path,
                      output_path: Path, duration_seconds: float) -> str:
    """Mux original audio into the annotated video and verify the audio stream."""
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_path),
         "-i", str(source_path), "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy",
         "-c:a", "aac", "-t", f"{duration_seconds:.6f}", "-movflags", "+faststart",
         str(output_path)],
        capture_output=True, text=True, timeout=180, check=False,
    )
    if result.returncode != 0 or not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg could not mux source audio: {(result.stderr or '').strip()[-1200:]}")
    probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(output_path),
         "-map", "0:a:0", "-frames:a", "1", "-f", "null", "-"],
        capture_output=True, text=True, timeout=30, check=False,
    )
    return "preserved" if probe.returncode == 0 else "source_has_no_audio"


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    config: Optional[TrackGuardConfig] = None,
    model: Optional[Any] = None,
    max_frames: Optional[int] = None,
    enable_tracking: bool = True,
    progress_callback: Optional[Callable[[ProcessingProgress], None]] = None,
    cancellation_requested: Optional[Callable[[], bool]] = None,
) -> VideoProcessingSummary:
    """Detects and annotates vehicles, optionally tracking them with ByteTrack.
    
    Args:
        input_path: Path to the input video file.
        output_path: Destination path for the annotated video.
        config: Optional TrackGuardConfig for inference parameters.
        model: Optional pre-instantiated YOLO model or compatible mock.
        max_frames: Optional limit on frames to process (useful for short tests/benchmarks).
        enable_tracking: Use ByteTrack to assign persistent IDs across video frames.
        progress_callback: Optional callback invoked after each completed frame.
        cancellation_requested: Optional predicate checked at frame boundaries.

    Returns:
        VideoProcessingSummary containing execution metrics.
    """
    in_path = Path(input_path)
    out_path = Path(output_path)

    if not in_path.is_file():
        raise FileNotFoundError(f"Input video file not found: {in_path}")
    if in_path.resolve() == out_path.resolve():
        raise ValueError("Input and output video paths must be different")

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
    writer: Optional[cv2.VideoWriter] = None
    video_only_path = out_path.with_name(f".{out_path.stem}.video-only{out_path.suffix}")
    muxed_path = out_path.with_name(f".{out_path.stem}.with-audio{out_path.suffix}")
    evidence_capture: Optional[EvidenceCapture] = None
    evidence_capture_finished = False
    audio_status = "unavailable"

    try:
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {in_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if source_fps <= 0 or np.isnan(source_fps):
            source_fps = 25.0

        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid video dimensions: {width}x{height}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_only_path), fourcc, source_fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to initialize video output writer at {video_only_path}")

        frames_processed = 0
        reported_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if reported_total_frames <= 0:
            reported_total_frames = None
        elif max_frames is not None:
            reported_total_frames = min(reported_total_frames, max_frames)
        total_vehicle_detections = 0
        unique_track_ids: set[int] = set()
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
            if enable_tracking:
                evidence_capture = EvidenceCapture(
                    evidence_dir=cfg.evidence_dir,
                    source_path=in_path,
                    fps=source_fps,
                    frame_size=(width, height),
                    pre_seconds=cfg.evidence_pre_seconds,
                    post_seconds=cfg.evidence_post_seconds,
                )
        start_time = time.perf_counter()

        while True:
            if cancellation_requested is not None and cancellation_requested():
                raise ProcessingCancelledError("Video processing was cancelled.")
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
            frame_events: list[TrackEventRecord] = []
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
                        if track_id is not None:
                            unique_track_ids.add(track_id)
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
                                frame_events.append(event)

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
                unique_tracks=len(unique_track_ids),
                candidate_count=len(event_candidates),
                mean_confidence=(sum(det[4] for det in frame_detections) / len(frame_detections))
                if frame_detections else 0.0,
                tracking_enabled=enable_tracking,
                boundary_configured=cfg.track_boundary is not None,
            )
            writer.write(annotated)
            if evidence_capture is not None:
                evidence_capture.push_frame(frames_processed, annotated, frame_events)
            if progress_callback is not None:
                progress_callback(ProcessingProgress(
                    processed_frames=frames_processed,
                    total_frames=reported_total_frames,
                ))
            if cancellation_requested is not None and cancellation_requested():
                raise ProcessingCancelledError("Video processing was cancelled.")

        if evidence_capture is not None:
            evidence_capture.finish()
            evidence_capture_finished = True
        # OpenCV does not write audio. Close its video stream, then mux the
        # original soundtrack and limit it to the frames actually processed.
        writer.release()
        writer = None
        total_elapsed = max(time.perf_counter() - start_time, 1e-5)
        overall_fps = frames_processed / total_elapsed
        ffmpeg = _get_ffmpeg_path()
        if ffmpeg is None:
            raise RuntimeError(
                "Audio muxing requires FFmpeg. Install project dependencies with "
                "'.venv\\Scripts\\python.exe -m pip install -r requirements.txt'."
            )
        audio_status = _mux_source_audio(
            ffmpeg, video_only_path, in_path, muxed_path, frames_processed / source_fps
        )
        muxed_path.replace(out_path)
        video_only_path.unlink(missing_ok=True)

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
            audio_status=audio_status,
        )

    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        video_only_path.unlink(missing_ok=True)
        muxed_path.unlink(missing_ok=True)
        if evidence_capture is not None and not evidence_capture_finished:
            evidence_capture.abort()


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
    parser.add_argument("--evidence-dir", default="evidence", help="Directory for local event frames and clips")
    parser.add_argument(
        "--evidence-before-seconds",
        type=float,
        default=3.0,
        help="Seconds of event clip before the crossing candidate (default: 3)",
    )
    parser.add_argument(
        "--evidence-after-seconds",
        type=float,
        default=3.0,
        help="Seconds of event clip after the crossing candidate (default: 3)",
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
        evidence_dir=args.evidence_dir,
        evidence_pre_seconds=args.evidence_before_seconds,
        evidence_post_seconds=args.evidence_after_seconds,
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
