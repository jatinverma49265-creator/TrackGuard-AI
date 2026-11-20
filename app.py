"""Local Streamlit foundation for the TrackGuard AI dashboard."""

import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import streamlit as st
from pydantic import ValidationError
from streamlit_image_coordinates import streamlit_image_coordinates
from src.config import TrackBoundary, TrackGuardConfig
from src.schemas import EventReviewData, EventReviewEntry, ReviewStatus, TrackEventRecord


MAX_UPLOAD_MB = 500
SUPPORTED_VIDEO_TYPES = ("mp4", "mov", "avi", "mkv")
VIDEO_MIME_TYPES = {
    "mp4": "video/mp4",
    "mov": "video/quicktime",
    "avi": "video/x-msvideo",
    "mkv": "video/x-matroska",
}
UPLOAD_CHUNK_BYTES = 1024 * 1024
MAX_PREVIEW_FRAME_BYTES = 8 * 1024 * 1024
MAX_PREVIEW_WIDTH = 1280
MAX_PREVIEW_HEIGHT = 720
FRAME_EXTRACTION_TIMEOUT_SECONDS = 120
MAX_REVIEW_NOTE_CHARS = 1000
EVENT_RECORDS_FILE = "detector_events.json"
REVIEW_DATA_FILE = "review_data.json"
REVIEW_FILTERS = (
    "All",
    "Pending Review",
    "Accepted",
    "Rejected",
    "Needs Further Review",
)
REVIEW_STATUS_LABELS = {
    ReviewStatus.PENDING_REVIEW: "Pending Review",
    ReviewStatus.UNDER_REVIEW: "Needs Further Review",
    ReviewStatus.REVIEWED_ACCEPTED: "Accepted",
    ReviewStatus.REVIEWED_REJECTED: "Rejected",
}


def _clear_boundary_state() -> None:
    for key in (
        "boundary_point_a",
        "track_boundary",
        "boundary_preview_rgb",
        "boundary_preview_timestamp",
        "boundary_timestamp",
        "boundary_last_click_time",
    ):
        st.session_state.pop(key, None)
    st.session_state["boundary_component_revision"] = (
        st.session_state.get("boundary_component_revision", 0) + 1
    )


def _clear_latest_run_state() -> None:
    for key in (
        "latest_run_summary",
        "latest_run_directory",
        "latest_run_output",
        "latest_run_evidence",
    ):
        st.session_state.pop(key, None)


def _existing_evidence_paths(evidence_paths: list[str]) -> list[Path]:
    return [path for reference in evidence_paths if (path := Path(reference)).is_file()]


def _current_run_events(session_state) -> object | None:
    summary = session_state.get("latest_run_summary")
    run_directory = session_state.get("latest_run_directory")
    output_path = session_state.get("latest_run_output")
    summary_output = getattr(summary, "output_path", None)
    if summary is None or not run_directory or not output_path or not summary_output:
        return None
    try:
        run_path = Path(run_directory).resolve()
        output = Path(output_path).resolve()
        summary_output_path = Path(summary_output).resolve()
        if output.parent != run_path or summary_output_path != output or not output.is_file():
            return None
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    return getattr(summary, "event_candidates", [])


def _validated_event_records(event_records: object) -> tuple[list[TrackEventRecord], int]:
    if event_records is None:
        return [], 0
    if not isinstance(event_records, (list, tuple)):
        return [], 1

    events = []
    seen_ids = set()
    malformed_count = 0
    for record in event_records:
        try:
            event = TrackEventRecord.model_validate(record)
        except (TypeError, ValueError):
            malformed_count += 1
            continue
        if event.event_id in seen_ids:
            malformed_count += 1
            continue
        seen_ids.add(event.event_id)
        events.append(event)
    return events, malformed_count


def _event_review_row(
    event: TrackEventRecord, review_status: ReviewStatus | None = None
) -> dict[str, object]:
    return {
        "Event ID": event.event_id,
        "Event type": event.event_type.value,
        "Timestamp": event.timestamp.isoformat(sep=" ", timespec="seconds")
        if event.timestamp
        else "—",
        "Video time (s)": f"{event.video_time_seconds:.2f}"
        if event.video_time_seconds is not None
        else "—",
        "Frame": event.frame_index if event.frame_index is not None else "—",
        "Track ID": event.track_id if event.track_id is not None else "—",
        "Confidence": f"{event.confidence:.3f}" if event.confidence is not None else "—",
        "Review status": REVIEW_STATUS_LABELS[review_status or event.review_status],
    }


def _event_review_rows(event_records: object) -> tuple[list[dict[str, object]], int]:
    events, malformed_count = _validated_event_records(event_records)
    return [_event_review_row(event) for event in events], malformed_count


def _filter_event_review_rows(rows: list[dict[str, object]], selected_filter: str) -> list[dict[str, object]]:
    if selected_filter not in REVIEW_FILTERS or selected_filter == "All":
        return rows
    return [row for row in rows if row["Review status"] == selected_filter]


def _event_review_details(
    event: TrackEventRecord, review_status: ReviewStatus | None = None
) -> dict[str, str]:
    details = {
        "Event ID": event.event_id,
        "Event type": event.event_type.value,
        "Review status": REVIEW_STATUS_LABELS[review_status or event.review_status],
    }
    if event.timestamp is not None:
        details["Timestamp"] = event.timestamp.isoformat(sep=" ", timespec="seconds")
    if event.video_time_seconds is not None:
        details["Video time"] = f"{event.video_time_seconds:.2f} s"
    if event.frame_index is not None:
        details["Frame"] = str(event.frame_index)
    if event.track_id is not None:
        details["Track ID"] = str(event.track_id)
    if event.confidence is not None:
        details["Confidence"] = f"{event.confidence:.3f}"
    if event.bbox is not None:
        details["Bounding box"] = (
            f"x1={event.bbox.x1:.1f}, y1={event.bbox.y1:.1f}, "
            f"x2={event.bbox.x2:.1f}, y2={event.bbox.y2:.1f}"
        )
    return details


def _sync_event_selection(session_state, run_id: str | None, event_ids: list[str]) -> None:
    if session_state.get("event_review_run_id") != run_id:
        session_state.pop("event_review_selected_event", None)
        session_state["event_review_run_id"] = run_id
        session_state["event_review_decisions"] = {}
        session_state["event_review_notes"] = {}
        session_state["event_review_note_owner"] = None
    if session_state.get("event_review_selected_event") not in event_ids:
        session_state.pop("event_review_selected_event", None)


def _set_event_review_status(
    session_state,
    run_id: str,
    selected_event_id: str | None,
    event_ids: list[str],
    status: ReviewStatus,
) -> bool:
    if (
        session_state.get("event_review_run_id") != run_id
        or selected_event_id not in event_ids
        or not isinstance(status, ReviewStatus)
    ):
        return False
    decisions = session_state.setdefault("event_review_decisions", {})
    decisions[selected_event_id] = status
    return True


def _event_review_status(session_state, event: TrackEventRecord) -> ReviewStatus:
    status = session_state.get("event_review_decisions", {}).get(event.event_id)
    return status if isinstance(status, ReviewStatus) else event.review_status


def _prepare_event_note_widget(session_state, run_id: str, event_id: str) -> None:
    owner = (run_id, event_id)
    if session_state.get("event_review_note_owner") != owner:
        session_state["event_review_note_widget"] = session_state.get(
            "event_review_notes", {}
        ).get(event_id, "")
        session_state["event_review_note_owner"] = owner


def _save_event_reviewer_note(
    session_state,
    run_id: str,
    selected_event_id: str | None,
    event_ids: list[str],
    note: str,
) -> bool:
    if (
        session_state.get("event_review_run_id") != run_id
        or selected_event_id not in event_ids
        or not isinstance(note, str)
        or len(note) > MAX_REVIEW_NOTE_CHARS
    ):
        return False
    notes = session_state.setdefault("event_review_notes", {})
    if note:
        notes[selected_event_id] = note
    else:
        notes.pop(selected_event_id, None)
    return True


def _run_json_path(run_directory: str | Path, filename: str) -> Path:
    if Path(filename).name != filename or "/" in filename or "\\" in filename:
        raise ValueError("Run data filename must be a simple name")
    run_path = Path(run_directory).resolve()
    target = run_path / filename
    resolved_target = target.resolve()
    if target.is_symlink() or resolved_target != target or resolved_target.parent != run_path:
        raise ValueError("Run data path must remain inside its run directory")
    return target


def _atomic_write_run_json(run_directory: str | Path, filename: str, payload: object) -> Path:
    run_path = Path(run_directory).resolve()
    if not run_path.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_path}")
    target = _run_json_path(run_path, filename)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{filename}.",
            suffix=".tmp",
            dir=run_path,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(payload, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, target)
        return target
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _persist_detector_events(run_directory: str | Path, events: list[TrackEventRecord]) -> Path:
    path = _run_json_path(run_directory, EVENT_RECORDS_FILE)
    if path.exists():
        raise FileExistsError(f"Detector event snapshot already exists: {path}")
    payload = [event.model_dump(mode="json") for event in events]
    return _atomic_write_run_json(run_directory, EVENT_RECORDS_FILE, payload)


def _load_detector_events(run_directory: str | Path, run_id: str) -> tuple[list[TrackEventRecord] | None, str | None]:
    try:
        run_path = Path(run_directory).resolve()
        if run_path.name != run_id:
            return None, "Selected run ID does not match its directory."
        path = _run_json_path(run_path, EVENT_RECORDS_FILE)
        if not path.is_file():
            return None, "This run has no saved detector event records."
        payload = json.loads(path.read_text(encoding="utf-8"))
        events, malformed_count = _validated_event_records(payload)
        if malformed_count:
            return events, f"Skipped {malformed_count} malformed detector event record(s)."
        return events, None
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
        return None, "Saved detector event records are malformed or unavailable."


def _available_review_runs(run_root: str | Path) -> list[Path]:
    try:
        root = Path(run_root).resolve()
        if not root.is_dir():
            return []
        entries = list(root.iterdir())
    except (OSError, RuntimeError, ValueError):
        return []

    runs = []
    for run_path in entries:
        try:
            resolved_run = run_path.resolve()
            if not run_path.is_dir() or resolved_run.parent != root:
                continue
            output_path = (resolved_run / "annotated_video.mp4").resolve()
            event_path = _run_json_path(resolved_run, EVENT_RECORDS_FILE)
            if output_path.parent == resolved_run and output_path.is_file() and event_path.is_file():
                runs.append((resolved_run.stat().st_mtime_ns, resolved_run.name, resolved_run))
        except (OSError, RuntimeError, ValueError):
            continue
    return [run for _, _, run in sorted(runs, reverse=True)]


def _load_review_data(
    run_directory: str | Path, run_id: str, event_ids: list[str]
) -> tuple[EventReviewData, str | None]:
    empty_data = EventReviewData(run_id=run_id)
    try:
        run_path = Path(run_directory).resolve()
        if run_path.name != run_id:
            return empty_data, "Review data run ID does not match its directory."
        path = _run_json_path(run_path, REVIEW_DATA_FILE)
        if not path.is_file():
            return empty_data, None
        data = EventReviewData.model_validate_json(path.read_text(encoding="utf-8"))
        if data.run_id != run_id:
            return empty_data, "Review data belongs to a different run and was ignored."
        allowed_ids = set(event_ids)
        reviews = {event_id: review for event_id, review in data.reviews.items() if event_id in allowed_ids}
        return EventReviewData(run_id=run_id, reviews=reviews), None
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
        return empty_data, "Saved review data is malformed or unavailable; no decisions were loaded."


def _save_review_data(run_directory: str | Path, data: EventReviewData, run_id: str) -> Path:
    run_path = Path(run_directory).resolve()
    if run_path.name != run_id or data.run_id != run_id:
        raise ValueError("Review data must match the current run ID")
    target = _run_json_path(run_path, REVIEW_DATA_FILE)
    if target.exists():
        try:
            current_data = EventReviewData.model_validate_json(
                target.read_text(encoding="utf-8")
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(
                "Existing review data is malformed and was preserved. Back it up or repair/delete "
                "review_data.json before retrying."
            ) from exc
        if current_data.run_id != run_id:
            raise ValueError("Existing review data belongs to a different run and was preserved.")
        if data.revision != current_data.revision + 1:
            raise RuntimeError("Review data changed since it was loaded; reload before saving.")
    elif data.revision not in (0, 1):
        raise RuntimeError("Review data revision is stale because its source file is missing.")
    return _atomic_write_run_json(
        run_path, REVIEW_DATA_FILE, data.model_dump(mode="json")
    )


def _updated_review_data(
    data: EventReviewData,
    run_id: str,
    event_ids: list[str],
    event_id: str | None,
    base_status: ReviewStatus,
    status: ReviewStatus | None = None,
    notes: str | None = None,
    update_notes: bool = False,
) -> EventReviewData | None:
    if data.run_id != run_id or event_id not in event_ids:
        return None
    existing = data.reviews.get(event_id)
    entry = existing or EventReviewEntry(status=base_status)
    try:
        updated_entry = EventReviewEntry(
            status=status or entry.status,
            notes=(notes or None) if update_notes else entry.notes,
        )
    except ValidationError:
        return None
    reviews = dict(data.reviews)
    reviews[event_id] = updated_entry
    return EventReviewData(run_id=run_id, revision=data.revision + 1, reviews=reviews)


def _resolve_run_evidence_path(reference: str | None, run_directory: str | Path) -> Path | None:
    if not isinstance(reference, str) or not reference.strip():
        return None
    try:
        run_path = Path(run_directory).resolve()
        evidence_root = (run_path / "evidence").resolve()
        if not evidence_root.is_relative_to(run_path):
            return None
        reference_path = Path(reference)
        candidate = reference_path if reference_path.is_absolute() else evidence_root / reference_path
        resolved = candidate.resolve()
        if resolved.is_relative_to(evidence_root) and resolved.is_file():
            return resolved
    except (OSError, RuntimeError, ValueError):
        return None
    return None


def _video_has_decodable_frame(path: Path) -> bool:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return False
        ok, frame = capture.read()
        return bool(ok and frame is not None and frame.size > 0)
    finally:
        capture.release()


def _render_event_evidence(event: TrackEventRecord, run_directory: str | Path) -> None:
    st.markdown('<div class="tg-section-title">Captured evidence</div>', unsafe_allow_html=True)
    image_reference = event.evidence_frame_reference
    if not image_reference:
        st.info("No evidence image was captured for this candidate.")
    else:
        image_path = _resolve_run_evidence_path(image_reference, run_directory)
        if image_path is None:
            st.warning("Evidence image is missing or is outside the current run's evidence folder.")
        else:
            try:
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            except (cv2.error, OSError):
                image = None
            if image is None or image.size == 0:
                st.warning("Evidence image exists but is invalid or unreadable.")
            else:
                st.image(image, channels="BGR", caption=f"Evidence frame · {event.event_id}")

    clip_reference = event.evidence_clip_reference
    if not clip_reference:
        st.info("No evidence video clip was captured for this candidate.")
    else:
        clip_path = _resolve_run_evidence_path(clip_reference, run_directory)
        if clip_path is None:
            st.warning("Evidence video is missing or is outside the current run's evidence folder.")
        else:
            try:
                clip_is_valid = _video_has_decodable_frame(clip_path)
            except (cv2.error, OSError):
                clip_is_valid = False
            if clip_is_valid:
                st.video(str(clip_path))
            else:
                st.warning("Evidence video exists but is invalid or unreadable.")


def _remember_video_upload() -> None:
    """Keep the uploaded file reference in this Streamlit session across navigation."""
    uploaded_video = st.session_state.get("analysis_video_widget")
    if uploaded_video is not st.session_state.get("analysis_video_upload"):
        _clear_latest_run_state()
        _clear_boundary_state()
        st.session_state["analysis_video_generation"] = (
            st.session_state.get("analysis_video_generation", 0) + 1
        )
        st.session_state["boundary_timestamp_widget"] = 0.0
        st.session_state["boundary_timestamp"] = 0.0
    st.session_state["analysis_video_upload"] = uploaded_video


def _extract_preview_frame(uploaded_video, timestamp_seconds: float, extension: str) -> np.ndarray:
    """Stream upload bytes to FFmpeg and return one bounded RGB preview frame."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        "1",
        "-i",
        "pipe:0",
        "-ss",
        f"{timestamp_seconds:.3f}",
        "-frames:v",
        "1",
        "-vf",
        f"scale={MAX_PREVIEW_WIDTH}:{MAX_PREVIEW_HEIGHT}:force_original_aspect_ratio=decrease",
        "-threads:v",
        "1",
        "-c:v",
        "mjpeg",
        "-q:v",
        "4",
        "-f",
        "image2pipe",
        "pipe:1",
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=UPLOAD_CHUNK_BYTES,
    )
    stderr_tail = bytearray()
    input_errors: list[OSError] = []
    frame_data = bytearray()
    timed_out = threading.Event()

    def feed_upload() -> None:
        try:
            uploaded_video.seek(0)
            while chunk := uploaded_video.read(UPLOAD_CHUNK_BYTES):
                process.stdin.write(chunk)
        except (BrokenPipeError, OSError) as exc:
            input_errors.append(exc)
        finally:
            try:
                process.stdin.close()
            except OSError:
                pass

    def drain_stderr() -> None:
        while chunk := process.stderr.read(4096):
            stderr_tail.extend(chunk)
            if len(stderr_tail) > 8192:
                del stderr_tail[:-8192]

    def stop_after_timeout() -> None:
        timed_out.set()
        if process.poll() is None:
            process.kill()

    feeder = threading.Thread(target=feed_upload, name="trackguard-upload-feed", daemon=True)
    stderr_reader = threading.Thread(target=drain_stderr, name="trackguard-ffmpeg-stderr", daemon=True)
    timeout = threading.Timer(FRAME_EXTRACTION_TIMEOUT_SECONDS, stop_after_timeout)
    try:
        feeder.start()
        stderr_reader.start()
        timeout.start()
        while chunk := process.stdout.read(64 * 1024):
            if len(frame_data) + len(chunk) > MAX_PREVIEW_FRAME_BYTES:
                raise RuntimeError("FFmpeg preview frame exceeded the 8 MiB output limit.")
            frame_data.extend(chunk)
        return_code = process.wait(timeout=10)
        feeder.join(timeout=10)
        stderr_reader.join(timeout=10)
    except Exception:
        if process.poll() is None:
            process.kill()
        process.wait()
        raise
    finally:
        timeout.cancel()
        if feeder.ident is not None:
            feeder.join(timeout=10)
        if stderr_reader.ident is not None:
            stderr_reader.join(timeout=10)
        uploaded_video.seek(0)
        for pipe in (process.stdin, process.stdout, process.stderr):
            if pipe is not None and not pipe.closed:
                pipe.close()

    if timed_out.is_set():
        raise RuntimeError("Frame extraction exceeded the 120-second limit. Choose an earlier timestamp.")
    if feeder.is_alive() or stderr_reader.is_alive():
        raise RuntimeError("FFmpeg did not release its input/output streams cleanly.")
    if not frame_data:
        diagnostics = stderr_tail.decode("utf-8", errors="replace").strip()
        nonseekable_message = (
            "This video could not be read through a non-seekable stream. "
            "Some MP4/MOV files place their index at the end. Re-export it as a fast-start/streamable file; "
            "TrackGuard will not create a disk copy."
        )
        if any(
            term in diagnostics.lower()
            for term in ("moov atom not found", "partial file", "seek", "could not find atom")
        ):
            raise RuntimeError(nonseekable_message)
        if diagnostics:
            format_hint = (
                " For MP4/MOV, try a fast-start export if the file index is at the end."
                if extension in {"mp4", "mov"} else ""
            )
            raise RuntimeError(f"FFmpeg could not extract this frame: {diagnostics[-1200:]}{format_hint}")
        if input_errors:
            raise RuntimeError(f"Could not stream uploaded video to FFmpeg: {input_errors[-1]}")
        if return_code != 0:
            raise RuntimeError("FFmpeg could not read this video. Check that the file is valid and try again.")
        raise RuntimeError("No frame exists at that timestamp. Choose an earlier time.")

    encoded_frame = np.frombuffer(frame_data, dtype=np.uint8)
    preview_bgr = cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR)
    if preview_bgr is None:
        raise RuntimeError("FFmpeg returned an unreadable preview frame.")
    return cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)


def _normalized_click(click: dict) -> tuple[float, float] | None:
    """Convert component display coordinates to aspect-preserving normalized coordinates."""
    try:
        x = float(click["x"])
        y = float(click["y"])
        display_width = float(click["width"])
        display_height = float(click["height"])
    except (KeyError, TypeError, ValueError):
        return None
    values = (x, y, display_width, display_height)
    if not all(math.isfinite(value) for value in values):
        return None
    if display_width <= 0 or display_height <= 0:
        return None
    if not (0 <= x <= display_width and 0 <= y <= display_height):
        return None
    # The image is scaled uniformly; these ratios equal source_x/source_width
    # and source_y/source_height, so source dimensions need not be duplicated.
    return x / display_width, y / display_height


def _render_boundary_configuration(uploaded_video, extension: str) -> None:
    st.markdown('<div class="tg-section-title">3 · Virtual boundary</div>', unsafe_allow_html=True)
    st.caption(
        "Choose a representative time, load its frame, then click Point A and Point B. "
        "The line is a virtual geometric reference for crossing candidates, not an official track limit."
    )
    timestamp = st.number_input(
        "Frame timestamp (seconds)",
        min_value=0.0,
        value=float(st.session_state.get("boundary_timestamp", 0.0)),
        step=1.0,
        key="boundary_timestamp_widget",
        on_change=_remember_boundary_timestamp,
        help="Use the video preview to choose a time. Later timestamps can take longer to decode.",
    )
    st.session_state["boundary_timestamp"] = timestamp
    if st.button("Load selected frame", key="load_boundary_frame"):
        _reset_boundary_points()
        st.session_state.pop("boundary_preview_rgb", None)
        st.session_state.pop("boundary_preview_timestamp", None)
        try:
            st.session_state["boundary_preview_rgb"] = _extract_preview_frame(
                uploaded_video, timestamp, extension
            )
            st.session_state["boundary_preview_timestamp"] = timestamp
        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
            st.error(str(exc))

    preview_rgb = st.session_state.get("boundary_preview_rgb")
    if preview_rgb is None:
        st.info("Load a frame to configure a virtual boundary.")
        return

    st.caption(f"Representative frame loaded at {st.session_state['boundary_preview_timestamp']:.2f} seconds.")
    preview = preview_rgb.copy()
    preview_height, preview_width = preview.shape[:2]
    boundary = st.session_state.get("track_boundary")
    point_a = st.session_state.get("boundary_point_a")
    if boundary is not None:
        point_a = (boundary.start_x, boundary.start_y)
        point_b = (boundary.end_x, boundary.end_y)
        a_pixel = (round(point_a[0] * (preview_width - 1)), round(point_a[1] * (preview_height - 1)))
        b_pixel = (round(point_b[0] * (preview_width - 1)), round(point_b[1] * (preview_height - 1)))
        cv2.line(preview, a_pixel, b_pixel, (255, 190, 40), 3, cv2.LINE_AA)
        cv2.circle(preview, a_pixel, 9, (60, 220, 120), -1, cv2.LINE_AA)
        cv2.circle(preview, b_pixel, 9, (70, 120, 255), -1, cv2.LINE_AA)
        cv2.putText(preview, "A", (a_pixel[0] + 12, a_pixel[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (60, 220, 120), 2, cv2.LINE_AA)
        cv2.putText(preview, "B", (b_pixel[0] + 12, b_pixel[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (70, 120, 255), 2, cv2.LINE_AA)
    elif point_a is not None:
        a_pixel = (round(point_a[0] * (preview_width - 1)), round(point_a[1] * (preview_height - 1)))
        cv2.circle(preview, a_pixel, 9, (60, 220, 120), -1, cv2.LINE_AA)
        cv2.putText(preview, "A", (a_pixel[0] + 12, a_pixel[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (60, 220, 120), 2, cv2.LINE_AA)

    upload_generation = st.session_state.get("analysis_video_generation", 0)
    revision = st.session_state.get("boundary_component_revision", 0)
    component_value = streamlit_image_coordinates(
        preview,
        width=960,
        key=f"boundary_click_{upload_generation}_{revision}",
        image_format="JPEG",
        jpeg_quality=85,
        cursor="crosshair",
    )
    if component_value:
        click_time = component_value.get("unix_time")
        click_signature = (click_time, component_value.get("x"), component_value.get("y"))
        if click_signature != st.session_state.get("boundary_last_click_time"):
            st.session_state["boundary_last_click_time"] = click_signature
            normalized_point = _normalized_click(component_value)
            if normalized_point is None:
                st.error("The image click returned invalid coordinates. Please click inside the frame.")
            elif st.session_state.get("track_boundary") is not None:
                st.info("Boundary is set. Reset it before selecting another pair of points.")
            elif st.session_state.get("boundary_point_a") is None:
                st.session_state["boundary_point_a"] = normalized_point
                st.success("Point A set. Click the frame again to place Point B.")
            else:
                point_a = st.session_state["boundary_point_a"]
                try:
                    st.session_state["track_boundary"] = TrackBoundary(
                        start_x=point_a[0],
                        start_y=point_a[1],
                        end_x=normalized_point[0],
                        end_y=normalized_point[1],
                    )
                    st.session_state.pop("boundary_point_a", None)
                    st.success("Boundary configured and validated.")
                except ValidationError as exc:
                    st.error(f"Invalid boundary: {exc.errors()[0]['msg']}")

    boundary = st.session_state.get("track_boundary")
    point_a = st.session_state.get("boundary_point_a")
    if boundary is not None:
        st.success("Validated virtual boundary saved for this video session.")
        st.caption(
            f"A=({boundary.start_x:.4f}, {boundary.start_y:.4f}) · "
            f"B=({boundary.end_x:.4f}, {boundary.end_y:.4f})"
        )
    elif point_a is not None:
        st.info(f"Point A=({point_a[0]:.4f}, {point_a[1]:.4f}) · click to set Point B.")
    else:
        st.info("No boundary configured. Click the frame to set Point A.")
    if boundary is not None or point_a is not None:
        st.button("Reset boundary points", on_click=_reset_boundary_points, key="reset_boundary_points")


def _reset_boundary_points() -> None:
    st.session_state.pop("boundary_point_a", None)
    st.session_state.pop("track_boundary", None)
    st.session_state.pop("boundary_last_click_time", None)
    st.session_state["boundary_component_revision"] = (
        st.session_state.get("boundary_component_revision", 0) + 1
    )


def _remember_confidence() -> None:
    st.session_state["analysis_confidence"] = st.session_state["analysis_confidence_widget"]


def _remember_device() -> None:
    st.session_state["analysis_device"] = st.session_state["analysis_device_widget"]


def _remember_boundary_timestamp() -> None:
    st.session_state["boundary_timestamp"] = st.session_state["boundary_timestamp_widget"]


def _process_uploaded_video(uploaded_video, confidence: float, device: str, boundary: TrackBoundary):
    """Run the existing detector with unique outputs and bounded upload copying."""
    project_root = Path(__file__).resolve().parent
    run_root = project_root / "runs" / "streamlit_runs"
    run_root.mkdir(parents=True, exist_ok=True)
    run_directory = run_root / uuid.uuid4().hex
    run_directory.mkdir(exist_ok=False)

    extension = uploaded_video.name.rsplit(".", 1)[-1].lower()
    temporary_input_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="trackguard_", suffix=f".{extension}", delete=False
        ) as temporary_input:
            temporary_input_path = Path(temporary_input.name)
            uploaded_video.seek(0)
            copied_bytes = 0
            while chunk := uploaded_video.read(UPLOAD_CHUNK_BYTES):
                copied_bytes += len(chunk)
                if copied_bytes > MAX_UPLOAD_MB * 1024 * 1024:
                    raise ValueError(f"Uploaded video exceeds the {MAX_UPLOAD_MB} MB limit.")
                temporary_input.write(chunk)
            uploaded_video.seek(0)

        if copied_bytes == 0:
            raise ValueError("Uploaded video is empty.")

        output_path = run_directory / "annotated_video.mp4"
        evidence_directory = run_directory / "evidence"
        config = TrackGuardConfig(
            confidence_threshold=confidence,
            device=device,
            track_boundary=boundary,
            evidence_dir=str(evidence_directory),
        )

        # Keep model/PyTorch imports out of ordinary dashboard startup.
        from src.detector import process_video

        summary = process_video(
            input_path=temporary_input_path,
            output_path=output_path,
            config=config,
        )
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError("Video processing returned without a readable output file.")

        _persist_detector_events(run_directory, summary.event_candidates)
        evidence_paths = []
        for event in summary.event_candidates:
            if event.evidence_frame_reference:
                evidence_paths.append(event.evidence_frame_reference)
            if event.evidence_clip_reference:
                evidence_paths.append(event.evidence_clip_reference)
        return summary, run_directory, output_path, evidence_paths
    except Exception:
        shutil.rmtree(run_directory, ignore_errors=True)
        raise
    finally:
        if temporary_input_path is not None:
            temporary_input_path.unlink(missing_ok=True)


st.set_page_config(
    page_title="TrackGuard AI | Race Control",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {
        --tg-bg: #0b1118;
        --tg-panel: #111b25;
        --tg-border: #263543;
        --tg-muted: #91a1b2;
        --tg-text: #e8eef5;
        --tg-accent: #f0b44c;
        --tg-green: #52d3a2;
      }
      .stApp { background: var(--tg-bg); color: var(--tg-text); }
      [data-testid="stSidebar"] {
        background: #0e1720;
        border-right: 1px solid var(--tg-border);
      }
      [data-testid="stMetric"] {
        background: var(--tg-panel);
        border: 1px solid var(--tg-border);
        border-radius: 12px;
        padding: 16px 18px;
      }
      [data-testid="stMetricLabel"] p {
        color: var(--tg-muted);
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      [data-testid="stMetricValue"] { color: var(--tg-text); }
      .tg-brand-mark {
        color: var(--tg-accent);
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.2em;
      }
      .tg-hero {
        background: linear-gradient(115deg, #152331, #101923 68%, #17251f);
        border: 1px solid var(--tg-border);
        border-radius: 16px;
        padding: 27px 30px;
        margin: 8px 0 24px;
      }
      .tg-kicker {
        color: var(--tg-green);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
      }
      .tg-hero h1 { color: var(--tg-text); margin: 8px 0 6px; font-size: 2rem; }
      .tg-hero p { color: var(--tg-muted); margin: 0; }
      .tg-local {
        display: inline-block;
        color: var(--tg-green);
        background: #143127;
        border: 1px solid #245744;
        border-radius: 20px;
        padding: 7px 12px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
      }
      .tg-section-title {
        color: var(--tg-text);
        font-size: 1rem;
        font-weight: 650;
        margin: 24px 0 12px;
      }
      .tg-stage {
        background: var(--tg-panel);
        border: 1px solid var(--tg-border);
        border-radius: 11px;
        min-height: 100px;
        padding: 16px;
      }
      .tg-stage-label {
        color: var(--tg-muted);
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.09em;
      }
      .tg-stage-status { color: var(--tg-text); margin-top: 10px; font-size: 0.9rem; }
      .tg-sidebar-note { color: var(--tg-muted); font-size: 0.78rem; line-height: 1.5; }
      div[data-testid="stAlert"] { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown('<div class="tg-brand-mark">TRACKGUARD AI</div>', unsafe_allow_html=True)
    st.caption("MOTORSPORT VIDEO REVIEW")
    st.markdown("---")
    section = st.radio(
        "Workspace",
        ("Dashboard", "Video Analysis", "Event Review", "Evidence"),
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        '<div class="tg-sidebar-note">Local workspace<br>Session data only · not persisted</div>',
        unsafe_allow_html=True,
    )


if section == "Dashboard":
    st.markdown(
        """
        <div class="tg-hero">
          <div class="tg-kicker">Trackside decision support</div>
          <h1>Race Control Overview</h1>
          <p>Vehicle tracking and reviewable boundary-crossing candidates, organized for human review.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(4)
    metrics = (
        ("Processed videos", "Unavailable"),
        ("Detected vehicles", "Unavailable"),
        ("Crossing candidates", "Unavailable"),
        ("Pending reviews", "Unavailable"),
    )
    for column, (label, value) in zip(metric_columns, metrics):
        with column:
            st.metric(label, value)
            st.caption("Awaiting video processing")

    st.markdown('<div class="tg-section-title">Analysis workflow</div>', unsafe_allow_html=True)
    stages = st.columns(4)
    for column, label in zip(
        stages,
        ("01 · VIDEO INGEST", "02 · VEHICLE TRACKING", "03 · CANDIDATE REVIEW", "04 · EVIDENCE"),
    ):
        with column:
            st.markdown(
                f'<div class="tg-stage"><div class="tg-stage-label">{label}</div>'
                '<div class="tg-stage-status">Awaiting first analysis</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "TrackGuard reports observations as candidates for human review. "
        "It does not determine or declare official FIA violations."
    )
elif section == "Video Analysis":
    st.title("Video Analysis")
    st.caption("Select a local race video, configure a virtual boundary, and run the analysis locally.")

    st.markdown('<div class="tg-section-title">1 · Select a video</div>', unsafe_allow_html=True)
    st.caption(
        "Supported formats: MP4, MOV, AVI, MKV. Maximum upload size: 500 MB. "
        "Uploads remain in this Streamlit session and are not saved to project folders."
    )
    uploaded_widget = st.file_uploader(
        "Choose a video file",
        type=list(SUPPORTED_VIDEO_TYPES),
        accept_multiple_files=False,
        max_upload_size=MAX_UPLOAD_MB,
        key="analysis_video_widget",
        on_change=_remember_video_upload,
        help="Only MP4, MOV, AVI, and MKV files up to 500 MB are accepted.",
    )
    if uploaded_widget is not None:
        st.session_state["analysis_video_upload"] = uploaded_widget
    uploaded_video = st.session_state.get("analysis_video_upload")

    video_valid = False
    if uploaded_video is None:
        st.info("No video uploaded yet. Choose a supported file to continue.")
    else:
        original_name = uploaded_video.name
        extension = original_name.rsplit(".", 1)[-1].lower() if "." in original_name else ""
        file_size = uploaded_video.size
        extension_valid = extension in SUPPORTED_VIDEO_TYPES
        size_valid = file_size <= MAX_UPLOAD_MB * 1024 * 1024

        details_left, details_right = st.columns(2)
        with details_left:
            st.markdown("**Original filename**")
            st.text(original_name)
        with details_right:
            st.markdown("**File details**")
            st.write(f"{file_size / (1024 * 1024):.2f} MB · .{extension or 'unknown'}")

        if not extension_valid:
            st.error("Unsupported file extension. Choose an MP4, MOV, AVI, or MKV video.")
        elif file_size == 0:
            st.error("This upload is empty. Choose a non-empty video file.")
        elif not size_valid:
            st.error(f"This file exceeds the {MAX_UPLOAD_MB} MB upload limit.")
        else:
            video_valid = True
            st.success("Upload validation passed: supported extension, non-empty file, and within size limit.")
            st.markdown("**Video preview**")
            uploaded_video.seek(0)
            st.video(uploaded_video, format=VIDEO_MIME_TYPES[extension])
            st.caption("Playback depends on browser support for the video's container and codec.")

    st.markdown('<div class="tg-section-title">2 · Inference configuration</div>', unsafe_allow_html=True)
    config_left, config_right = st.columns(2)
    with config_left:
        confidence = st.slider(
            "Confidence threshold",
            min_value=0.10,
            max_value=1.00,
            value=float(st.session_state.get("analysis_confidence", 0.25)),
            step=0.05,
            key="analysis_confidence_widget",
            on_change=_remember_confidence,
            help="Minimum confidence score requested for retaining a vehicle detection.",
        )
        st.caption("Lower values retain more candidate detections; higher values apply a stricter threshold.")
        st.session_state["analysis_confidence"] = confidence

    with config_right:
        device = st.selectbox(
            "Inference device",
            options=("Auto", "CPU", "CUDA"),
            index=("Auto", "CPU", "CUDA").index(
                st.session_state.get("analysis_device", "Auto")
            ),
            key="analysis_device_widget",
            on_change=_remember_device,
        )
        device_explanations = {
            "Auto": "Let the pipeline resolve the available inference device during processing.",
            "CPU": "Request CPU inference. Performance has not been evaluated here.",
            "CUDA": "Request CUDA. Availability has not been checked.",
        }
        st.caption(device_explanations[device])
        st.session_state["analysis_device"] = device

    if video_valid:
        _render_boundary_configuration(uploaded_video, extension)
    else:
        st.info("Upload a valid video before configuring a virtual boundary.")

    st.markdown('<div class="tg-section-title">4 · Ready for Processing</div>', unsafe_allow_html=True)
    configured_boundary = st.session_state.get("track_boundary")
    can_process = video_valid and configured_boundary is not None
    missing_requirements = []
    if not video_valid:
        missing_requirements.append("upload a valid video")
    if configured_boundary is None:
        missing_requirements.append("configure a valid virtual boundary")
    if can_process:
        st.success("Video and boundary configuration are ready for analysis.")
    else:
        st.info("To enable processing, " + " and ".join(missing_requirements) + ".")

    prepared_left, prepared_right = st.columns(2)
    with prepared_left:
        st.write(f"**Video:** {'Valid upload' if video_valid else 'Unavailable'}")
        st.write(f"**Confidence threshold:** {confidence:.2f}")
        st.write(f"**Selected device:** {device}")
        st.write(f"**Virtual boundary:** {'Configured' if configured_boundary is not None else 'Not configured'}")
    with prepared_right:
        run_analysis = st.button("Run video analysis", disabled=not can_process, type="primary")
        st.caption("YOLO + ByteTrack processing runs locally when selected.")

    if run_analysis:
        _clear_latest_run_state()
        requested_device = {"Auto": "auto", "CPU": "cpu", "CUDA": "cuda"}[device]
        try:
            with st.spinner("Running vehicle detection, tracking, and boundary candidate analysis…"):
                summary, run_directory, output_path, evidence_paths = _process_uploaded_video(
                    uploaded_video,
                    confidence,
                    requested_device,
                    configured_boundary,
                )
        except Exception as exc:
            _clear_latest_run_state()
            st.error(f"Video analysis failed: {exc}")
        else:
            st.session_state["latest_run_summary"] = summary
            st.session_state["latest_run_directory"] = str(run_directory)
            st.session_state["latest_run_output"] = str(output_path)
            st.session_state["latest_run_evidence"] = evidence_paths
            st.session_state.pop("event_review_active_run", None)

    latest_summary = st.session_state.get("latest_run_summary")
    latest_output = st.session_state.get("latest_run_output")
    if latest_summary is not None and latest_output and Path(latest_output).is_file():
        st.markdown('<div class="tg-section-title">Latest completed run</div>', unsafe_allow_html=True)
        metric_columns = st.columns(5)
        run_metrics = (
            ("Frames", latest_summary.total_frames),
            ("Vehicle detections", latest_summary.vehicle_detections_count),
            ("Crossing candidates", len(latest_summary.event_candidates)),
            ("Processing FPS", f"{latest_summary.processing_fps:.2f}"),
            ("Audio", latest_summary.audio_status),
        )
        for column, (label, value) in zip(metric_columns, run_metrics):
            with column:
                st.metric(label, value)
        st.video(latest_output)
        st.caption(f"Run output: {latest_output}")
        evidence_references = st.session_state.get("latest_run_evidence", [])
        evidence_paths = _existing_evidence_paths(evidence_references)
        if evidence_paths:
            st.caption("Evidence artifacts: " + " · ".join(str(path) for path in evidence_paths))
        elif evidence_references:
            st.info("Evidence references were returned, but the referenced files are unavailable.")
        else:
            st.caption("No boundary-crossing evidence was generated in this run.")

elif section == "Event Review":
    st.title("Event Review Queue")
    st.caption(
        "Review candidate observations from a successful local run. "
        "Every candidate requires human review and is not an official FIA violation."
    )
    run_root = Path(__file__).resolve().parent / "runs" / "streamlit_runs"
    available_runs = _available_review_runs(run_root)
    run_ids = [run_path.name for run_path in available_runs]
    if not run_ids:
        _sync_event_selection(st.session_state, None, [])
        st.info("No completed runs with saved event records are available yet. Process a video to populate the review queue.")
    else:
        latest_run_directory = st.session_state.get("latest_run_directory")
        latest_run_id = Path(latest_run_directory).name if latest_run_directory else None
        active_run_id = st.session_state.get("event_review_active_run")
        if active_run_id not in run_ids:
            active_run_id = latest_run_id if latest_run_id in run_ids else run_ids[0]
            st.session_state["event_review_active_run"] = active_run_id
        selected_run_id = st.selectbox(
            "Successful run",
            options=run_ids,
            key="event_review_active_run",
        )
        run_directory = next(path for path in available_runs if path.name == selected_run_id)
        event_records, event_load_warning = _load_detector_events(run_directory, selected_run_id)
        if event_records is None:
            _sync_event_selection(st.session_state, None, [])
            st.warning(event_load_warning or "Could not load this run's event records.")
        else:
            st.caption(f"Run ID: {selected_run_id}")
            if event_load_warning:
                st.warning(event_load_warning)
            events, malformed_count = _validated_event_records(event_records)
            _sync_event_selection(
                st.session_state, selected_run_id, [event.event_id for event in events]
            )
            review_data, review_load_warning = _load_review_data(
                run_directory,
                selected_run_id,
                [event.event_id for event in events],
            )
            if review_load_warning:
                st.warning(review_load_warning)
            st.session_state["event_review_decisions"] = {
                event_id: review.status for event_id, review in review_data.reviews.items()
            }
            st.session_state["event_review_notes"] = {
                event_id: review.notes
                for event_id, review in review_data.reviews.items()
                if review.notes is not None
            }
            event_rows = [
                _event_review_row(event, _event_review_status(st.session_state, event))
                for event in events
            ]
            if malformed_count:
                st.warning(
                    f"Skipped {malformed_count} malformed event record(s). "
                    "The remaining validated candidates are shown below."
                )
            if not event_rows:
                _sync_event_selection(st.session_state, selected_run_id, [])
                st.info("No valid crossing candidates are available for this run.")
            else:
                selected_filter = st.selectbox(
                    "Review status",
                    options=REVIEW_FILTERS,
                    key="event_review_status_filter",
                )
                filtered_rows = _filter_event_review_rows(event_rows, selected_filter)
                if filtered_rows:
                    st.dataframe(filtered_rows, use_container_width=True, hide_index=True)
                    visible_event_ids = [row["Event ID"] for row in filtered_rows]
                    _sync_event_selection(st.session_state, selected_run_id, visible_event_ids)
                    selected_event_id = st.selectbox(
                        "Select a candidate",
                        options=visible_event_ids,
                        key="event_review_selected_event",
                    )
                    selected_event = next(
                        (event for event in events if event.event_id == selected_event_id), None
                    )
                    if selected_event is not None:
                        st.markdown('<div class="tg-section-title">Candidate details</div>', unsafe_allow_html=True)
                        detail_columns = st.columns(3)
                        current_status = _event_review_status(st.session_state, selected_event)
                        for index, (label, value) in enumerate(
                            _event_review_details(selected_event, current_status).items()
                        ):
                            with detail_columns[index % len(detail_columns)]:
                                st.markdown(f"**{label}**")
                                st.write(value)

                        _prepare_event_note_widget(
                            st.session_state, selected_run_id, selected_event.event_id
                        )
                        reviewer_note = st.text_area(
                            "Optional reviewer notes",
                            key="event_review_note_widget",
                            max_chars=MAX_REVIEW_NOTE_CHARS,
                            help="Notes are saved locally with this run.",
                        )
                        current_note_entry = review_data.reviews.get(selected_event.event_id)
                        current_note = current_note_entry.notes if current_note_entry else None
                        if reviewer_note != (current_note or ""):
                            updated_data = _updated_review_data(
                                review_data,
                                selected_run_id,
                                visible_event_ids,
                                selected_event.event_id,
                                current_status,
                                notes=reviewer_note,
                                update_notes=True,
                            )
                            try:
                                if updated_data is None:
                                    raise ValueError("Selected candidate is no longer available.")
                                _save_review_data(run_directory, updated_data, selected_run_id)
                            except Exception as exc:
                                st.error(f"Could not save reviewer notes: {exc}")
                            else:
                                if _save_event_reviewer_note(
                                    st.session_state,
                                    selected_run_id,
                                    selected_event.event_id,
                                    visible_event_ids,
                                    reviewer_note,
                                ):
                                    review_data = updated_data
                                else:
                                    st.error("Could not associate notes with the selected candidate.")

                        action_columns = st.columns(3)
                        with action_columns[0]:
                            accept_clicked = st.button("Accept Candidate", key="event_review_accept")
                        with action_columns[1]:
                            reject_clicked = st.button("Reject Candidate", key="event_review_reject")
                        with action_columns[2]:
                            further_review_clicked = st.button(
                                "Needs Further Review", key="event_review_further_review"
                            )
                        selected_status = None
                        if accept_clicked:
                            selected_status = ReviewStatus.REVIEWED_ACCEPTED
                        elif reject_clicked:
                            selected_status = ReviewStatus.REVIEWED_REJECTED
                        elif further_review_clicked:
                            selected_status = ReviewStatus.UNDER_REVIEW
                        if selected_status is not None:
                            updated_data = _updated_review_data(
                                review_data,
                                selected_run_id,
                                visible_event_ids,
                                selected_event.event_id,
                                current_status,
                                status=selected_status,
                            )
                            try:
                                if updated_data is None:
                                    raise ValueError("Selected candidate is no longer available.")
                                _save_review_data(run_directory, updated_data, selected_run_id)
                            except Exception as exc:
                                st.error(f"Could not save review decision: {exc}")
                            else:
                                if _set_event_review_status(
                                    st.session_state,
                                    selected_run_id,
                                    selected_event.event_id,
                                    visible_event_ids,
                                    selected_status,
                                ):
                                    st.rerun()
                                st.warning("This candidate is no longer available for a review action.")
                        _render_event_evidence(selected_event, run_directory)
                else:
                    _sync_event_selection(st.session_state, selected_run_id, [])
                    st.info(f"No candidates match the “{selected_filter}” filter.")
else:
    st.title(section)
    st.info(f"{section} is a navigation placeholder for the next dashboard implementation stage.")

