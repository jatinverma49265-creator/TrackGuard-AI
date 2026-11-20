"""Worker-neutral adapter from serialized jobs to the video engine.

Job references are resolved by the caller; this module only translates the
portable contract into the existing local-file processing function.
"""

from pathlib import Path
from typing import Any, Callable, Optional

from src.detector import process_video
from src.schemas import ProcessingJobInput, ProcessingJobResult, ProcessingProgress


def execute_processing_job(
    job: ProcessingJobInput,
    input_path: str | Path,
    output_path: str | Path,
    output_reference: str,
    *,
    model: Optional[Any] = None,
    progress_callback: Optional[Callable[[ProcessingProgress], None]] = None,
    cancellation_requested: Optional[Callable[[], bool]] = None,
) -> ProcessingJobResult:
    """Execute one portable job after an adapter resolves its video reference.

    ``input_path`` and ``output_path`` are execution-local paths and are not
    serialized into the job or result. ``output_reference`` is the opaque
    reference the caller will publish (for example, an object key).
    """
    local_output_path = Path(output_path)
    local_evidence_directory = local_output_path.parent / "evidence"
    config = job.config.model_copy(
        update={"evidence_dir": str(local_evidence_directory)}
    )
    summary = process_video(
        input_path=input_path,
        output_path=local_output_path,
        config=config,
        model=model,
        max_frames=job.max_frames,
        enable_tracking=job.enable_tracking,
        progress_callback=progress_callback,
        cancellation_requested=cancellation_requested,
    )
    evidence_root = local_evidence_directory.resolve()
    portable_events = []
    for event in summary.event_candidates:
        portable_event = event.model_copy(deep=True)
        for field_name in (
            "evidence_reference",
            "evidence_frame_reference",
            "evidence_clip_reference",
        ):
            reference = getattr(portable_event, field_name)
            if reference:
                try:
                    relative_reference = Path(reference).resolve().relative_to(evidence_root)
                except (OSError, RuntimeError, ValueError) as exc:
                    raise ValueError(
                        f"Evidence reference for {portable_event.event_id} is outside its job workspace."
                    ) from exc
                setattr(portable_event, field_name, relative_reference.as_posix())
        portable_events.append(portable_event)

    return ProcessingJobResult(
        job_id=job.job_id,
        output_reference=output_reference,
        total_frames=summary.total_frames,
        vehicle_detections_count=summary.vehicle_detections_count,
        elapsed_seconds=summary.elapsed_seconds,
        processing_fps=summary.processing_fps,
        video_width=summary.video_width,
        video_height=summary.video_height,
        source_fps=summary.source_fps,
        audio_status=summary.audio_status,
        event_candidates=portable_events,
    )
