import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from src.schemas import (
    EventType,
    ProcessingJobInput,
    ProcessingJobResult,
    ReviewStatus,
    TrackEventRecord,
)
from src.processing import execute_processing_job


class TestProcessingContracts(unittest.TestCase):
    def test_job_input_serializes_and_validates(self):
        job = ProcessingJobInput(
            job_id="job-123",
            video_id="video-456",
            config={"device": "cpu", "confidence_threshold": 0.4},
            max_frames=120,
            enable_tracking=False,
        )

        serialized = job.model_dump_json()
        restored = ProcessingJobInput.model_validate_json(serialized)

        self.assertEqual(restored.job_id, "job-123")
        self.assertEqual(restored.video_id, "video-456")
        self.assertEqual(restored.config.device, "cpu")
        self.assertEqual(restored.max_frames, 120)
        self.assertFalse(restored.enable_tracking)
        self.assertNotIn("evidence_dir", job.model_dump(mode="json")["config"])
        with self.assertRaises(ValidationError):
            ProcessingJobInput(job_id="", video_id="video-456")
        with self.assertRaises(ValidationError):
            ProcessingJobInput(job_id="job-123", video_id="video-456", max_frames=0)

    def test_job_result_serializes_events_and_worker_neutral_reference(self):
        event = TrackEventRecord(
            event_id="evt-1",
            event_type=EventType.TRACK_LIMIT_CANDIDATE,
            frame_index=12,
            video_time_seconds=0.4,
            review_status=ReviewStatus.PENDING_REVIEW,
        )
        result = ProcessingJobResult(
            job_id="job-123",
            output_reference="results/job-123/annotated.mp4",
            total_frames=30,
            vehicle_detections_count=4,
            elapsed_seconds=1.2,
            processing_fps=25.0,
            video_width=320,
            video_height=240,
            source_fps=25.0,
            audio_status="preserved",
            event_candidates=[event],
        )

        restored = ProcessingJobResult.model_validate_json(result.model_dump_json())

        self.assertEqual(restored.output_reference, "results/job-123/annotated.mp4")
        self.assertEqual(restored.event_candidates[0].event_id, "evt-1")
        with self.assertRaises(ValidationError):
            ProcessingJobResult(
                job_id="job-123", output_reference="", total_frames=-1,
                vehicle_detections_count=0, elapsed_seconds=1.0,
                processing_fps=1.0, video_width=320, video_height=240,
                source_fps=25.0, audio_status="unavailable",
            )
        with self.assertRaises(ValidationError):
            ProcessingJobResult(
                job_id="job-123", output_reference="results/job-123/output.mp4",
                total_frames=1, vehicle_detections_count=0, elapsed_seconds=1.0,
                processing_fps=1.0, video_width=320, video_height=240,
                source_fps=25.0, audio_status="not-a-supported-status",
            )

    def test_execution_adapter_maps_contract_to_existing_pipeline(self):
        job = ProcessingJobInput(job_id="job-adapter", video_id="video-adapter")
        event = TrackEventRecord(
            event_id="evt-adapter",
            event_type=EventType.TRACK_LIMIT_CANDIDATE,
            evidence_reference="scratch/evidence/evt-adapter/event_clip.mp4",
            evidence_clip_reference="scratch/evidence/evt-adapter/event_clip.mp4",
        )
        summary = SimpleNamespace(
            total_frames=8,
            vehicle_detections_count=3,
            elapsed_seconds=0.5,
            processing_fps=16.0,
            video_width=640,
            video_height=360,
            source_fps=30.0,
            audio_status="preserved",
            event_candidates=[event],
        )
        progress = lambda item: None
        cancel = lambda: False

        with patch("src.processing.process_video", return_value=summary) as process:
            result = execute_processing_job(
                job,
                "scratch/input.mp4",
                "scratch/output.mp4",
                "object://output/job-adapter.mp4",
                progress_callback=progress,
                cancellation_requested=cancel,
            )

        self.assertEqual(result.job_id, "job-adapter")
        self.assertEqual(result.output_reference, "object://output/job-adapter.mp4")
        self.assertEqual(result.total_frames, 8)
        self.assertEqual(result.event_candidates[0].evidence_reference,
                         "evt-adapter/event_clip.mp4")
        self.assertEqual(event.evidence_reference,
                         "scratch/evidence/evt-adapter/event_clip.mp4")
        self.assertEqual(process.call_args.kwargs["input_path"], "scratch/input.mp4")
        self.assertEqual(process.call_args.kwargs["config"].evidence_dir,
                         str(Path("scratch") / "evidence"))
        self.assertIs(process.call_args.kwargs["progress_callback"], progress)
        self.assertIs(process.call_args.kwargs["cancellation_requested"], cancel)


if __name__ == "__main__":
    unittest.main()
