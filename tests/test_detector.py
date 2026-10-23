import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np

from src.config import TrackBoundary, TrackGuardConfig
from src.detector import (
    COCO_VEHICLE_CLASSES,
    VEHICLE_CLASS_IDS,
    annotate_frame,
    process_video,
    resolve_device,
)


class TestDetector(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_synthetic_video(self, filename: str, num_frames: int = 5, width: int = 320, height: int = 240) -> Path:
        vid_path = self.temp_path / filename
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (width, height))
        for i in range(num_frames):
            frame = np.full((height, width, 3), fill_value=(i * 30) % 255, dtype=np.uint8)
            cv2.rectangle(frame, (20, 20), (80, 80), (255, 0, 0), -1)
            writer.write(frame)
        writer.release()
        return vid_path

    def test_resolve_device(self):
        self.assertEqual(resolve_device("cpu"), "cpu")
        # In CPU-only environment, auto should resolve to cpu
        self.assertEqual(resolve_device("auto"), "cpu")

    def test_invalid_input_path_raises_file_not_found(self):
        non_existent = self.temp_path / "non_existent.mp4"
        out_path = self.temp_path / "out.mp4"
        with self.assertRaises(FileNotFoundError):
            process_video(non_existent, out_path)

    def test_corrupt_or_empty_video_raises_runtime_error(self):
        corrupt_file = self.temp_path / "corrupt.mp4"
        corrupt_file.write_text("not a video")
        out_path = self.temp_path / "out.mp4"
        with self.assertRaises(RuntimeError):
            process_video(corrupt_file, out_path)

    def test_annotate_frame_output_shape_and_type(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # Sample detection: x1, y1, x2, y2, conf, cls_id (2 = car)
        detections = [(30.0, 40.0, 100.0, 120.0, 0.88, 2)]
        annotated = annotate_frame(frame, detections, frame_idx=1, current_fps=30.0)
        self.assertEqual(annotated.shape, frame.shape)
        self.assertEqual(annotated.dtype, frame.dtype)

    def test_synthetic_video_processing_with_mock_model(self):
        synthetic_video = self._create_synthetic_video("test_in.mp4", num_frames=5)
        out_path = self.temp_path / "test_out.mp4"

        # Mock YOLO model box output
        mock_box = MagicMock()
        mock_box.cls = [MagicMock(item=lambda: 2)]  # class 2 = car
        mock_box.xyxy = [MagicMock(tolist=lambda: [30.0, 40.0, 100.0, 120.0])]
        mock_box.conf = [MagicMock(item=lambda: 0.92)]
        mock_box.id = [MagicMock(item=lambda: 17)]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]

        mock_model = MagicMock()
        mock_model.track.return_value = [mock_result]

        cfg = TrackGuardConfig(confidence_threshold=0.3, iou_threshold=0.5)
        with patch("src.detector.annotate_frame", wraps=annotate_frame) as annotate:
            summary = process_video(
                input_path=synthetic_video,
                output_path=out_path,
                config=cfg,
                model=mock_model,
            )

        self.assertTrue(out_path.exists())
        self.assertGreater(out_path.stat().st_size, 0)
        self.assertEqual(summary.total_frames, 5)
        self.assertEqual(summary.vehicle_detections_count, 5)
        self.assertEqual(summary.event_candidates, [])
        self.assertGreater(summary.processing_fps, 0.0)
        self.assertEqual(summary.video_width, 320)
        self.assertEqual(summary.video_height, 240)
        self.assertEqual(mock_model.track.call_count, 5)
        self.assertEqual(mock_model.track.call_args.kwargs["tracker"], "bytetrack.yaml")
        self.assertTrue(mock_model.track.call_args.kwargs["persist"])
        self.assertEqual(annotate.call_args_list[0].args[1][0][-1], 17)

    def test_detection_only_mode_uses_model_without_tracking(self):
        synthetic_video = self._create_synthetic_video("detection_only_in.mp4")
        out_path = self.temp_path / "detection_only_out.mp4"
        mock_box = MagicMock()
        mock_box.cls = [MagicMock(item=lambda: 2)]
        mock_box.xyxy = [MagicMock(tolist=lambda: [30.0, 40.0, 100.0, 120.0])]
        mock_box.conf = [MagicMock(item=lambda: 0.92)]
        mock_box.id = None
        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_model = MagicMock()
        mock_model.return_value = [mock_result]

        summary = process_video(
            synthetic_video,
            out_path,
            model=mock_model,
            enable_tracking=False,
        )

        self.assertEqual(summary.total_frames, 5)
        self.assertEqual(summary.vehicle_detections_count, 5)
        self.assertEqual(mock_model.call_count, 5)
        mock_model.track.assert_not_called()

    def test_tracker_reinitializes_between_videos_and_persists_within_each(self):
        synthetic_video = self._create_synthetic_video("tracker_lifecycle_in.mp4")
        mock_box = MagicMock()
        mock_box.cls = [MagicMock(item=lambda: 2)]
        mock_box.xyxy = [MagicMock(tolist=lambda: [30.0, 40.0, 100.0, 120.0])]
        mock_box.conf = [MagicMock(item=lambda: 0.92)]
        mock_box.id = [MagicMock(item=lambda: 17)]
        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_model = MagicMock()
        mock_model.track.return_value = [mock_result]

        first_output = self.temp_path / "first_video_out.mp4"
        second_output = self.temp_path / "second_video_out.mp4"
        process_video(synthetic_video, first_output, model=mock_model)
        process_video(synthetic_video, second_output, model=mock_model)

        self.assertEqual(mock_model.track.call_count, 10)
        persist_values = [call.kwargs["persist"] for call in mock_model.track.call_args_list]
        self.assertEqual(persist_values, [False, True, True, True, True] * 2)

    def test_synthetic_video_emits_boundary_candidate_without_model_download(self):
        synthetic_video = self._create_synthetic_video("boundary_crossing_in.mp4")
        out_path = self.temp_path / "boundary_crossing_out.mp4"
        centers = [120.0, 140.0, 180.0, 200.0, 220.0]
        frame_results = []
        for center_x in centers:
            box = MagicMock()
            box.cls = [MagicMock(item=lambda: 2)]
            box.xyxy = [MagicMock(tolist=lambda cx=center_x: [cx - 10.0, 100.0, cx + 10.0, 120.0])]
            box.conf = [MagicMock(item=lambda: 0.91)]
            box.id = [MagicMock(item=lambda: 23)]
            result = MagicMock()
            result.boxes = [box]
            frame_results.append([result])

        mock_model = MagicMock()
        mock_model.track.side_effect = frame_results
        cfg = TrackGuardConfig(
            device="cpu",
            evidence_dir=str(self.temp_path / "evidence"),
            track_boundary=TrackBoundary(
                start_x=0.5,
                start_y=0.2,
                end_x=0.5,
                end_y=0.8,
            ),
            boundary_hysteresis_pixels=5.0,
        )

        with patch("src.evidence.shutil.which", return_value=None):
            summary = process_video(synthetic_video, out_path, config=cfg, model=mock_model)

        self.assertEqual(summary.total_frames, 5)
        self.assertTrue(out_path.is_file())
        self.assertEqual(len(summary.event_candidates), 1)
        event = summary.event_candidates[0]
        self.assertEqual(event.track_id, 23)
        self.assertEqual(event.frame_index, 3)
        self.assertEqual(event.video_time_seconds, 0.2)
        self.assertIsNone(event.timestamp)
        self.assertEqual(event.review_status.value, "pending_review")
        self.assertIn("positive_to_negative", event.notes)
        self.assertTrue(Path(event.evidence_frame_reference).is_file())
        self.assertTrue(Path(event.evidence_clip_reference).is_file())
        self.assertEqual(event.evidence_reference, event.evidence_clip_reference)
        self.assertEqual(event.evidence_audio_status, "unavailable")
        evidence_clip = cv2.VideoCapture(event.evidence_clip_reference)
        decoded_evidence_frames = 0
        while evidence_clip.read()[0]:
            decoded_evidence_frames += 1
        evidence_clip.release()
        self.assertEqual(decoded_evidence_frames, 5)
        serialized_event = summary.to_dict()["event_candidates"][0]
        self.assertEqual(serialized_event["video_time_seconds"], 0.2)
        self.assertEqual(serialized_event["review_status"], "pending_review")
        self.assertEqual(serialized_event["evidence_audio_status"], "unavailable")

    def test_configured_boundary_without_events_creates_no_evidence_files(self):
        synthetic_video = self._create_synthetic_video("no_boundary_crossing_in.mp4")
        evidence_dir = self.temp_path / "unused_evidence"
        mock_box = MagicMock()
        mock_box.cls = [MagicMock(item=lambda: 2)]
        mock_box.xyxy = [MagicMock(tolist=lambda: [30.0, 40.0, 100.0, 120.0])]
        mock_box.conf = [MagicMock(item=lambda: 0.92)]
        mock_box.id = [MagicMock(item=lambda: 17)]
        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_model = MagicMock()
        mock_model.track.return_value = [mock_result]
        cfg = TrackGuardConfig(
            device="cpu",
            evidence_dir=str(evidence_dir),
            track_boundary=TrackBoundary(start_x=0.8, start_y=0.1, end_x=0.8, end_y=0.9),
        )

        summary = process_video(
            synthetic_video,
            self.temp_path / "no_boundary_crossing_out.mp4",
            config=cfg,
            model=mock_model,
        )

        self.assertEqual(summary.event_candidates, [])
        self.assertFalse(evidence_dir.exists())

    def test_annotate_frame_includes_track_id_when_available(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        detections = [(30.0, 40.0, 100.0, 120.0, 0.88, 2, 17)]

        with patch("src.detector.cv2.putText") as put_text:
            annotate_frame(frame, detections, frame_idx=1, current_fps=30.0)

        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("car ID:17 0.88", labels)


if __name__ == "__main__":
    unittest.main()
