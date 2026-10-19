import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock
import cv2
import numpy as np

from src.config import TrackGuardConfig
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

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]

        mock_model = MagicMock()
        mock_model.return_value = [mock_result]

        cfg = TrackGuardConfig(confidence_threshold=0.3, iou_threshold=0.5)
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
        self.assertGreater(summary.processing_fps, 0.0)
        self.assertEqual(summary.video_width, 320)
        self.assertEqual(summary.video_height, 240)


if __name__ == "__main__":
    unittest.main()
