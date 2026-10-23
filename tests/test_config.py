import unittest
from pydantic import ValidationError

from src.config import TrackBoundary, TrackGuardConfig


class TestTrackGuardConfig(unittest.TestCase):
    def test_default_config_is_valid(self):
        cfg = TrackGuardConfig()
        self.assertEqual(cfg.device, "auto")
        self.assertTrue(cfg.enable_fp16)
        self.assertEqual(cfg.batch_size, 1)
        self.assertEqual(cfg.input_width, 640)
        self.assertEqual(cfg.input_height, 640)
        self.assertEqual(cfg.confidence_threshold, 0.35)
        self.assertEqual(cfg.iou_threshold, 0.45)
        self.assertEqual(cfg.evidence_dir, "evidence")
        self.assertEqual(cfg.max_event_buffer_seconds, 10.0)
        self.assertEqual(cfg.evidence_pre_seconds, 3.0)
        self.assertEqual(cfg.evidence_post_seconds, 3.0)
        self.assertEqual(cfg.track_limits_margin_pixels, 0.0)
        self.assertIsNone(cfg.track_boundary)
        self.assertEqual(cfg.boundary_hysteresis_pixels, 5.0)

    def test_custom_valid_config(self):
        cfg = TrackGuardConfig(
            device="cuda",
            enable_fp16=True,
            confidence_threshold=0.5,
            iou_threshold=0.6,
            evidence_dir="custom_evidence",
            max_event_buffer_seconds=15.0,
            track_limits_margin_pixels=5.0,
        )
        self.assertEqual(cfg.device, "cuda")
        self.assertEqual(cfg.confidence_threshold, 0.5)
        self.assertEqual(cfg.evidence_dir, "custom_evidence")

    def test_invalid_device_raises_validation_error(self):
        with self.assertRaises(ValidationError):
            TrackGuardConfig(device="tpu")

    def test_invalid_confidence_threshold_raises_validation_error(self):
        with self.assertRaises(ValidationError):
            TrackGuardConfig(confidence_threshold=1.5)

        with self.assertRaises(ValidationError):
            TrackGuardConfig(confidence_threshold=-0.1)

    def test_invalid_buffer_seconds_raises_validation_error(self):
        with self.assertRaises(ValidationError):
            TrackGuardConfig(max_event_buffer_seconds=0.0)

    def test_track_boundary_requires_normalized_distinct_endpoints(self):
        boundary = TrackBoundary(start_x=0.25, start_y=0.2, end_x=0.75, end_y=0.8)
        self.assertEqual(boundary.start_x, 0.25)

        with self.assertRaises(ValidationError):
            TrackBoundary(start_x=0.5, start_y=0.5, end_x=0.5, end_y=0.5)

        with self.assertRaises(ValidationError):
            TrackBoundary(start_x=-0.1, start_y=0.5, end_x=0.5, end_y=0.8)

    def test_boundary_hysteresis_margin_must_be_non_negative(self):
        with self.assertRaises(ValidationError):
            TrackGuardConfig(boundary_hysteresis_pixels=-1.0)

    def test_evidence_window_must_fit_inside_maximum_buffer(self):
        cfg = TrackGuardConfig(evidence_pre_seconds=4.0, evidence_post_seconds=4.0)
        self.assertEqual(cfg.evidence_pre_seconds + cfg.evidence_post_seconds, 8.0)
        with self.assertRaises(ValidationError):
            TrackGuardConfig(evidence_pre_seconds=6.0, evidence_post_seconds=5.0)


if __name__ == "__main__":
    unittest.main()
