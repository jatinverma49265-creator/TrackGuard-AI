import unittest

from src.config import TrackBoundary
from src.events import BoundaryEventDetector, segment_intersection
from src.schemas import EventType, ReviewStatus


class TestBoundaryEventDetector(unittest.TestCase):
    def setUp(self):
        self.boundary = TrackBoundary(
            start_x=0.5,
            start_y=0.2,
            end_x=0.5,
            end_y=0.8,
        )

    def _detector(self, margin=5.0, boundary=None):
        return BoundaryEventDetector(
            boundary or self.boundary,
            frame_width=101,
            frame_height=101,
            hysteresis_pixels=margin,
        )

    @staticmethod
    def _bbox(center_x, bottom_y=50.0):
        return (center_x - 1.0, bottom_y - 10.0, center_x + 1.0, bottom_y)

    @staticmethod
    def _update(detector, track_id, center_x, frame_index, bottom_y=50.0):
        return detector.update(
            track_id=track_id,
            bbox=TestBoundaryEventDetector._bbox(center_x, bottom_y),
            frame_index=frame_index,
            video_time_seconds=(frame_index - 1) / 10.0,
            confidence=0.9,
        )

    def test_finite_segment_geometry(self):
        crossing = segment_intersection((30.0, 50.0), (70.0, 50.0), (50.0, 20.0), (50.0, 80.0))
        self.assertEqual(crossing, (50.0, 50.0))
        self.assertIsNone(segment_intersection((30.0, 10.0), (70.0, 10.0), (50.0, 20.0), (50.0, 80.0)))
        self.assertIsNone(segment_intersection((30.0, 20.0), (40.0, 20.0), (50.0, 20.0), (50.0, 80.0)))

    def test_crossings_in_both_directions_create_neutral_candidates(self):
        positive_to_negative = self._detector()
        self.assertIsNone(self._update(positive_to_negative, 1, 30.0, 1))
        forward = self._update(positive_to_negative, 1, 70.0, 2)

        negative_to_positive = self._detector()
        self.assertIsNone(self._update(negative_to_positive, 2, 70.0, 1))
        reverse = self._update(negative_to_positive, 2, 30.0, 2)

        self.assertEqual(forward.event_type, EventType.TRACK_LIMIT_CANDIDATE)
        self.assertEqual(forward.track_id, 1)
        self.assertEqual(forward.review_status, ReviewStatus.PENDING_REVIEW)
        self.assertIn("positive_to_negative", forward.notes)
        self.assertIn("negative_to_positive", reverse.notes)
        self.assertIsNone(forward.timestamp)
        self.assertEqual(forward.frame_index, 2)
        self.assertEqual(forward.video_time_seconds, 0.1)

    def test_same_side_motion_and_crossing_outside_finite_segment_do_not_emit(self):
        detector = self._detector()
        self.assertIsNone(self._update(detector, 1, 30.0, 1))
        self.assertIsNone(self._update(detector, 1, 40.0, 2))

        outside_segment = self._detector()
        self.assertIsNone(self._update(outside_segment, 3, 30.0, 1, bottom_y=10.0))
        self.assertIsNone(self._update(outside_segment, 3, 70.0, 2, bottom_y=10.0))

    def test_hysteresis_suppresses_jitter_duplicates(self):
        detector = self._detector(margin=5.0)
        self.assertIsNone(self._update(detector, 1, 40.0, 1))
        self.assertIsNone(self._update(detector, 1, 53.0, 2))
        self.assertIsNone(self._update(detector, 1, 47.0, 3))
        self.assertIsNone(self._update(detector, 1, 44.0, 4))
        first_crossing = self._update(detector, 1, 56.0, 5)
        self.assertIsNotNone(first_crossing)
        self.assertIsNone(self._update(detector, 1, 53.0, 6))
        self.assertIsNone(self._update(detector, 1, 56.0, 7))

    def test_later_genuine_recrossing_emits_another_candidate(self):
        detector = self._detector()
        self._update(detector, 1, 30.0, 1)
        first = self._update(detector, 1, 70.0, 2)
        second = self._update(detector, 1, 30.0, 3)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertNotEqual(first.event_id, second.event_id)
        self.assertIn("positive_to_negative", first.notes)
        self.assertIn("negative_to_positive", second.notes)

    def test_missing_track_id_is_ignored(self):
        detector = self._detector()
        self.assertIsNone(self._update(detector, None, 30.0, 1))
        self.assertIsNone(self._update(detector, 1, 30.0, 2))
        event = self._update(detector, 1, 70.0, 3)
        self.assertIsNotNone(event)
        self.assertEqual(event.track_id, 1)

    def test_multiple_vehicles_keep_independent_crossing_state(self):
        detector = self._detector()
        self._update(detector, 1, 30.0, 1)
        self._update(detector, 2, 70.0, 1)

        first = self._update(detector, 1, 70.0, 2)
        second = self._update(detector, 2, 30.0, 2)
        self.assertEqual(first.track_id, 1)
        self.assertEqual(second.track_id, 2)
        self.assertIn("positive_to_negative", first.notes)
        self.assertIn("negative_to_positive", second.notes)


if __name__ == "__main__":
    unittest.main()
