import unittest
from datetime import datetime, timezone
from pydantic import ValidationError

from src.schemas import (
    BoundingBox,
    EventType,
    ReviewStatus,
    TrackEventRecord,
)


class TestTrackEventSchemas(unittest.TestCase):
    def test_valid_event_record_creation(self):
        now = datetime.now(timezone.utc)
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=110.0, y2=120.0)
        record = TrackEventRecord(
            event_id="evt-001",
            event_type=EventType.TRACK_LIMIT_CANDIDATE,
            timestamp=now,
            track_id=42,
            frame_index=150,
            confidence=0.89,
            bbox=bbox,
            evidence_reference="evidence/evt-001.mp4",
        )
        self.assertEqual(record.event_id, "evt-001")
        self.assertEqual(record.event_type, EventType.TRACK_LIMIT_CANDIDATE)
        self.assertEqual(record.review_status, ReviewStatus.PENDING_REVIEW)
        self.assertEqual(record.track_id, 42)
        self.assertEqual(record.confidence, 0.89)

    def test_default_review_status_is_neutral(self):
        record = TrackEventRecord(
            event_id="evt-002",
            event_type=EventType.INCIDENT_INDICATOR_CANDIDATE,
            timestamp=datetime.now(timezone.utc),
        )
        self.assertEqual(record.review_status, ReviewStatus.PENDING_REVIEW)
        self.assertIsNone(record.track_id)
        self.assertIsNone(record.evidence_reference)

    def test_video_relative_candidate_does_not_invent_wall_clock_timestamp(self):
        record = TrackEventRecord(
            event_id="evt-video-time",
            event_type=EventType.TRACK_LIMIT_CANDIDATE,
            frame_index=12,
            video_time_seconds=0.44,
        )

        self.assertIsNone(record.timestamp)
        self.assertEqual(record.frame_index, 12)
        self.assertEqual(record.video_time_seconds, 0.44)
        self.assertEqual(record.review_status, ReviewStatus.PENDING_REVIEW)

    def test_evidence_references_and_audio_status_are_optional(self):
        record = TrackEventRecord(
            event_id="evt-evidence",
            event_type=EventType.TRACK_LIMIT_CANDIDATE,
            timestamp=None,
        )
        self.assertIsNone(record.evidence_reference)
        self.assertIsNone(record.evidence_frame_reference)
        self.assertIsNone(record.evidence_clip_reference)
        self.assertIsNone(record.evidence_audio_status)

        record.evidence_frame_reference = "evidence/frame.jpg"
        record.evidence_clip_reference = "evidence/clip.mp4"
        record.evidence_reference = record.evidence_clip_reference
        record.evidence_audio_status = "unavailable"
        self.assertEqual(record.model_dump(mode="json")["evidence_audio_status"], "unavailable")

    def test_invalid_confidence_raises_error(self):
        with self.assertRaises(ValidationError):
            TrackEventRecord(
                event_id="evt-bad-conf",
                event_type=EventType.STATIONARY_VEHICLE_CANDIDATE,
                timestamp=datetime.now(timezone.utc),
                confidence=1.2,
            )

    def test_invalid_bbox_coordinates_raise_error(self):
        # x2 must be >= x1
        with self.assertRaises(ValidationError):
            BoundingBox(x1=100.0, y1=50.0, x2=50.0, y2=100.0)

        # negative coordinate
        with self.assertRaises(ValidationError):
            BoundingBox(x1=-10.0, y1=50.0, x2=50.0, y2=100.0)

    def test_event_record_serialization_roundtrip(self):
        record = TrackEventRecord(
            event_id="evt-003",
            event_type=EventType.TRAJECTORY_DEVIATION_CANDIDATE,
            timestamp=datetime.now(timezone.utc),
            confidence=0.75,
            review_status=ReviewStatus.UNDER_REVIEW,
        )
        dumped = record.model_dump()
        reloaded = TrackEventRecord.model_validate(dumped)
        self.assertEqual(reloaded.event_id, record.event_id)
        self.assertEqual(reloaded.review_status, ReviewStatus.UNDER_REVIEW)


if __name__ == "__main__":
    unittest.main()
