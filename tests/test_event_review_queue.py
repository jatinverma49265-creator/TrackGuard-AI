import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pydantic import ValidationError

from src.schemas import BoundingBox, EventType, ReviewStatus, TrackEventRecord


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
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


def _load_review_helpers():
    names = {
        "_current_run_events",
        "_validated_event_records",
        "_event_review_row",
        "_event_review_rows",
        "_filter_event_review_rows",
        "_event_review_details",
        "_sync_event_selection",
        "_resolve_run_evidence_path",
        "_video_has_decodable_frame",
        "_render_event_evidence",
    }
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"), filename=str(APP_PATH))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {function.name for function in functions} != names:
        raise AssertionError("Event review helpers are missing from app.py")
    namespace = {
        "Path": Path,
        "cv2": _FakeCV2(),
        "st": _FakeStreamlit(),
        "TrackEventRecord": TrackEventRecord,
        "ReviewStatus": ReviewStatus,
        "ValidationError": ValidationError,
        "REVIEW_STATUS_LABELS": REVIEW_STATUS_LABELS,
        "REVIEW_FILTERS": REVIEW_FILTERS,
        "REVIEW_STATUS_LABELS": REVIEW_STATUS_LABELS,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(APP_PATH), "exec"), namespace)
    return namespace


class _FakeStreamlit:
    def __init__(self):
        self.images = []
        self.videos = []
        self.messages = []

    def markdown(self, *args, **kwargs):
        pass

    def info(self, message):
        self.messages.append(message)

    def warning(self, message):
        self.messages.append(message)

    def image(self, image, **kwargs):
        self.images.append((image, kwargs))

    def video(self, path, **kwargs):
        self.videos.append((path, kwargs))


class _FakeCapture:
    def __init__(self, opened=True, frame=True):
        self.opened = opened
        self.frame = SimpleNamespace(size=1) if frame else None
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        return self.frame is not None, self.frame

    def release(self):
        self.released = True


class _FakeCV2:
    IMREAD_COLOR = 1
    error = RuntimeError

    def __init__(self):
        self.image = SimpleNamespace(size=1)
        self.capture = _FakeCapture()

    def imread(self, path, mode):
        return self.image

    def VideoCapture(self, path):
        return self.capture


def _event(event_id, status=ReviewStatus.PENDING_REVIEW, **values):
    return TrackEventRecord(
        event_id=event_id,
        event_type=EventType.TRACK_LIMIT_CANDIDATE,
        review_status=status,
        **values,
    )


class TestEventReviewQueue(unittest.TestCase):
    def setUp(self):
        self.helpers = _load_review_helpers()

    def test_events_from_latest_successful_run_are_formatted_for_display(self):
        current_event = _event("current-event", track_id=8, frame_index=42, confidence=0.91)
        with tempfile.TemporaryDirectory() as root:
            run_directory = Path(root) / "current-run"
            run_directory.mkdir()
            output = run_directory / "annotated.mp4"
            output.touch()
            session_state = {
                "latest_run_directory": str(run_directory),
                "latest_run_output": str(output),
                "latest_run_summary": SimpleNamespace(
                    output_path=str(output), event_candidates=[current_event]
                ),
            }

            records = self.helpers["_current_run_events"](session_state)
            rows, malformed = self.helpers["_event_review_rows"](records)

        self.assertEqual(malformed, 0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Event ID"], "current-event")
        self.assertEqual(rows[0]["Frame"], 42)
        self.assertEqual(rows[0]["Track ID"], 8)
        self.assertEqual(rows[0]["Confidence"], "0.910")
        self.assertEqual(rows[0]["Review status"], "Pending Review")

    def test_empty_event_list_normalizes_to_empty_queue(self):
        rows, malformed = self.helpers["_event_review_rows"]([])

        self.assertEqual(rows, [])
        self.assertEqual(malformed, 0)

    def test_each_review_status_filter_selects_only_matching_events(self):
        events = [
            _event("pending", ReviewStatus.PENDING_REVIEW),
            _event("accepted", ReviewStatus.REVIEWED_ACCEPTED),
            _event("rejected", ReviewStatus.REVIEWED_REJECTED),
            _event("further", ReviewStatus.UNDER_REVIEW),
        ]
        rows, malformed = self.helpers["_event_review_rows"](events)
        filter_rows = self.helpers["_filter_event_review_rows"]

        self.assertEqual(malformed, 0)
        self.assertEqual([row["Event ID"] for row in filter_rows(rows, "All")],
                         ["pending", "accepted", "rejected", "further"])
        self.assertEqual([row["Event ID"] for row in filter_rows(rows, "Pending Review")], ["pending"])
        self.assertEqual([row["Event ID"] for row in filter_rows(rows, "Accepted")], ["accepted"])
        self.assertEqual([row["Event ID"] for row in filter_rows(rows, "Rejected")], ["rejected"])
        self.assertEqual([row["Event ID"] for row in filter_rows(rows, "Needs Further Review")],
                         ["further"])

    def test_missing_optional_fields_are_displayed_as_unavailable(self):
        rows, malformed = self.helpers["_event_review_rows"](
            [{"event_id": "minimal", "event_type": "track_limit_candidate"}]
        )

        self.assertEqual(malformed, 0)
        self.assertEqual(rows[0]["Timestamp"], "—")
        self.assertEqual(rows[0]["Video time (s)"], "—")
        self.assertEqual(rows[0]["Frame"], "—")
        self.assertEqual(rows[0]["Track ID"], "—")
        self.assertEqual(rows[0]["Confidence"], "—")

    def test_malformed_records_are_skipped_and_counted(self):
        rows, malformed = self.helpers["_event_review_rows"](
            [
                {"event_id": "valid", "event_type": "track_limit_candidate"},
                {"event_type": "not-a-real-event"},
                None,
            ]
        )

        self.assertEqual(malformed, 2)
        self.assertEqual([row["Event ID"] for row in rows], ["valid"])

    def test_previous_run_events_are_not_mixed_with_current_run(self):
        current_event = _event("current")
        previous_event = _event("previous")
        with tempfile.TemporaryDirectory() as root:
            run_directory = Path(root) / "current-run"
            run_directory.mkdir()
            output = run_directory / "annotated.mp4"
            output.touch()
            session_state = {
                "latest_run_directory": str(run_directory),
                "latest_run_output": str(output),
                "latest_run_summary": SimpleNamespace(
                    output_path=str(output), event_candidates=[current_event]
                ),
                "previous_run_events": [previous_event],
            }

            records = self.helpers["_current_run_events"](session_state)
            rows, _ = self.helpers["_event_review_rows"](records)
            session_state["latest_run_summary"] = SimpleNamespace(
                output_path=str(Path(root) / "previous-run" / "annotated.mp4"),
                event_candidates=[previous_event],
            )
            mismatched_records = self.helpers["_current_run_events"](session_state)

        self.assertEqual([row["Event ID"] for row in rows], ["current"])
        self.assertIsNone(mismatched_records)

    def test_selected_event_details_show_only_available_schema_values(self):
        event = _event(
            "selected",
            frame_index=25,
            track_id=4,
            confidence=0.87,
            bbox=BoundingBox(x1=1, y1=2, x2=30, y2=40),
        )
        details = self.helpers["_event_review_details"](event)

        self.assertEqual(details["Event ID"], "selected")
        self.assertEqual(details["Frame"], "25")
        self.assertEqual(details["Track ID"], "4")
        self.assertEqual(details["Confidence"], "0.870")
        self.assertIn("x1=1.0", details["Bounding box"])
        self.assertNotIn("Timestamp", details)
        self.assertNotIn("Video time", details)

    def test_selection_resets_for_new_run_and_invalid_event_but_survives_rerun(self):
        state = {
            "event_review_run_id": "run-one",
            "event_review_selected_event": "evt-a",
        }
        sync = self.helpers["_sync_event_selection"]

        sync(state, "run-one", ["evt-a", "evt-b"])
        self.assertEqual(state["event_review_selected_event"], "evt-a")
        sync(state, "run-two", ["evt-a", "evt-c"])
        self.assertNotIn("event_review_selected_event", state)
        state["event_review_selected_event"] = "evt-c"
        sync(state, "run-two", ["evt-a"])
        self.assertNotIn("event_review_selected_event", state)
        sync(state, None, [])
        self.assertNotIn("event_review_selected_event", state)

    def test_valid_evidence_image_and_clip_are_previewed(self):
        with tempfile.TemporaryDirectory() as root:
            run_directory = Path(root) / "run"
            evidence_directory = run_directory / "evidence"
            evidence_directory.mkdir(parents=True)
            image_path = evidence_directory / "frame.jpg"
            clip_path = evidence_directory / "clip.mp4"
            image_path.touch()
            clip_path.touch()
            event = _event(
                "with-evidence",
                evidence_frame_reference=str(image_path),
                evidence_clip_reference=str(clip_path),
            )

            self.helpers["_render_event_evidence"](event, run_directory)

        streamlit = self.helpers["st"]
        cv2 = self.helpers["cv2"]
        self.assertEqual(len(streamlit.images), 1)
        self.assertEqual(len(streamlit.videos), 1)
        self.assertTrue(cv2.capture.released)

    def test_missing_and_invalid_evidence_do_not_attempt_previews(self):
        with tempfile.TemporaryDirectory() as root:
            run_directory = Path(root) / "run"
            (run_directory / "evidence").mkdir(parents=True)
            missing = _event(
                "missing",
                evidence_frame_reference=str(run_directory / "evidence" / "missing.jpg"),
                evidence_clip_reference=str(run_directory / "evidence" / "missing.mp4"),
            )
            self.helpers["_render_event_evidence"](missing, run_directory)

            image_path = run_directory / "evidence" / "broken.jpg"
            clip_path = run_directory / "evidence" / "broken.mp4"
            image_path.touch()
            clip_path.touch()
            self.helpers["cv2"].image = None
            self.helpers["cv2"].capture = _FakeCapture(opened=False)
            invalid = _event(
                "invalid",
                evidence_frame_reference=str(image_path),
                evidence_clip_reference=str(clip_path),
            )
            self.helpers["_render_event_evidence"](invalid, run_directory)

        streamlit = self.helpers["st"]
        self.assertEqual(streamlit.images, [])
        self.assertEqual(streamlit.videos, [])
        self.assertGreaterEqual(len(streamlit.messages), 4)

    def test_paths_outside_or_from_another_run_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            current_run = Path(root) / "current"
            other_run = Path(root) / "other"
            current_evidence = current_run / "evidence"
            other_evidence = other_run / "evidence"
            current_evidence.mkdir(parents=True)
            other_evidence.mkdir(parents=True)
            other_file = other_evidence / "event.jpg"
            other_file.touch()
            escaped_path = current_evidence / ".." / ".." / "outside.jpg"
            escaped_path.resolve().touch()

            resolve = self.helpers["_resolve_run_evidence_path"]
            self.assertIsNone(resolve(str(other_file), current_run))
            self.assertIsNone(resolve(str(escaped_path), current_run))


if __name__ == "__main__":
    unittest.main()
