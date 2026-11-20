import ast
import unittest
from pathlib import Path

from src.schemas import EventType, ReviewStatus, TrackEventRecord


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
REVIEW_STATUS_LABELS = {
    ReviewStatus.PENDING_REVIEW: "Pending Review",
    ReviewStatus.UNDER_REVIEW: "Needs Further Review",
    ReviewStatus.REVIEWED_ACCEPTED: "Accepted",
    ReviewStatus.REVIEWED_REJECTED: "Rejected",
}
MAX_REVIEW_NOTE_CHARS = 1000


def _load_review_action_helpers():
    names = {
        "_sync_event_selection",
        "_set_event_review_status",
        "_event_review_status",
        "_event_review_row",
        "_event_review_details",
        "_prepare_event_note_widget",
        "_save_event_reviewer_note",
    }
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"), filename=str(APP_PATH))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {function.name for function in functions} != names:
        raise AssertionError("Event review action helpers are missing from app.py")
    namespace = {
        "ReviewStatus": ReviewStatus,
        "TrackEventRecord": TrackEventRecord,
        "REVIEW_STATUS_LABELS": REVIEW_STATUS_LABELS,
        "MAX_REVIEW_NOTE_CHARS": MAX_REVIEW_NOTE_CHARS,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(APP_PATH), "exec"), namespace)
    return namespace


def _event(event_id):
    return TrackEventRecord(
        event_id=event_id,
        event_type=EventType.TRACK_LIMIT_CANDIDATE,
        review_status=ReviewStatus.PENDING_REVIEW,
        notes="detector crossing metadata",
    )


class TestEventReviewActions(unittest.TestCase):
    def setUp(self):
        self.helpers = _load_review_action_helpers()
        self.state = {
            "event_review_run_id": "run-current",
            "event_review_decisions": {},
            "event_review_notes": {},
        }

    def test_each_action_updates_only_selected_event_and_queue_status(self):
        events = [_event("evt-one"), _event("evt-two")]
        decisions = (
            ReviewStatus.REVIEWED_ACCEPTED,
            ReviewStatus.REVIEWED_REJECTED,
            ReviewStatus.UNDER_REVIEW,
        )
        for decision in decisions:
            self.state["event_review_decisions"] = {}
            self.assertTrue(
                self.helpers["_set_event_review_status"](
                    self.state, "run-current", "evt-two", [e.event_id for e in events], decision
                )
            )
            self.assertEqual(self.state["event_review_decisions"], {"evt-two": decision})
            status = self.helpers["_event_review_status"](self.state, events[1])
            row = self.helpers["_event_review_row"](events[1], status)
            self.assertEqual(row["Review status"], REVIEW_STATUS_LABELS[decision])
            self.assertEqual(
                self.helpers["_event_review_status"](self.state, events[0]),
                ReviewStatus.PENDING_REVIEW,
            )

    def test_invalid_or_missing_selection_cannot_be_modified(self):
        set_status = self.helpers["_set_event_review_status"]
        self.assertFalse(
            set_status(
                self.state,
                "run-current",
                "evt-missing",
                ["evt-one"],
                ReviewStatus.REVIEWED_ACCEPTED,
            )
        )
        self.assertFalse(
            set_status(
                self.state,
                "stale-run",
                "evt-one",
                ["evt-one"],
                ReviewStatus.REVIEWED_REJECTED,
            )
        )
        self.assertFalse(
            set_status(
                self.state,
                "run-current",
                None,
                ["evt-one"],
                ReviewStatus.UNDER_REVIEW,
            )
        )
        self.assertEqual(self.state["event_review_decisions"], {})

    def test_notes_persist_by_event_across_reruns_and_event_switches(self):
        prepare = self.helpers["_prepare_event_note_widget"]
        save = self.helpers["_save_event_reviewer_note"]

        prepare(self.state, "run-current", "evt-one")
        self.assertEqual(self.state["event_review_note_widget"], "")
        self.state["event_review_note_widget"] = "Reviewed first candidate"
        self.assertTrue(save(
            self.state, "run-current", "evt-one", ["evt-one", "evt-two"],
            self.state["event_review_note_widget"],
        ))

        # The same owner on an ordinary rerun must not overwrite the live widget value.
        self.state["event_review_note_widget"] = "Edited note survives rerun"
        prepare(self.state, "run-current", "evt-one")
        self.assertEqual(self.state["event_review_note_widget"], "Edited note survives rerun")
        save(self.state, "run-current", "evt-one", ["evt-one", "evt-two"],
             self.state["event_review_note_widget"])

        prepare(self.state, "run-current", "evt-two")
        self.assertEqual(self.state["event_review_note_widget"], "")
        self.state["event_review_note_widget"] = "Reviewed second candidate"
        save(self.state, "run-current", "evt-two", ["evt-one", "evt-two"],
             self.state["event_review_note_widget"])
        prepare(self.state, "run-current", "evt-one")
        self.assertEqual(self.state["event_review_note_widget"], "Edited note survives rerun")
        self.assertEqual(
            self.state["event_review_notes"],
            {
                "evt-one": "Edited note survives rerun",
                "evt-two": "Reviewed second candidate",
            },
        )

    def test_notes_reject_invalid_selection_and_overlong_content(self):
        save = self.helpers["_save_event_reviewer_note"]
        self.assertFalse(save(self.state, "run-current", "missing", ["evt-one"], "note"))
        self.assertFalse(save(self.state, "old-run", "evt-one", ["evt-one"], "note"))
        self.assertFalse(save(
            self.state,
            "run-current",
            "evt-one",
            ["evt-one"],
            "x" * (MAX_REVIEW_NOTE_CHARS + 1),
        ))
        self.assertEqual(self.state["event_review_notes"], {})

    def test_run_change_clears_prior_run_session_decisions_and_notes(self):
        self.state["event_review_selected_event"] = "evt-one"
        self.state["event_review_decisions"] = {"evt-one": ReviewStatus.REVIEWED_ACCEPTED}
        self.state["event_review_notes"] = {"evt-one": "old run note"}

        self.helpers["_sync_event_selection"](self.state, "run-next", ["evt-next"])

        self.assertEqual(self.state["event_review_run_id"], "run-next")
        self.assertNotIn("event_review_selected_event", self.state)
        self.assertEqual(self.state["event_review_decisions"], {})
        self.assertEqual(self.state["event_review_notes"], {})

    def test_review_actions_leave_original_detection_metadata_unchanged(self):
        event = _event("evt-unchanged")
        initial_status = event.review_status
        initial_notes = event.notes

        self.helpers["_set_event_review_status"](
            self.state,
            "run-current",
            event.event_id,
            [event.event_id],
            ReviewStatus.REVIEWED_REJECTED,
        )
        self.helpers["_save_event_reviewer_note"](
            self.state,
            "run-current",
            event.event_id,
            [event.event_id],
            "reviewer note",
        )

        self.assertEqual(event.review_status, initial_status)
        self.assertEqual(event.notes, initial_notes)
        self.assertEqual(
            self.helpers["_event_review_status"](self.state, event),
            ReviewStatus.REVIEWED_REJECTED,
        )
        self.assertEqual(self.state["event_review_notes"][event.event_id], "reviewer note")


if __name__ == "__main__":
    unittest.main()
