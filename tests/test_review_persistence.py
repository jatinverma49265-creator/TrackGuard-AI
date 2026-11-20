import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from src.schemas import (
    EventReviewData,
    EventReviewEntry,
    EventType,
    ReviewStatus,
    TrackEventRecord,
)


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
EVENT_RECORDS_FILE = "detector_events.json"
REVIEW_DATA_FILE = "review_data.json"


def _load_persistence_helpers():
    names = {
        "_run_json_path",
        "_atomic_write_run_json",
        "_persist_detector_events",
        "_validated_event_records",
        "_load_detector_events",
        "_available_review_runs",
        "_load_review_data",
        "_save_review_data",
        "_updated_review_data",
    }
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"), filename=str(APP_PATH))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {function.name for function in functions} != names:
        raise AssertionError("Persistence helpers are missing from app.py")
    namespace = {
        "Path": Path,
        "tempfile": tempfile,
        "json": json,
        "os": os,
        "EventReviewData": EventReviewData,
        "EventReviewEntry": EventReviewEntry,
        "ReviewStatus": ReviewStatus,
        "TrackEventRecord": TrackEventRecord,
        "ValidationError": ValidationError,
        "ValidationError": ValidationError,
        "EVENT_RECORDS_FILE": EVENT_RECORDS_FILE,
        "REVIEW_DATA_FILE": REVIEW_DATA_FILE,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(APP_PATH), "exec"), namespace)
    return namespace


def _event(event_id, status=ReviewStatus.PENDING_REVIEW, note="detector metadata"):
    return TrackEventRecord(
        event_id=event_id,
        event_type=EventType.TRACK_LIMIT_CANDIDATE,
        review_status=status,
        notes=note,
        confidence=0.83,
    )


class TestReviewPersistence(unittest.TestCase):
    def setUp(self):
        self.helpers = _load_persistence_helpers()
        self.temp_dir = tempfile.TemporaryDirectory(prefix="trackguard_review_persistence_")
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _run(self, run_id):
        run_directory = self.root / run_id
        run_directory.mkdir()
        (run_directory / "annotated_video.mp4").write_bytes(b"video")
        return run_directory

    def _discoverable_run(self, run_id):
        run = self._run(run_id)
        self.helpers["_persist_detector_events"](run, [_event(f"evt-{run_id}")])
        return run

    def test_saves_each_review_status(self):
        statuses = (
            ReviewStatus.PENDING_REVIEW,
            ReviewStatus.UNDER_REVIEW,
            ReviewStatus.REVIEWED_ACCEPTED,
            ReviewStatus.REVIEWED_REJECTED,
        )
        run = self._run("run-statuses")
        reviews = {
            f"evt-{index}": EventReviewEntry(status=status)
            for index, status in enumerate(statuses)
        }
        data = EventReviewData(run_id=run.name, reviews=reviews)

        self.helpers["_save_review_data"](run, data, run.name)
        loaded, warning = self.helpers["_load_review_data"](run, run.name, list(reviews))

        self.assertIsNone(warning)
        self.assertEqual({entry.status for entry in loaded.reviews.values()}, set(statuses))

    def test_saves_and_reloads_reviewer_notes_after_simulated_restart(self):
        run = self._run("run-notes")
        data = EventReviewData(
            run_id=run.name,
            reviews={"evt-1": EventReviewEntry(status=ReviewStatus.REVIEWED_ACCEPTED,
                                                notes="Reviewed the captured frame.")},
        )
        self.helpers["_save_review_data"](run, data, run.name)

        fresh_process_helpers = _load_persistence_helpers()
        loaded, warning = fresh_process_helpers["_load_review_data"](run, run.name, ["evt-1"])

        self.assertIsNone(warning)
        self.assertEqual(loaded.reviews["evt-1"].notes, "Reviewed the captured frame.")
        self.assertEqual(loaded.reviews["evt-1"].status, ReviewStatus.REVIEWED_ACCEPTED)

    def test_review_data_is_isolated_by_run_id(self):
        first = self._run("run-one")
        second = self._run("run-two")
        for run, status in ((first, ReviewStatus.REVIEWED_ACCEPTED),
                            (second, ReviewStatus.REVIEWED_REJECTED)):
            data = EventReviewData(
                run_id=run.name,
                reviews={"same-event-id": EventReviewEntry(status=status)},
            )
            self.helpers["_save_review_data"](run, data, run.name)

        first_data, _ = self.helpers["_load_review_data"](first, first.name, ["same-event-id"])
        second_data, _ = self.helpers["_load_review_data"](second, second.name, ["same-event-id"])
        self.assertEqual(first_data.reviews["same-event-id"].status, ReviewStatus.REVIEWED_ACCEPTED)
        self.assertEqual(second_data.reviews["same-event-id"].status, ReviewStatus.REVIEWED_REJECTED)

    def test_missing_review_file_returns_empty_run_scoped_data(self):
        run = self._run("run-missing")

        data, warning = self.helpers["_load_review_data"](run, run.name, ["evt-1"])

        self.assertIsNone(warning)
        self.assertEqual(data.run_id, run.name)
        self.assertEqual(data.reviews, {})

    def test_malformed_or_corrupted_json_is_ignored_safely(self):
        run = self._run("run-corrupt")
        corrupted = run / REVIEW_DATA_FILE
        corrupted.write_text("{not valid json", encoding="utf-8")
        original_content = corrupted.read_bytes()

        data, warning = self.helpers["_load_review_data"](run, run.name, ["evt-1"])

        self.assertIsNotNone(warning)
        self.assertEqual(data.reviews, {})
        self.assertEqual(data.run_id, run.name)
        self.assertEqual(corrupted.read_bytes(), original_content)

    def test_subsequent_save_does_not_destroy_corrupted_review_source(self):
        run = self._run("run-corrupt-save")
        corrupted = run / REVIEW_DATA_FILE
        corrupted.write_bytes(b"{broken source data")
        original_content = corrupted.read_bytes()
        empty_data, warning = self.helpers["_load_review_data"](run, run.name, ["evt-1"])
        updated = self.helpers["_updated_review_data"](
            empty_data,
            run.name,
            ["evt-1"],
            "evt-1",
            ReviewStatus.PENDING_REVIEW,
            status=ReviewStatus.REVIEWED_ACCEPTED,
        )

        self.assertIsNotNone(warning)
        with self.assertRaisesRegex(ValueError, "was preserved"):
            self.helpers["_save_review_data"](run, updated, run.name)
        self.assertEqual(corrupted.read_bytes(), original_content)

    def test_atomic_write_failure_keeps_previous_file_and_cleans_temp_file(self):
        run = self._run("run-atomic")
        target = run / REVIEW_DATA_FILE
        target.write_text('{"old": true}', encoding="utf-8")

        with patch.object(os, "replace", side_effect=OSError("replace failure")):
            with self.assertRaisesRegex(OSError, "replace failure"):
                self.helpers["_atomic_write_run_json"](run, REVIEW_DATA_FILE, {"new": True})

        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"old": True})
        self.assertEqual(list(run.glob(f".{REVIEW_DATA_FILE}.*.tmp")), [])

    def test_detector_event_snapshot_is_separate_and_unchanged_by_review_updates(self):
        run = self._run("run-events")
        event = _event("evt-1")
        persist_events = self.helpers["_persist_detector_events"]
        persist_events(run, [event])
        original = (run / EVENT_RECORDS_FILE).read_text(encoding="utf-8")
        with self.assertRaises(FileExistsError):
            persist_events(run, [_event("replacement")])
        self.assertEqual((run / EVENT_RECORDS_FILE).read_text(encoding="utf-8"), original)

        review = EventReviewData(
            run_id=run.name,
            reviews={"evt-1": EventReviewEntry(status=ReviewStatus.REVIEWED_REJECTED)},
        )
        self.helpers["_save_review_data"](run, review, run.name)
        loaded_events, warning = self.helpers["_load_detector_events"](run, run.name)

        self.assertIsNone(warning)
        self.assertEqual((run / EVENT_RECORDS_FILE).read_text(encoding="utf-8"), original)
        self.assertEqual(loaded_events[0].review_status, ReviewStatus.PENDING_REVIEW)
        self.assertEqual(loaded_events[0].notes, "detector metadata")
        self.assertNotEqual(run / EVENT_RECORDS_FILE, run / REVIEW_DATA_FILE)

    def test_updating_one_event_preserves_other_existing_decisions(self):
        run = self._run("run-update")
        existing = EventReviewData(
            run_id=run.name,
            reviews={
                "evt-one": EventReviewEntry(
                    status=ReviewStatus.REVIEWED_ACCEPTED, notes="keep this note"
                )
            },
        )
        updated = self.helpers["_updated_review_data"](
            existing,
            run.name,
            ["evt-one", "evt-two"],
            "evt-two",
            ReviewStatus.PENDING_REVIEW,
            status=ReviewStatus.UNDER_REVIEW,
        )
        self.assertIsNotNone(updated)
        self.helpers["_save_review_data"](run, updated, run.name)
        loaded, _ = self.helpers["_load_review_data"](run, run.name, ["evt-one", "evt-two"])

        self.assertEqual(loaded.reviews["evt-one"].status, ReviewStatus.REVIEWED_ACCEPTED)
        self.assertEqual(loaded.reviews["evt-one"].notes, "keep this note")
        self.assertEqual(loaded.reviews["evt-two"].status, ReviewStatus.UNDER_REVIEW)

    def test_stale_revision_is_rejected_without_losing_newer_decision(self):
        run = self._run("run-revision")
        initial = EventReviewData(
            run_id=run.name,
            reviews={"evt-one": EventReviewEntry(status=ReviewStatus.PENDING_REVIEW)},
        )
        self.helpers["_save_review_data"](run, initial, run.name)
        stale_base, _ = self.helpers["_load_review_data"](
            run, run.name, ["evt-one", "evt-two"]
        )
        newer_update = self.helpers["_updated_review_data"](
            stale_base,
            run.name,
            ["evt-one", "evt-two"],
            "evt-one",
            ReviewStatus.PENDING_REVIEW,
            status=ReviewStatus.REVIEWED_ACCEPTED,
        )
        stale_update = self.helpers["_updated_review_data"](
            stale_base,
            run.name,
            ["evt-one", "evt-two"],
            "evt-two",
            ReviewStatus.PENDING_REVIEW,
            status=ReviewStatus.REVIEWED_REJECTED,
        )
        self.helpers["_save_review_data"](run, newer_update, run.name)

        with self.assertRaisesRegex(RuntimeError, "changed since it was loaded"):
            self.helpers["_save_review_data"](run, stale_update, run.name)
        loaded, _ = self.helpers["_load_review_data"](
            run, run.name, ["evt-one", "evt-two"]
        )
        self.assertEqual(loaded.reviews["evt-one"].status, ReviewStatus.REVIEWED_ACCEPTED)
        self.assertNotIn("evt-two", loaded.reviews)

    def test_saved_runs_are_discoverable_for_reopening(self):
        run = self._discoverable_run("run-reopen")

        available = self.helpers["_available_review_runs"](self.root)

        self.assertEqual(available, [run.resolve()])

    def test_run_discovery_returns_empty_when_root_is_inaccessible(self):
        root = self.root / "blocked-root"
        root.mkdir()
        with patch.object(Path, "iterdir", side_effect=PermissionError("denied")):
            available = self.helpers["_available_review_runs"](root)

        self.assertEqual(available, [])

    def test_run_disappearing_during_discovery_is_skipped(self):
        valid = self._discoverable_run("valid-run")
        disappearing = self._discoverable_run("vanishing-run")
        original_resolve = Path.resolve

        def resolve(path, *args, **kwargs):
            if path.name == disappearing.name:
                raise FileNotFoundError("run directory disappeared")
            return original_resolve(path, *args, **kwargs)

        with patch.object(Path, "resolve", resolve):
            available = self.helpers["_available_review_runs"](self.root)

        self.assertEqual(available, [valid.resolve()])

    def test_inaccessible_run_is_skipped_while_valid_runs_remain_available(self):
        valid = self._discoverable_run("valid-run")
        inaccessible = self._discoverable_run("inaccessible-run")
        original_resolve = Path.resolve

        def resolve(path, *args, **kwargs):
            if path.name == inaccessible.name:
                raise PermissionError("run directory unavailable")
            return original_resolve(path, *args, **kwargs)

        with patch.object(Path, "resolve", resolve):
            available = self.helpers["_available_review_runs"](self.root)

        self.assertEqual(available, [valid.resolve()])

    def test_run_file_path_cannot_escape_its_directory(self):
        run = self._run("run-path")

        with self.assertRaises(ValueError):
            self.helpers["_run_json_path"](run, "..\\outside.json")


if __name__ == "__main__":
    unittest.main()
