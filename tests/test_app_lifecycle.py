import ast
import io
import shutil
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _load_app_functions(names, namespace):
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"), filename=str(APP_PATH))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {function.name for function in functions} != set(names):
        raise AssertionError(f"Could not find requested app functions: {names}")
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, str(APP_PATH), "exec"), namespace)
    return namespace


class _Config:
    def __init__(self, **values):
        self.values = values


class _Upload(io.BytesIO):
    name = "race.mp4"


class TestDashboardLifecycle(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="trackguard_app_lifecycle_")
        self.root = Path(self.temp_dir.name)
        (self.root / "app.py").touch()
        self.pipeline_calls = []
        namespace = {
            "__file__": str(self.root / "app.py"),
            "Path": Path,
            "shutil": shutil,
            "tempfile": tempfile,
            "uuid": uuid,
            "TrackBoundary": object,
            "TrackGuardConfig": _Config,
            "UPLOAD_CHUNK_BYTES": 1024 * 1024,
            "MAX_UPLOAD_MB": 500,
            "_persist_detector_events": lambda run_directory, events: None,
        }
        self.functions = _load_app_functions(
            {"_process_uploaded_video", "_existing_evidence_paths"}, namespace
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _pipeline_module(self, fail=False):
        calls = self.pipeline_calls

        def process_video(input_path, output_path, config):
            input_path = Path(input_path)
            output_path = Path(output_path)
            calls.append((input_path, output_path, config))
            self.assertTrue(input_path.is_file())
            output_path.write_bytes(b"annotated video")
            evidence_dir = Path(config.values["evidence_dir"])
            evidence_dir.mkdir(parents=True)
            frame_path = evidence_dir / "event.jpg"
            clip_path = evidence_dir / "event.mp4"
            frame_path.write_bytes(b"evidence frame")
            clip_path.write_bytes(b"evidence clip")
            if fail:
                raise RuntimeError("synthetic pipeline failure")
            event = types.SimpleNamespace(
                evidence_frame_reference=str(frame_path),
                evidence_clip_reference=str(clip_path),
            )
            return types.SimpleNamespace(event_candidates=[event])

        package = types.ModuleType("src")
        package.__path__ = []
        detector = types.ModuleType("src.detector")
        detector.process_video = process_video
        return {"src": package, "src.detector": detector}

    def _process(self, fail=False):
        with patch.dict(sys.modules, self._pipeline_module(fail)):
            return self.functions["_process_uploaded_video"](
                _Upload(b"video bytes"), 0.25, "auto", object()
            )

    def test_success_keeps_unique_run_output_and_evidence_and_removes_input(self):
        first = self._process()
        second = self._process()

        for summary, run_dir, output_path, evidence_paths in (first, second):
            self.assertTrue(run_dir.is_dir())
            self.assertTrue(output_path.is_file())
            self.assertEqual(len(evidence_paths), 2)
            self.assertTrue(all(Path(path).is_file() for path in evidence_paths))
        self.assertNotEqual(first[1], second[1])
        self.assertTrue(all(not call[0].exists() for call in self.pipeline_calls))

    def test_failure_removes_only_its_run_and_input_but_preserves_prior_success(self):
        successful = self._process()
        successful_run, successful_output, successful_evidence = (
            successful[1], successful[2], successful[3]
        )

        with self.assertRaisesRegex(RuntimeError, "synthetic pipeline failure"):
            self._process(fail=True)

        failed_input, failed_output, _ = self.pipeline_calls[-1]
        self.assertFalse(failed_input.exists())
        self.assertFalse(failed_output.parent.exists())
        self.assertTrue(successful_run.is_dir())
        self.assertTrue(successful_output.is_file())
        self.assertTrue(all(Path(path).is_file() for path in successful_evidence))

    def test_missing_evidence_references_are_filtered_without_error(self):
        present = self.root / "present.jpg"
        present.touch()
        missing = self.root / "missing.mp4"

        self.assertEqual(
            self.functions["_existing_evidence_paths"]([str(present), str(missing)]),
            [present],
        )
        self.assertEqual(self.functions["_existing_evidence_paths"]([]), [])

    def test_upload_change_clears_result_state_but_same_upload_rerun_preserves_it(self):
        session_state = {
            "analysis_video_widget": object(),
            "analysis_video_upload": object(),
            "latest_run_summary": object(),
            "latest_run_directory": "old-run",
            "latest_run_output": "old-output",
            "latest_run_evidence": ["old-evidence"],
            "analysis_video_generation": 4,
            "boundary_timestamp": 5.0,
        }
        session = types.SimpleNamespace(session_state=session_state)

        def clear_boundary():
            session_state.pop("track_boundary", None)

        namespace = {"st": session, "_clear_boundary_state": clear_boundary}
        callbacks = _load_app_functions(
            {"_clear_latest_run_state", "_remember_video_upload"}, namespace
        )
        changed_upload = session_state["analysis_video_widget"]
        callbacks["_remember_video_upload"]()
        self.assertTrue(
            all(
                key not in session_state
                for key in (
                    "latest_run_summary",
                    "latest_run_directory",
                    "latest_run_output",
                    "latest_run_evidence",
                )
            )
        )
        self.assertIs(session_state["analysis_video_upload"], changed_upload)
        self.assertEqual(session_state["analysis_video_generation"], 5)

        session_state.update(
            latest_run_summary=object(),
            latest_run_directory="newer-run",
            latest_run_output="newer-output",
            latest_run_evidence=["newer-evidence"],
        )
        callbacks["_remember_video_upload"]()
        self.assertEqual(session_state["latest_run_directory"], "newer-run")
        self.assertEqual(session_state["latest_run_output"], "newer-output")

    def test_failed_rerun_leaves_no_stale_latest_result(self):
        session_state = {
            "latest_run_summary": object(),
            "latest_run_directory": "previous-run",
            "latest_run_output": "previous-output",
            "latest_run_evidence": ["previous-evidence"],
        }
        callbacks = _load_app_functions(
            {"_clear_latest_run_state"},
            {"st": types.SimpleNamespace(session_state=session_state)},
        )

        callbacks["_clear_latest_run_state"]()
        with self.assertRaisesRegex(RuntimeError, "synthetic pipeline failure"):
            self._process(fail=True)
        callbacks["_clear_latest_run_state"]()

        self.assertFalse(
            any(
                key in session_state
                for key in (
                    "latest_run_summary",
                    "latest_run_directory",
                    "latest_run_output",
                    "latest_run_evidence",
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
