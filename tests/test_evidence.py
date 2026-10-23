import tempfile
import unittest
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from src.evidence import BoundedFrameBuffer, EvidenceCapture
from src.schemas import EventType, TrackEventRecord


def _event(event_id="evt-test", frame_index=1, video_time_seconds=0.0):
    return TrackEventRecord(
        event_id=event_id,
        event_type=EventType.TRACK_LIMIT_CANDIDATE,
        frame_index=frame_index,
        video_time_seconds=video_time_seconds,
        track_id=7,
    )


class TestEvidenceCapture(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source = self.root / "sample.mp4"
        self.source.write_bytes(b"synthetic source placeholder")
        self.frame_size = (64, 48)

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def _frame(index):
        return np.full((48, 64, 3), index * 20, dtype=np.uint8)

    def test_frame_buffer_is_bounded_and_jpeg_compressed(self):
        buffer = BoundedFrameBuffer(max_frames=2)
        for index in range(1, 4):
            buffer.append(index, self._frame(index))

        frames = buffer.snapshot()
        self.assertEqual([frame.frame_index for frame in frames], [2, 3])
        self.assertTrue(all(isinstance(frame.jpeg, bytes) for frame in frames))
        self.assertTrue(all(cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8), 1) is not None for frame in frames))

    def test_event_folder_uses_unique_path_when_event_folder_already_exists(self):
        capture = EvidenceCapture(self.root / "evidence", self.source, 10.0, self.frame_size)
        first = capture._new_event_folder(_event())
        first.rmdir()
        first.mkdir()
        second = capture._new_event_folder(_event())
        self.assertNotEqual(first, second)
        self.assertTrue(first.is_dir())
        self.assertTrue(second.is_dir())
        capture.abort()

    def test_clip_uses_available_start_frames_and_finishes_at_video_end(self):
        capture = EvidenceCapture(
            self.root / "evidence", self.source, 10.0, self.frame_size,
            pre_seconds=3.0, post_seconds=3.0,
        )
        event = _event(frame_index=1, video_time_seconds=0.0)
        with patch("src.evidence.shutil.which", return_value=None):
            for index in range(1, 6):
                capture.push_frame(index, self._frame(index), [event] if index == 1 else [])
            capture.finish()

        self.assertTrue(Path(event.evidence_frame_reference).is_file())
        self.assertTrue(Path(event.evidence_clip_reference).is_file())
        cap = cv2.VideoCapture(event.evidence_clip_reference)
        decoded = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            self.assertEqual(frame.shape[:2], (48, 64))
            decoded += 1
        cap.release()
        self.assertEqual(decoded, 5)
        self.assertEqual(event.evidence_audio_status, "unavailable")
        self.assertEqual(event.review_status.value, "pending_review")

    def test_video_start_clip_has_no_invented_pre_event_frames(self):
        capture = EvidenceCapture(
            self.root / "evidence", self.source, 10.0, self.frame_size,
            pre_seconds=3.0, post_seconds=0.0,
        )
        event = _event(frame_index=1)
        with patch("src.evidence.shutil.which", return_value=None):
            capture.push_frame(1, self._frame(1), [event])
            capture.finish()
        cap = cv2.VideoCapture(event.evidence_clip_reference)
        ok, frame = cap.read()
        self.assertTrue(ok)
        self.assertIsNotNone(frame)
        self.assertFalse(cap.read()[0])
        cap.release()

    def test_middle_clip_uses_bounded_pre_event_window_and_requested_post_window(self):
        capture = EvidenceCapture(
            self.root / "evidence", self.source, 10.0, self.frame_size,
            pre_seconds=0.2, post_seconds=0.2,
        )
        event = _event(frame_index=5, video_time_seconds=0.4)
        with patch("src.evidence.shutil.which", return_value=None):
            for index in range(1, 8):
                capture.push_frame(index, self._frame(index), [event] if index == 5 else [])
            capture.finish()

        cap = cv2.VideoCapture(event.evidence_clip_reference)
        decoded = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded.append(int(frame[0, 0, 0]))
        cap.release()
        self.assertEqual(len(decoded), 5)  # frames 3-7: two before, event, two after
        self.assertAlmostEqual(decoded[0], 60, delta=10)
        self.assertAlmostEqual(decoded[-1], 140, delta=10)

    def test_muxed_video_only_output_is_not_reported_as_audio_preserved(self):
        capture = EvidenceCapture(
            self.root / "evidence", self.source, 10.0, self.frame_size,
            pre_seconds=0.0, post_seconds=0.0,
        )
        capture.ffmpeg_path = "ffmpeg"
        capture.ffprobe_path = "ffprobe"
        event = _event()

        def fake_ffmpeg(command, **kwargs):
            if command[0] == "ffmpeg":
                video_only_path = Path(command[command.index("-i") + 1])
                Path(command[-1]).write_bytes(video_only_path.read_bytes())
                return SimpleNamespace(returncode=0, stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch.object(capture, "_probe_audio", return_value=True), patch(
            "src.evidence.subprocess.run", side_effect=fake_ffmpeg
        ) as run:
            capture.push_frame(1, self._frame(1), [event])
            capture.finish()

        self.assertEqual(event.evidence_audio_status, "unavailable")
        self.assertTrue(Path(event.evidence_clip_reference).is_file())
        command = run.call_args_list[0].args[0]
        self.assertIn("1:a:0?", command)
        self.assertIn(str(self.source), command)
        self.assertFalse((Path(event.evidence_clip_reference).parent / "event_video_only.mp4").exists())
        self.assertEqual(run.call_count, 2)

    def test_output_audio_verification_confirms_reported_audio_stream(self):
        capture = EvidenceCapture(self.root / "evidence", self.source, 10.0, self.frame_size)
        capture.ffprobe_path = "ffprobe"
        clip_path = self.root / "output.mp4"
        clip_path.write_bytes(b"mock output inspected by ffprobe")
        with patch(
            "src.evidence.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout="audio\n", stderr=""),
        ) as run:
            self.assertTrue(capture._verify_output_audio(clip_path))
        self.assertIn("-select_streams", run.call_args.args[0])
        self.assertIn("a", run.call_args.args[0])

    def test_ffmpeg_fallback_verifies_audio_when_ffprobe_is_unavailable(self):
        capture = EvidenceCapture(self.root / "evidence", self.source, 10.0, self.frame_size)
        capture.ffprobe_path = None
        capture.ffmpeg_path = "ffmpeg"
        clip_path = self.root / "output.mp4"
        with patch(
            "src.evidence.subprocess.run",
            return_value=SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="Input #0, mov,mp4: Stream #0:1: Audio: aac, 48000 Hz",
            ),
        ):
            self.assertTrue(capture._verify_output_audio(clip_path))

    def test_audio_verification_failure_reports_unavailable(self):
        capture = EvidenceCapture(
            self.root / "evidence", self.source, 10.0, self.frame_size,
            pre_seconds=0.0, post_seconds=0.0,
        )
        capture.ffmpeg_path = "ffmpeg"
        capture.ffprobe_path = "ffprobe"
        event = _event()

        def fake_tools(command, **kwargs):
            if command[0] == "ffmpeg":
                video_only_path = Path(command[command.index("-i") + 1])
                Path(command[-1]).write_bytes(video_only_path.read_bytes())
                return SimpleNamespace(returncode=0, stderr="")
            raise subprocess.TimeoutExpired(command, timeout=15)

        with patch("src.evidence.subprocess.run", side_effect=fake_tools):
            with patch.object(capture, "_probe_audio", return_value=True):
                capture.push_frame(1, self._frame(1), [event])
                capture.finish()

        self.assertEqual(event.evidence_audio_status, "unavailable")
        self.assertTrue(Path(event.evidence_clip_reference).is_file())

    def test_evidence_failure_cleans_partial_files_and_raises(self):
        capture = EvidenceCapture(self.root / "evidence", self.source, 10.0, self.frame_size)
        event = _event()
        with patch("src.evidence.cv2.imwrite", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "Could not write evidence frame"):
                capture.push_frame(1, self._frame(1), [event])
        self.assertFalse((self.root / "evidence").exists() and list((self.root / "evidence").iterdir()))
        self.assertIsNone(event.evidence_frame_reference)
        self.assertIsNone(event.evidence_clip_reference)


if __name__ == "__main__":
    unittest.main()
