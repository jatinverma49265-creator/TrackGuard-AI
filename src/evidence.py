"""Bounded, local evidence capture for video event candidates."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
from typing import Optional

import cv2
import numpy as np

from src.schemas import TrackEventRecord


@dataclass(frozen=True)
class EncodedFrame:
    frame_index: int
    jpeg: bytes


class BoundedFrameBuffer:
    """Retain only a fixed number of JPEG-compressed frames."""

    def __init__(self, max_frames: int, jpeg_quality: int = 85) -> None:
        if max_frames < 0:
            raise ValueError("max_frames must be non-negative")
        self._frames: deque[EncodedFrame] = deque(maxlen=max_frames)
        self.jpeg_quality = jpeg_quality

    def append(self, frame_index: int, frame: np.ndarray) -> None:
        ok, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            raise RuntimeError(f"Could not JPEG-encode frame {frame_index} for evidence")
        self._frames.append(EncodedFrame(frame_index, encoded.tobytes()))

    def snapshot(self) -> list[EncodedFrame]:
        return list(self._frames)

    def __len__(self) -> int:
        return len(self._frames)


@dataclass
class _ActiveClip:
    event: TrackEventRecord
    folder: Path
    frame_path: Path
    clip_path: Path
    temporary_video_path: Path
    writer: cv2.VideoWriter
    trigger_frame_index: int
    start_frame_index: int
    frame_count: int
    remaining_post_frames: int


class EvidenceCapture:
    """Capture per-event annotated evidence with bounded pre-event memory."""

    def __init__(
        self,
        evidence_dir: str | Path,
        source_path: str | Path,
        fps: float,
        frame_size: tuple[int, int],
        pre_seconds: float = 3.0,
        post_seconds: float = 3.0,
    ) -> None:
        self.evidence_dir = Path(evidence_dir)
        self.source_path = Path(source_path)
        self.fps = fps
        self.frame_size = frame_size
        self.pre_frame_count = int(np.floor(pre_seconds * fps))
        self.post_frame_count = int(np.floor(post_seconds * fps))
        self.buffer = BoundedFrameBuffer(self.pre_frame_count)
        self.ffmpeg_path = shutil.which("ffmpeg")
        self.ffprobe_path = shutil.which("ffprobe")
        self._active: list[_ActiveClip] = []
        self._completed: list[_ActiveClip] = []
        self._folders: set[Path] = set()
        self._finished = False

    def _new_event_folder(self, event: TrackEventRecord) -> Path:
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", event.event_id)
        base_name = f"{self.source_path.stem}_{safe_id}"
        for suffix in range(10000):
            name = base_name if suffix == 0 else f"{base_name}_{suffix}"
            folder = self.evidence_dir / name
            try:
                folder.mkdir(exist_ok=False)
            except FileExistsError:
                continue
            self._folders.add(folder)
            return folder
        raise RuntimeError(f"Could not reserve a unique evidence folder for {event.event_id}")

    @staticmethod
    def _decode(encoded: EncodedFrame) -> np.ndarray:
        frame = cv2.imdecode(np.frombuffer(encoded.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Could not decode buffered evidence frame {encoded.frame_index}")
        return frame

    def _start_clip(
        self, event: TrackEventRecord, frame_index: int, annotated_frame: np.ndarray
    ) -> _ActiveClip:
        folder = self._new_event_folder(event)
        frame_path = folder / "event_frame.jpg"
        clip_path = folder / "event_clip.mp4"
        temporary_video_path = folder / "event_video_only.mp4"
        writer: Optional[cv2.VideoWriter] = None
        try:
            if not cv2.imwrite(str(frame_path), annotated_frame):
                raise RuntimeError(f"Could not write evidence frame: {frame_path}")
            writer = cv2.VideoWriter(
                str(temporary_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                self.frame_size,
            )
            if not writer.isOpened():
                writer.release()
                raise RuntimeError(f"Could not open evidence clip writer: {temporary_video_path}")

            previous = self.buffer.snapshot()
            start_frame = previous[0].frame_index if previous else frame_index
            clip = _ActiveClip(
                event=event,
                folder=folder,
                frame_path=frame_path,
                clip_path=clip_path,
                temporary_video_path=temporary_video_path,
                writer=writer,
                trigger_frame_index=frame_index,
                start_frame_index=start_frame,
                frame_count=0,
                remaining_post_frames=self.post_frame_count,
            )
            for buffered_frame in previous:
                clip.writer.write(self._decode(buffered_frame))
                clip.frame_count += 1
            clip.writer.write(annotated_frame)
            clip.frame_count += 1
            return clip
        except Exception:
            if writer is not None:
                writer.release()
            shutil.rmtree(folder, ignore_errors=True)
            self._folders.discard(folder)
            raise

    def push_frame(
        self,
        frame_index: int,
        annotated_frame: np.ndarray,
        new_events: list[TrackEventRecord],
    ) -> None:
        """Buffer a frame and feed it to any event clips currently in progress."""
        if self._finished:
            raise RuntimeError("Evidence capture is already finished")
        try:
            for event in new_events:
                self._active.append(self._start_clip(event, frame_index, annotated_frame))

            still_active: list[_ActiveClip] = []
            for clip in self._active:
                if clip.trigger_frame_index != frame_index:
                    clip.writer.write(annotated_frame)
                    clip.frame_count += 1
                    clip.remaining_post_frames -= 1
                if clip.remaining_post_frames <= 0:
                    clip.writer.release()
                    self._completed.append(clip)
                else:
                    still_active.append(clip)
            self._active = still_active
            self.buffer.append(frame_index, annotated_frame)
        except Exception as exc:
            self.abort()
            if isinstance(exc, RuntimeError) and "evidence" in str(exc).lower():
                raise
            raise RuntimeError(f"Evidence capture failed at frame {frame_index}: {exc}") from exc

    def _probe_audio(self) -> bool:
        assert self.ffmpeg_path is not None
        result = subprocess.run(
            [self.ffmpeg_path, "-hide_banner", "-i", str(self.source_path)],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        diagnostic = result.stderr or ""
        if "Input #0" not in diagnostic:
            raise RuntimeError(f"FFmpeg could not inspect source audio: {diagnostic[-1000:]}")
        return re.search(r"Stream #0:\d+.*Audio:", diagnostic) is not None

    def _verify_output_audio(self, clip_path: Path) -> bool:
        """Return whether the completed clip can be confirmed to contain audio."""
        try:
            if self.ffprobe_path is not None:
                result = subprocess.run(
                    [
                        self.ffprobe_path,
                        "-v",
                        "error",
                        "-select_streams",
                        "a",
                        "-show_entries",
                        "stream=codec_type",
                        "-of",
                        "default=noprint_wrappers=1:nokey=1",
                        str(clip_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                return result.returncode == 0 and any(
                    line.strip().lower() == "audio" for line in (result.stdout or "").splitlines()
                )

            if self.ffmpeg_path is None:
                return False
            result = subprocess.run(
                [self.ffmpeg_path, "-hide_banner", "-i", str(clip_path)],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            diagnostic = result.stderr or ""
            return (
                "Input #0" in diagnostic
                and re.search(r"Stream #0:\d+.*Audio:", diagnostic) is not None
            )
        except Exception:
            return False

    def _package_clip(self, clip: _ActiveClip) -> str:
        if not clip.temporary_video_path.is_file() or clip.temporary_video_path.stat().st_size == 0:
            raise RuntimeError(f"Evidence clip is empty for event {clip.event.event_id}")

        if self.ffmpeg_path is None:
            clip.temporary_video_path.replace(clip.clip_path)
            return "unavailable"

        has_audio = self._probe_audio()
        if not has_audio:
            clip.temporary_video_path.replace(clip.clip_path)
            return "source_has_no_audio"

        start_seconds = (clip.start_frame_index - 1) / self.fps
        duration_seconds = clip.frame_count / self.fps
        result = subprocess.run(
            [
                self.ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(clip.temporary_video_path),
                "-ss",
                f"{start_seconds:.6f}",
                "-i",
                str(self.source_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-t",
                f"{duration_seconds:.6f}",
                str(clip.clip_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if result.returncode != 0 or not clip.clip_path.is_file() or clip.clip_path.stat().st_size == 0:
            raise RuntimeError(
                f"FFmpeg could not preserve audio for event {clip.event.event_id}: "
                f"{(result.stderr or '').strip()[-1000:]}"
            )
        clip.temporary_video_path.unlink(missing_ok=True)
        return "preserved" if self._verify_output_audio(clip.clip_path) else "unavailable"

    def _finalize_clip(self, clip: _ActiveClip) -> None:
        clip.writer.release()
        try:
            audio_status = self._package_clip(clip)
            image = cv2.imread(str(clip.frame_path), cv2.IMREAD_COLOR)
            if image is None or (image.shape[1], image.shape[0]) != self.frame_size:
                raise RuntimeError(f"Evidence frame is missing or unreadable for event {clip.event.event_id}")
            video = cv2.VideoCapture(str(clip.clip_path))
            decoded_frames = 0
            if video.isOpened():
                while True:
                    ok, frame = video.read()
                    if not ok:
                        break
                    if frame is None:
                        break
                    decoded_frames += 1
            video.release()
            if decoded_frames != clip.frame_count:
                raise RuntimeError(
                    f"Evidence clip for event {clip.event.event_id} decoded {decoded_frames} "
                    f"of {clip.frame_count} expected frames"
                )
            clip.event.evidence_frame_reference = str(clip.frame_path)
            clip.event.evidence_clip_reference = str(clip.clip_path)
            # Keep the original generic evidence path useful to older consumers.
            clip.event.evidence_reference = str(clip.clip_path)
            clip.event.evidence_audio_status = audio_status
        except Exception as exc:
            raise RuntimeError(f"Evidence finalization failed for event {clip.event.event_id}: {exc}") from exc

    def finish(self) -> None:
        """Close clips at their post-event bound or at end of video, then package."""
        if self._finished:
            return
        try:
            for clip in self._active:
                clip.writer.release()
                self._completed.append(clip)
            self._active.clear()
            for clip in self._completed:
                self._finalize_clip(clip)
            self._finished = True
        except Exception:
            self.abort()
            raise

    def abort(self) -> None:
        """Close active writers and remove this run's incomplete evidence folders."""
        for clip in self._active + self._completed:
            try:
                clip.writer.release()
            except Exception:
                pass
        for folder in self._folders:
            shutil.rmtree(folder, ignore_errors=True)
        self._active.clear()
        self._completed.clear()
        self._folders.clear()
        self._finished = True
