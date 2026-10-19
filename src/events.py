"""Lightweight geometry and state for track-boundary event candidates."""

from dataclasses import dataclass
from uuid import uuid4

from src.config import TrackBoundary
from src.schemas import BoundingBox, EventType, TrackEventRecord

Point = tuple[float, float]


def segment_intersection(a: Point, b: Point, c: Point, d: Point) -> Point | None:
    """Return the intersection of two finite segments, if one exists."""
    r = (b[0] - a[0], b[1] - a[1])
    s = (d[0] - c[0], d[1] - c[1])
    denominator = r[0] * s[1] - r[1] * s[0]
    if abs(denominator) < 1e-9:
        return None

    c_minus_a = (c[0] - a[0], c[1] - a[1])
    t = (c_minus_a[0] * s[1] - c_minus_a[1] * s[0]) / denominator
    u = (c_minus_a[0] * r[1] - c_minus_a[1] * r[0]) / denominator
    if not (-1e-9 <= t <= 1.0 + 1e-9 and -1e-9 <= u <= 1.0 + 1e-9):
        return None
    return (a[0] + t * r[0], a[1] + t * r[1])


@dataclass
class _TrackSideState:
    side: int
    position: Point


class BoundaryEventDetector:
    """Detect stable crossings of one configured finite boundary segment."""

    def __init__(
        self,
        boundary: TrackBoundary,
        frame_width: int,
        frame_height: int,
        hysteresis_pixels: float,
    ) -> None:
        x_scale = max(frame_width - 1, 1)
        y_scale = max(frame_height - 1, 1)
        self.start = (boundary.start_x * x_scale, boundary.start_y * y_scale)
        self.end = (boundary.end_x * x_scale, boundary.end_y * y_scale)
        self.hysteresis_pixels = hysteresis_pixels
        self._line_length = ((self.end[0] - self.start[0]) ** 2 + (self.end[1] - self.start[1]) ** 2) ** 0.5
        if self._line_length == 0:
            raise ValueError("Track boundary must have non-zero length at the input frame resolution")
        self._tracks: dict[int, _TrackSideState] = {}

    def _signed_distance(self, point: Point) -> float:
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return (dx * (point[1] - self.start[1]) - dy * (point[0] - self.start[0])) / self._line_length

    def update(
        self,
        track_id: int | None,
        bbox: tuple[float, float, float, float],
        frame_index: int,
        video_time_seconds: float,
        confidence: float | None = None,
    ) -> TrackEventRecord | None:
        """Update one track using its box bottom-center and return a new candidate."""
        if track_id is None:
            return None

        x1, y1, x2, y2 = bbox
        position = ((x1 + x2) / 2.0, y2)
        distance = self._signed_distance(position)
        if distance > self.hysteresis_pixels:
            side = 1
        elif distance < -self.hysteresis_pixels:
            side = -1
        else:
            # Keep the previous stable position while the point is in the deadband.
            return None

        state = self._tracks.get(track_id)
        if state is None:
            self._tracks[track_id] = _TrackSideState(side, position)
            return None

        event = None
        if side != state.side:
            crossing = segment_intersection(state.position, position, self.start, self.end)
            if crossing is not None:
                direction = "positive_to_negative" if state.side > side else "negative_to_positive"
                event = TrackEventRecord(
                    event_id=f"evt-{uuid4().hex}",
                    event_type=EventType.TRACK_LIMIT_CANDIDATE,
                    timestamp=None,
                    track_id=track_id,
                    frame_index=frame_index,
                    video_time_seconds=video_time_seconds,
                    confidence=confidence,
                    bbox=BoundingBox(
                        x1=max(0.0, x1),
                        y1=max(0.0, y1),
                        x2=max(0.0, x2),
                        y2=max(0.0, y2),
                    ),
                    notes=(
                        f"crossing_direction={direction}; "
                        f"crossing_point=({crossing[0]:.1f}, {crossing[1]:.1f})"
                    ),
                )

        # Re-arm after each stable side change, allowing a later genuine recrossing.
        self._tracks[track_id] = _TrackSideState(side, position)
        return event
