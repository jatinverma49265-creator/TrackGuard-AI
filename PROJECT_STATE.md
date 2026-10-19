# Project State: TrackGuard AI

## Current Status
- **Current Loop**: Loop 4 — Track-Boundary Event Detection (Complete)
- **Repository State**: Loop 4 changes pending review on branch `main`; baseline commit `a48ebfc`.
- **Target Platform**: Windows 11, Python 3.13.7, NVIDIA GeForce RTX 2050 Laptop GPU (4 GB VRAM)
- **Runtime Environment**: `.venv` with `ultralytics 8.4.182`, `torch 2.14.1+cpu`, `opencv-python 5.0.0.93`, `pydantic 2.14.1`

## Component Inventory
- `.gitignore`: Configured with standard Python ignores plus TrackGuard media/weight caches (`data/`, `models/`, `weights/`, `evidence/`, video containers).
- `requirements.txt`: Core dependencies (`pydantic>=2.6.0,<3.0.0`, `numpy>=1.26.0`, `opencv-python>=4.9.0`, `pytest>=8.0.0`, `ultralytics>=8.0.0`).
- `weights/yolov8n.pt`: Official pre-trained YOLOv8n weights (~6.25 MB) verified and loaded for COCO vehicle detection.
- `src/config.py`: Validated configuration schema with Pydantic (`TrackGuardConfig`), including device resolution, thresholds, and an optional normalized finite track-boundary segment.
- `src/schemas.py`: Neutral candidate event schema (`TrackEventRecord`, `EventType`, `ReviewStatus`, `BoundingBox`), including frame index and video-relative time without requiring a recording timestamp.
- `src/events.py`: CPU-light finite-segment geometry and per-track stable-side crossing detection with hysteresis.
- `src/detector.py`: Video detection and ByteTrack pipeline with optional `--no-track`, vehicle annotation, telemetry, and pending-review boundary candidate collection when configured.
- `tests/test_config.py`: Unit tests for configuration parameters and validation constraints (7 passed).
- `tests/test_schemas.py`: Unit tests for event schema, coordinates, and neutral review status (6 passed).
- `tests/test_detector.py`: Unit tests for input validation, annotation, tracking/detection-only modes, and synthetic-video boundary event processing.
- `tests/test_events.py`: Offline geometry, direction, finite-segment, hysteresis, recrossing, missing-ID, and multi-vehicle tests.
- `DECISIONS.md`: Record of architectural and technical decisions made to date.

## Test Verification Summary
- **Total Tests**: 29 passed across `pytest` (0 failures, 0 errors).
- **Inference Mode**: Verified on CPU (PyTorch CPU build).

## Next Planned Steps
- Loop 5 scope to be selected after review of Loop 4.
