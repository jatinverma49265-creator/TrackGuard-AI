# Project State: TrackGuard AI

## Current Status
- **Current Loop**: Loop 2 — Basic Vehicle Detection (Complete)
- **Repository State**: Clean working tree on branch `main` (uncommitted local changes pending review)
- **Target Platform**: Windows 11, Python 3.13.7, NVIDIA GeForce RTX 2050 Laptop GPU (4 GB VRAM)
- **Runtime Environment**: `.venv` with `ultralytics 8.4.182`, `torch 2.14.1+cpu`, `opencv-python 5.0.0.93`, `pydantic 2.14.1`

## Component Inventory
- `.gitignore`: Configured with standard Python ignores plus TrackGuard media/weight caches (`data/`, `models/`, `weights/`, `evidence/`, video containers).
- `requirements.txt`: Core dependencies (`pydantic>=2.6.0,<3.0.0`, `numpy>=1.26.0`, `opencv-python>=4.9.0`, `pytest>=8.0.0`, `ultralytics>=8.0.0`).
- `weights/yolov8n.pt`: Official pre-trained YOLOv8n weights (~6.25 MB) verified and loaded for COCO vehicle detection.
- `src/config.py`: Validated configuration schema with Pydantic (`TrackGuardConfig`), including device resolution, resolution bounds, and thresholds.
- `src/schemas.py`: Neutral, non-accusatory candidate observation schema (`TrackEventRecord`, `EventType`, `ReviewStatus`, `BoundingBox`).
- `src/detector.py`: Vehicle detection engine supporting local video ingestion, COCO vehicle filtering (`car`, `motorcycle`, `bus`, `truck`), bounding box annotation, telemetry HUD overlay, FPS tracking, robust error handling, and resource release.
- `tests/test_config.py`: Unit tests for configuration parameters and validation constraints (5 passed).
- `tests/test_schemas.py`: Unit tests for event schema, coordinates, and neutral review status (5 passed).
- `tests/test_detector.py`: Unit tests for input validation, corrupt file handling, frame annotation, and synthetic video processing with mock models (5 passed).
- `DECISIONS.md`: Record of architectural and technical decisions made to date.

## Test Verification Summary
- **Total Tests**: 15 passed across `unittest` and `pytest` (0 failures, 0 errors).
- **Inference Mode**: Verified on CPU (PyTorch CPU build).

## Next Planned Steps (Loop 3)
1. Design minimal multi-object tracking integration (e.g. ByteTrack or Sort tracklet association).
2. Assign consistent vehicle tracking IDs across frames.
3. Keep tracking decoupled from event decision rules.
