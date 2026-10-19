# Architectural & Technical Decisions: TrackGuard AI

## Decision 1: Strict Pydantic Validation Boundary
- **Context**: Configuration parameters and incoming/outgoing event payloads require strict typing and bounds enforcement to prevent downstream pipeline crashes.
- **Decision**: Use Pydantic v2 exclusively for `TrackGuardConfig` and `TrackEventRecord`. Avoid over-engineering; use standard Python primitives where Pydantic does not add validation value.
- **Consequence**: Configuration errors (e.g. invalid thresholds, negative coordinates) fail fast before video inference or streaming begins.

## Decision 2: Neutral Candidate Event Framing (Evidentiary Impartiality)
- **Context**: Automated computer vision systems should not make final authoritative rulings or pronounce infractions without human race official oversight.
- **Decision**: All system observations are categorized as *candidates* (`track_limit_candidate`, `incident_indicator_candidate`, etc.) with review status defaulting to `pending_review`.
- **Consequence**: The system functions strictly as a decision-support and evidence-gathering instrument, never asserting definitive "violations".

## Decision 3: Hardware Constraint Optimization (NVIDIA RTX 2050 Laptop / 4 GB VRAM)
- **Context**: The deployment target is a laptop with 4 GB of VRAM running Windows and Python 3.13. Large monolithic models will trigger CUDA out-of-memory errors.
- **Decision**: Enforce FP16 half-precision toggle by default, single-frame batch sizes (`batch_size=1`), default input resolution `640x640`, and headless OpenCV.
- **Consequence**: Keeps peak VRAM usage comfortably beneath the 4 GB ceiling while enabling real-time FPS on trackside feeds.

## Decision 4: Test Suite Framework Decoupling
- **Context**: Dependencies cannot be installed in the initial setup loop, but comprehensive test suites must be provided immediately.
- **Decision**: Implement tests using standard library `unittest` conventions. This allows execution via `python -m unittest` or `pytest` interchangeably once dependencies are installed.
- **Consequence**: Zero external test runner locking; tests remain clean and standard.

## Decision 5: Vehicle-Class Whitelist Filtering
- **Context**: Standard COCO models detect 80 object classes (people, animals, sports items). Trackside motorsport monitoring only cares about motor vehicles.
- **Decision**: Restrict detections to COCO IDs `2` (`car`), `3` (`motorcycle`), `5` (`bus`), and `7` (`truck`). Ignore all other background detections at the inference level (`classes=[2, 3, 5, 7]`).
- **Consequence**: Drastically reduces false positives (spectators, marshals, trackside equipment) and saves downstream tracking computation.

## Decision 6: Synthetic Video Unit Testing
- **Context**: Unit tests must execute quickly and reliably in CI and developer environments without downloading multi-gigabyte video datasets or heavy weights.
- **Decision**: Use OpenCV in unit tests to dynamically generate lightweight 5-frame synthetic videos and mock model predictions.
- **Consequence**: 100% offline, deterministic testing of the video ingest-detect-annotate-export pipeline in under 3 seconds.
