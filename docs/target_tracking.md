# UAV Target Detection and Tracking

## Detection + tracking path

The final LOCK implementation uses `YoloByteTracker.track_with_detections()` so one Ultralytics `model.track()` call provides both:

- raw YOLO detections captured before ByteTrack filtering, and
- tracked boxes with persistent ByteTrack IDs.

This avoids running a second YOLO inference just to obtain untracked detections.

## Why both outputs are used

The combined competition logic treats detection and tracking as related but distinct signals:

- raw YOLO detections are available for lock/timer validation,
- ByteTrack IDs are used for persistent target selection and manual focus/target publishing.

This separation prevents tracker behavior from hiding the detector information needed by the lock logic while still preserving target identity for tracking.

## Target state manager

`TargetStateManager` tracks the selected target ID and exposes four main states:

- `Active`
- `Temporary Lost`
- `Lost`
- `Reset`

The public example configuration uses frame-count thresholds for temporary loss, lost state, and reset.

## Tracking outputs

For the selected target, `src/metrics.py` calculates:

- center pixel coordinates,
- normalized horizontal error,
- normalized vertical error,
- normalized target velocity,
- bounding-box area ratio.

These values are suitable for downstream guidance/control consumers through the ROS target-data interface.

## Lock logic

The final integrated build uses:

- default lock duration: **4.0 seconds**,
- temporary YOLO-miss tolerance: **200 ms**,
- hit-area checks,
- target-size validation,
- server-time-based mission timestamps.

The lock engine also records latency and reliability metrics such as YOLO+tracking time, queue wait, frame age, evaluation update time, and mode/runtime statistics.

## Manual target controls

In LOCK mode:

- `B` selects the previous candidate,
- `N` selects the next candidate,
- `SPACE` toggles/focuses the selected target,
- `-` resets the lock timer.

The final application can maintain a short manual target continuity window and uses recent target motion information when needed by the manual focus logic.
