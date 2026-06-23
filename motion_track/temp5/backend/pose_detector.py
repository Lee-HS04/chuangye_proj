"""
backend/pose_detector.py
------------------------
YOLO-based pose detection with:
  - T-pose / anatomical-stand validation for calibration
  - Per-joint correction cues ("move right knee to the left")
  - Smart recording state machine:
      WAITING → POSE_DETECTED → RECORDING → COMPLETE
      Any pose interruption during RECORDING → back to WAITING,
      discard partial clip, restart when pose is re-established.

YOLO model used: yolov8n-pose  (ultralytics)
Keypoint indices follow COCO-17:
  0=nose 1=left_eye 2=right_eye 3=left_ear 4=right_ear
  5=left_shoulder 6=right_shoulder 7=left_elbow 8=right_elbow
  9=left_wrist 10=right_wrist 11=left_hip 12=right_hip
  13=left_knee 14=right_knee 15=left_ankle 16=right_ankle
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    _model = YOLO("yolov8n-pose.pt")   # downloads automatically on first run
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False
    print("WARNING: ultralytics not installed. Pose detection will be simulated.")


# ─────────────────────────────────────────────
# COCO-17 keypoint indices (named for clarity)
# ─────────────────────────────────────────────
KP = {
    "nose": 0,
    "l_shoulder": 5, "r_shoulder": 6,
    "l_elbow": 7,    "r_elbow": 8,
    "l_wrist": 9,    "r_wrist": 10,
    "l_hip": 11,     "r_hip": 12,
    "l_knee": 13,    "r_knee": 14,
    "l_ankle": 15,   "r_ankle": 16,
}

CONF_THRESHOLD = 0.4   # minimum keypoint confidence to use


# ─────────────────────────────────────────────
# RECORDING STATE MACHINE
# ─────────────────────────────────────────────

class RecordState(Enum):
    WAITING        = "waiting"        # no valid pose
    POSE_DETECTED  = "pose_detected"  # pose valid, holding countdown
    RECORDING      = "recording"      # actively recording + pose valid
    COMPLETE       = "complete"       # required duration captured


@dataclass
class PoseResult:
    """Returned every frame from PoseDetector.process_frame()"""
    state:          RecordState
    corrections:    list[str]          # human-readable fix instructions
    hold_progress:  float              # 0.0–1.0, how long pose has been held
    record_seconds: float              # seconds recorded so far
    frame_annotated: Optional[np.ndarray] = None   # frame with skeleton drawn
    is_complete:    bool = False


# ─────────────────────────────────────────────
# POSE VALIDATOR
# ─────────────────────────────────────────────

class TStandValidator:
    """
    Validates the calibration / stability pose:
      - Anatomical stand: feet ~shoulder-width apart
      - Arms at sides (wrists roughly at hip level, not raised)
      - Body upright (shoulders level, hips level)
      - Both feet visible
    Returns a list of correction strings (empty = pose is good).
    """

    def validate(self, kps: dict[str, tuple[float, float, float]]) -> list[str]:
        """
        kps: {name: (x, y, conf)} in pixel coordinates, normalised 0–1.
        Returns list of correction cues. Empty list = pose is valid.
        """
        corrections = []

        # Helper: get keypoint or None if below confidence threshold
        def get(name) -> Optional[tuple[float, float]]:
            p = kps.get(name)
            if p is None or p[2] < CONF_THRESHOLD:
                return None
            return (p[0], p[1])

        l_shoulder = get("l_shoulder")
        r_shoulder = get("r_shoulder")
        l_hip      = get("l_hip")
        r_hip      = get("r_hip")
        l_knee     = get("l_knee")
        r_knee     = get("r_knee")
        l_ankle    = get("l_ankle")
        r_ankle    = get("r_ankle")
        l_wrist    = get("l_wrist")
        r_wrist    = get("r_wrist")

        # ── Both feet must be visible ──────────────────────────────────────
        if l_ankle is None:
            corrections.append("Move left foot into frame")
        if r_ankle is None:
            corrections.append("Move right foot into frame")

        # ── Shoulders level (lateral tilt < 5% of frame height) ───────────
        if l_shoulder and r_shoulder:
            tilt = abs(l_shoulder[1] - r_shoulder[1])
            if tilt > 0.05:
                side = "right" if l_shoulder[1] > r_shoulder[1] else "left"
                corrections.append(f"Level your shoulders — tilt {side} shoulder down")

        # ── Hips level ─────────────────────────────────────────────────────
        if l_hip and r_hip:
            tilt = abs(l_hip[1] - r_hip[1])
            if tilt > 0.05:
                side = "right" if l_hip[1] > r_hip[1] else "left"
                corrections.append(f"Level your hips — drop {side} hip slightly")

        # ── Feet shoulder-width apart (not too close, not too wide) ────────
        if l_ankle and r_ankle and l_shoulder and r_shoulder:
            foot_width     = abs(l_ankle[0] - r_ankle[0])
            shoulder_width = abs(l_shoulder[0] - r_shoulder[0])
            ratio = foot_width / (shoulder_width + 1e-6)
            if ratio < 0.6:
                corrections.append("Move feet further apart — shoulder-width stance")
            elif ratio > 1.6:
                corrections.append("Move feet closer together — shoulder-width stance")

        # ── Arms at sides (wrists near hip height, not raised) ─────────────
        if l_wrist and l_hip:
            wrist_above_hip = l_hip[1] - l_wrist[1]   # positive = wrist above hip
            if wrist_above_hip > 0.15:
                corrections.append("Lower your left arm — keep it at your side")
        if r_wrist and r_hip:
            wrist_above_hip = r_hip[1] - r_wrist[1]
            if wrist_above_hip > 0.15:
                corrections.append("Lower your right arm — keep it at your side")

        # ── Knees not bent excessively ──────────────────────────────────────
        if l_hip and l_knee and l_ankle:
            knee_bend = _knee_bend_deg(l_hip, l_knee, l_ankle)
            if knee_bend > 20:
                corrections.append("Straighten your left knee")
        if r_hip and r_knee and r_ankle:
            knee_bend = _knee_bend_deg(r_hip, r_knee, r_ankle)
            if knee_bend > 20:
                corrections.append("Straighten your right knee")

        # ── Knees aligned (not knocked in / out too much) ───────────────────
        if l_hip and l_knee:
            lateral_offset = l_knee[0] - l_hip[0]
            if lateral_offset > 0.08:   # knee too far right of hip
                corrections.append("Move left knee slightly to the left")
            elif lateral_offset < -0.08:
                corrections.append("Move left knee slightly to the right")
        if r_hip and r_knee:
            lateral_offset = r_knee[0] - r_hip[0]
            if lateral_offset < -0.08:
                corrections.append("Move right knee slightly to the right")
            elif lateral_offset > 0.08:
                corrections.append("Move right knee slightly to the left")

        return corrections


# ─────────────────────────────────────────────
# MAIN POSE DETECTOR
# ─────────────────────────────────────────────

class PoseDetector:
    """
    Processes video frames one at a time. Maintains recording state machine.

    Usage:
        detector = PoseDetector(required_seconds=5.0, hold_seconds=2.0)
        result = detector.process_frame(frame_bgr)
        if result.is_complete:
            frames = detector.get_recorded_frames()
    """

    def __init__(
        self,
        required_seconds: float = 5.0,   # how long to record once pose is held
        hold_seconds: float = 2.0,        # how long pose must be held before recording starts
        fps: float = 30.0,
        on_complete: Optional[Callable[[list], None]] = None,
        on_imu_zero: Optional[Callable[[], None]] = None,
    ):
        self.required_seconds = required_seconds
        self.hold_seconds     = hold_seconds
        self.fps              = fps
        self.on_complete      = on_complete
        # Called ONCE at the exact frame the pose hold threshold is crossed.
        # This is the ideal moment to snapshot IMU quaternions as zero-reference
        # because the user has been perfectly still for hold_seconds already.
        self.on_imu_zero      = on_imu_zero

        self._validator = TStandValidator()
        self._state     = RecordState.WAITING

        self._pose_hold_start:  Optional[float] = None
        self._record_start:     Optional[float] = None
        self._recorded_frames:  list[np.ndarray] = []
        self._imu_zeroed:       bool = False   # fire on_imu_zero only once per run

        self._last_corrections: list[str] = []

    # ── State machine ──────────────────────────────────────────────────────

    def process_frame(self, frame_bgr: np.ndarray) -> PoseResult:
        """Main entry point. Call once per camera frame."""
        now = time.time()

        kps, annotated = self._detect_keypoints(frame_bgr)
        corrections = self._validator.validate(kps) if kps else ["No person detected — step into frame"]

        pose_valid = len(corrections) == 0

        # ── State transitions ────────────────────────────────────────────────
        if self._state == RecordState.WAITING:
            if pose_valid:
                self._pose_hold_start = now
                self._state = RecordState.POSE_DETECTED

        elif self._state == RecordState.POSE_DETECTED:
            if not pose_valid:
                self._pose_hold_start = None
                self._state = RecordState.WAITING
            else:
                held = now - self._pose_hold_start
                if held >= self.hold_seconds:
                    # ── IMU zero snapshot fires HERE ──────────────────────────
                    # The user has been stationary in the correct pose for the
                    # full hold_seconds. This is the best possible moment to
                    # capture quaternion offsets as the neutral reference.
                    if self.on_imu_zero and not self._imu_zeroed:
                        self._imu_zeroed = True
                        try:
                            self.on_imu_zero()
                        except Exception as e:
                            print(f"on_imu_zero callback error: {e}")
                    # ── Start video recording ─────────────────────────────────
                    self._record_start    = now
                    self._recorded_frames = []
                    self._state = RecordState.RECORDING

        elif self._state == RecordState.RECORDING:
            if not pose_valid:
                # Interrupted — discard and restart
                self._recorded_frames = []
                self._record_start     = None
                self._pose_hold_start  = None
                self._state = RecordState.WAITING
            else:
                self._recorded_frames.append(frame_bgr.copy())
                elapsed = now - self._record_start
                if elapsed >= self.required_seconds:
                    self._state = RecordState.COMPLETE
                    if self.on_complete:
                        self.on_complete(self._recorded_frames)

        # ── Build result ──────────────────────────────────────────────────────
        hold_progress   = 0.0
        record_seconds  = 0.0

        if self._state == RecordState.POSE_DETECTED and self._pose_hold_start:
            hold_progress = min(1.0, (now - self._pose_hold_start) / self.hold_seconds)
        if self._state == RecordState.RECORDING and self._record_start:
            record_seconds = now - self._record_start
        if self._state == RecordState.COMPLETE:
            record_seconds = self.required_seconds
            hold_progress  = 1.0

        self._last_corrections = corrections

        return PoseResult(
            state           = self._state,
            corrections     = corrections,
            hold_progress   = hold_progress,
            record_seconds  = record_seconds,
            frame_annotated = annotated,
            is_complete     = self._state == RecordState.COMPLETE,
        )

    def reset(self):
        """Reset to WAITING state, discard any recording."""
        self._state           = RecordState.WAITING
        self._pose_hold_start = None
        self._record_start    = None
        self._recorded_frames = []
        self._imu_zeroed      = False

    def get_recorded_frames(self) -> list[np.ndarray]:
        return self._recorded_frames

    # ── YOLO inference ────────────────────────────────────────────────────────

    def _detect_keypoints(
        self, frame_bgr: np.ndarray
    ) -> tuple[dict[str, tuple[float, float, float]], np.ndarray]:
        """
        Returns:
          kps: {name: (x_norm, y_norm, conf)} — normalised 0-1 coords
          annotated: frame with skeleton drawn
        """
        if not YOLO_AVAILABLE:
            return self._simulate_keypoints(frame_bgr)

        h, w = frame_bgr.shape[:2]
        results = _model(frame_bgr, verbose=False)

        annotated = results[0].plot()   # YOLO draws its own skeleton

        if not results[0].keypoints or results[0].keypoints.xy is None:
            return {}, annotated

        # Take the first (highest-confidence) person
        kps_xy   = results[0].keypoints.xy[0].cpu().numpy()    # (17, 2) pixel coords
        kps_conf = results[0].keypoints.conf[0].cpu().numpy()  # (17,) confidence

        kps_named: dict[str, tuple[float, float, float]] = {}
        for name, idx in KP.items():
            x, y = kps_xy[idx]
            conf = float(kps_conf[idx])
            kps_named[name] = (float(x) / w, float(y) / h, conf)

        return kps_named, annotated

    def _simulate_keypoints(
        self, frame_bgr: np.ndarray
    ) -> tuple[dict[str, tuple[float, float, float]], np.ndarray]:
        """
        Fallback when ultralytics/YOLO is not installed.

        Strategy (in order of preference):
          1. OpenCV HOG person detector — checks a real person is present
             and estimates rough body proportions for keypoints.
             Validates the pose is roughly correct (full body visible,
             upright, not too close/far).
          2. If HOG also fails to detect anyone → returns empty kps
             (triggers "No person detected" correction on the frontend).

        This ensures the hold-and-record flow still requires a real
        person to be standing correctly in front of the camera, rather
        than auto-passing every frame.
        """
        h, w = frame_bgr.shape[:2]
        annotated = frame_bgr.copy()

        # ── HOG person detection ─────────────────────────────────────────────
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Resize for speed — HOG works fine at 320px wide
        scale     = min(1.0, 320 / w)
        small     = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
        boxes, _  = hog.detectMultiScale(small, winStride=(8, 8), padding=(4, 4), scale=1.05)

        cv2.putText(annotated, "[NO YOLO — HOG fallback]", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 2)

        if len(boxes) == 0:
            cv2.putText(annotated, "No person detected", (10, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 60, 255), 2)
            return {}, annotated

        # Use the largest detection (most likely the subject)
        areas = [bw * bh for (_, _, bw, bh) in boxes]
        x, y, bw, bh = boxes[int(np.argmax(areas))]

        # Scale box back to original frame coordinates
        x  = int(x  / scale);  y  = int(y  / scale)
        bw = int(bw / scale);  bh = int(bh / scale)

        # Draw bounding box
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 200, 100), 2)

        # ── Validate body proportion / fullness in frame ──────────────────────
        corrections = []

        # Box must cover at least 50% of frame height (full body visible)
        if bh < h * 0.50:
            corrections.append("Step back — full body must be visible")

        # Box must not fill more than 90% of frame (too close)
        if bh > h * 0.90:
            corrections.append("Step further from the camera")

        # Person must be roughly centred (within middle 60% of frame width)
        cx = x + bw / 2
        if cx < w * 0.20 or cx > w * 0.80:
            corrections.append("Move towards the centre of the frame")

        # Box aspect ratio: a standing person is taller than wide (ratio > 1.4)
        if bw > 0 and (bh / bw) < 1.2:
            corrections.append("Stand upright — full height must be visible")

        if corrections:
            for i, c in enumerate(corrections):
                cv2.putText(annotated, f"! {c}", (10, 52 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 255), 1)
            return {}, annotated   # return empty → validator will show corrections

        # ── Estimate keypoints from bounding box proportions ─────────────────
        # These are rough anatomical estimates, not real keypoints.
        # Sufficient to pass TStandValidator's gross checks (full body visible,
        # centred, upright). Fine-grained angle checks are skipped in HOG mode.
        xn = x / w;  yn = y / h;  wn = bw / w;  hn = bh / h

        def kp(rx, ry):
            """rx/ry are fractions within the bounding box."""
            return (xn + rx * wn, yn + ry * hn, 0.75)

        kps = {
            "nose":       kp(0.50, 0.04),
            "l_shoulder": kp(0.28, 0.18),
            "r_shoulder": kp(0.72, 0.18),
            "l_elbow":    kp(0.18, 0.35),
            "r_elbow":    kp(0.82, 0.35),
            "l_wrist":    kp(0.20, 0.52),   # arms at sides → wrists near hips
            "r_wrist":    kp(0.80, 0.52),
            "l_hip":      kp(0.38, 0.54),
            "r_hip":      kp(0.62, 0.54),
            "l_knee":     kp(0.38, 0.73),
            "r_knee":     kp(0.62, 0.73),
            "l_ankle":    kp(0.38, 0.93),
            "r_ankle":    kp(0.62, 0.93),
        }

        cv2.putText(annotated, "Person detected — check pose", (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 100), 1)
        return kps, annotated


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _knee_bend_deg(
    hip: tuple, knee: tuple, ankle: tuple
) -> float:
    """Knee flexion angle at the knee joint, in degrees. 0 = straight."""
    h = np.array(hip)
    k = np.array(knee)
    a = np.array(ankle)
    v1 = h - k
    v2 = a - k
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_a = np.dot(v1, v2) / (n1 * n2)
    return 180.0 - float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))