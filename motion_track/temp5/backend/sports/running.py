"""
backend/sports/running.py
--------------------------
Running-specific biomechanics computed from lower-body IMU skeleton.

Metrics:
  stride_symmetry_pct      — L/R stride length balance (%)
  cadence_spm               — steps per minute
  gct_asymmetry_ms          — ground contact time difference L vs R (ms)
  vertical_oscillation_cm   — peak-to-trough pelvis Y displacement per stride (cm)
  running_smoothness        — inverse normalised jerk (%)
  fatigue_gait_drift        — linear trend in coordination metrics over run
  knee_asymmetry_deg        — live L/R knee angle difference

All computed incrementally; call update() per skeleton frame.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
STRIDE_WINDOW  = 20    # strides to average for symmetry
CADENCE_WINDOW = 10    # seconds rolling window for cadence
JERK_WINDOW    = 60    # frames for jerk/smoothness
FATIGUE_WINDOW = 500   # frames for drift trend


class RunningEngine:
    """
    Feed joint position frames from the skeleton engine.
    Call get_metrics() to retrieve current values.
    """

    def __init__(self, fps: float = 50):
        self.fps = fps
        self.dt  = 1.0 / fps
        self.reset()

    def reset(self):
        # Pelvis vertical history for oscillation and stride detection
        self._pelvis_y: deque      = deque(maxlen=int(self.fps * 10))
        self._pelvis_y_times: deque = deque(maxlen=int(self.fps * 10))

        # Ankle Y for ground contact detection
        self._ankle_l_y: deque = deque(maxlen=int(self.fps * 5))
        self._ankle_r_y: deque = deque(maxlen=int(self.fps * 5))

        # Stride events
        self._l_contacts: list[float] = []    # timestamps of left foot strikes
        self._r_contacts: list[float] = []    # timestamps of right foot strikes
        self._l_in_contact = False
        self._r_in_contact = False
        self._l_contact_start: float | None = None
        self._r_contact_start: float | None = None
        self._l_contact_durations: deque = deque(maxlen=STRIDE_WINDOW)
        self._r_contact_durations: deque = deque(maxlen=STRIDE_WINDOW)

        # Stride lengths (estimated from timing)
        self._l_stride_times: deque = deque(maxlen=STRIDE_WINDOW)
        self._r_stride_times: deque = deque(maxlen=STRIDE_WINDOW)

        # Jerk history (pelvis)
        self._pelvis_positions: deque = deque(maxlen=JERK_WINDOW)
        self._jerk_history: deque = deque(maxlen=FATIGUE_WINDOW)

        # Knee angles for asymmetry
        self._knee_l_history: deque = deque(maxlen=FATIGUE_WINDOW)
        self._knee_r_history: deque = deque(maxlen=FATIGUE_WINDOW)

        # Fatigue tracking
        self._asym_history: deque = deque(maxlen=FATIGUE_WINDOW)

        self._frame = 0
        self._start_time = time.time()

    # ─────────────────────────────────────────────
    # UPDATE
    # ─────────────────────────────────────────────

    def update(self, joints: dict[str, list]) -> dict:
        """
        Call once per skeleton frame.
        joints: {name: [x, y, z]} from SkeletonEngine.get_joints()
        """
        self._frame += 1
        now = time.time()

        j = {k: np.array(v) for k, v in joints.items()}

        pelvis   = j.get("pelvis")
        ankle_l  = j.get("ankle_l")
        ankle_r  = j.get("ankle_r")
        hip_l    = j.get("hip_l")
        hip_r    = j.get("hip_r")
        knee_l   = j.get("knee_l")
        knee_r   = j.get("knee_r")

        # ── Pelvis tracking ─────────────────────────────────────────────────
        if pelvis is not None:
            self._pelvis_y.append(float(pelvis[1]))
            self._pelvis_y_times.append(now)
            self._pelvis_positions.append(pelvis.copy())

        # ── Ankle contact detection ──────────────────────────────────────────
        # A foot is "on the ground" when its Y is below a dynamic threshold
        # (lowest 20th percentile of ankle Y in the window = ground level)
        if ankle_l is not None:
            self._ankle_l_y.append((float(ankle_l[1]), now))
            self._detect_contact("l", ankle_l[1], now)

        if ankle_r is not None:
            self._ankle_r_y.append((float(ankle_r[1]), now))
            self._detect_contact("r", ankle_r[1], now)

        # ── Knee angles ──────────────────────────────────────────────────────
        if all(x is not None for x in [hip_l, knee_l, ankle_l]):
            kl = _joint_angle(hip_l, knee_l, ankle_l)
            self._knee_l_history.append(kl)
        if all(x is not None for x in [hip_r, knee_r, ankle_r]):
            kr = _joint_angle(hip_r, knee_r, ankle_r)
            self._knee_r_history.append(kr)

        # ── Asymmetry tracking ───────────────────────────────────────────────
        if self._knee_l_history and self._knee_r_history:
            asym = abs(self._knee_l_history[-1] - self._knee_r_history[-1])
            self._asym_history.append(asym)

        # ── Jerk ─────────────────────────────────────────────────────────────
        if len(self._pelvis_positions) >= 4:
            pos = np.array(list(self._pelvis_positions)[-4:])
            vel   = np.diff(pos, axis=0) * self.fps
            accel = np.diff(vel, axis=0) * self.fps
            jerk  = np.diff(accel, axis=0) * self.fps
            self._jerk_history.append(float(np.linalg.norm(jerk[0])))

        return self.get_metrics()

    # ─────────────────────────────────────────────
    # GROUND CONTACT DETECTION
    # ─────────────────────────────────────────────

    def _detect_contact(self, side: str, ankle_y: float, now: float):
        """
        Detects foot strike and toe-off events using a rolling ground threshold.
        Ground threshold = 10th percentile of ankle Y history (Y is up, so lower Y = higher position).
        A contact occurs when ankle Y rises above threshold (foot near ground).
        """
        history = self._ankle_l_y if side == "l" else self._ankle_r_y
        if len(history) < 10:
            return

        ys = [h[0] for h in history]
        # Ground level = maximum Y (lowest physical position in Y-up coordinates)
        ground_y = np.percentile(ys, 90)
        threshold = ground_y - 0.05   # 5 cm above ground = contact zone

        in_contact   = ankle_y >= threshold
        was_contact  = self._l_in_contact if side == "l" else self._r_in_contact
        contact_start = self._l_contact_start if side == "l" else self._r_contact_start

        if in_contact and not was_contact:
            # Foot strike
            contacts = self._l_contacts if side == "l" else self._r_contacts
            contacts.append(now)
            if side == "l":
                self._l_in_contact = True
                self._l_contact_start = now
            else:
                self._r_in_contact = True
                self._r_contact_start = now

        elif not in_contact and was_contact and contact_start:
            # Toe-off — record contact duration
            duration_ms = (now - contact_start) * 1000
            if 80 < duration_ms < 500:   # plausible range
                if side == "l":
                    self._l_contact_durations.append(duration_ms)
                    self._l_in_contact = False
                    self._l_contact_start = None
                else:
                    self._r_contact_durations.append(duration_ms)
                    self._r_in_contact = False
                    self._r_contact_start = None

    # ─────────────────────────────────────────────
    # METRICS
    # ─────────────────────────────────────────────

    def get_metrics(self) -> dict:
        return {
            "stride_symmetry_pct":     round(self._stride_symmetry(), 1),
            "cadence_spm":             round(self._cadence(), 1),
            "gct_asymmetry_ms":        round(self._gct_asymmetry(), 1),
            "vertical_oscillation_cm": round(self._vertical_oscillation(), 2),
            "running_smoothness":      round(self._smoothness(), 1),
            "fatigue_gait_drift":      round(self._fatigue_drift(), 5),
            "knee_asymmetry_deg":      round(self._knee_asymmetry(), 1),
        }

    def _stride_symmetry(self) -> float:
        """
        Symmetry = 100 * (1 - |L_stride - R_stride| / mean_stride)
        Stride time estimated from consecutive ipsilateral foot contacts.
        """
        def stride_times(contacts: list) -> list:
            if len(contacts) < 2:
                return []
            return [contacts[i+1] - contacts[i] for i in range(len(contacts)-1)]

        lt = stride_times(self._l_contacts[-STRIDE_WINDOW:])
        rt = stride_times(self._r_contacts[-STRIDE_WINDOW:])

        if not lt or not rt:
            return 100.0

        mean_l = np.mean(lt)
        mean_r = np.mean(rt)
        mean_all = (mean_l + mean_r) / 2
        if mean_all < 1e-6:
            return 100.0
        sym = 100 * (1 - abs(mean_l - mean_r) / mean_all)
        return max(0.0, min(100.0, sym))

    def _cadence(self) -> float:
        """Steps per minute from all contacts in last 10 s."""
        now = time.time()
        recent_l = [t for t in self._l_contacts if now - t < 10]
        recent_r = [t for t in self._r_contacts if now - t < 10]
        total_steps = len(recent_l) + len(recent_r)
        if total_steps < 2:
            return 0.0
        window_s = min(10.0, now - self._start_time)
        return (total_steps / window_s) * 60

    def _gct_asymmetry(self) -> float:
        """Mean ground contact time difference L vs R in ms."""
        if not self._l_contact_durations or not self._r_contact_durations:
            return 0.0
        return abs(np.mean(self._l_contact_durations) - np.mean(self._r_contact_durations))

    def _vertical_oscillation(self) -> float:
        """Peak-to-trough pelvis Y displacement per stride in cm."""
        if len(self._pelvis_y) < int(self.fps * 0.5):
            return 0.0
        y = np.array(list(self._pelvis_y)[-int(self.fps * 2):])  # last 2 s
        osc_m = float(np.max(y) - np.min(y))
        return osc_m * 100   # metres → cm

    def _smoothness(self) -> float:
        """100 = perfectly smooth; decreases with jerk."""
        if not self._jerk_history:
            return 100.0
        mean_jerk = np.mean(self._jerk_history)
        # Normalise: jerk < 0.5 → ~100%, jerk > 5.0 → ~0%
        score = max(0.0, 100 * (1 - mean_jerk / 5.0))
        return round(score, 1)

    def _fatigue_drift(self) -> float:
        """Linear regression slope of knee asymmetry over the run."""
        asym = list(self._asym_history)
        if len(asym) < 20:
            return 0.0
        x = np.arange(len(asym), dtype=float)
        y = np.array(asym, dtype=float)
        slope = float(np.polyfit(x, y, 1)[0])
        return slope

    def _knee_asymmetry(self) -> float:
        if not self._knee_l_history or not self._knee_r_history:
            return 0.0
        return abs(self._knee_l_history[-1] - self._knee_r_history[-1])

    def get_summary(self) -> dict:
        m = self.get_metrics()
        lc = list(self._l_contact_durations)
        rc = list(self._r_contact_durations)
        return {
            **m,
            "total_strides_l": len(self._l_contacts),
            "total_strides_r": len(self._r_contacts),
            "mean_gct_l_ms": round(float(np.mean(lc)), 1) if lc else 0,
            "mean_gct_r_ms": round(float(np.mean(rc)), 1) if rc else 0,
            "duration_s": round(self._frame / self.fps, 1),
        }


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────

def _joint_angle(a, b, c) -> float:
    v1 = a - b
    v2 = c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_a = np.dot(v1, v2) / (n1 * n2)
    return 180.0 - float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))