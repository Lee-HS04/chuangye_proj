"""
Yaw drift corrector for the R2P mocap engine.

Gyroscope integration accumulates yaw error over time.  Without
magnetometers, the best we can do is:

  1. Soft yaw reference — the pelvis yaw slowly bleeds back toward the
     initialised "forward" direction when the subject is stationary.
  2. Relative-drift lock — other sensors are corrected to stay within a
     plausible yaw offset of the pelvis rather than drifting freely.
  3. Stand-still zero-velocity update (ZUPT) — when the accelerometer
     norm is close to gravity and gyro magnitude is near zero, treat the
     frame as stationary and reset gyro integration state.
"""

from __future__ import annotations
from typing import Dict, Optional
import numpy as np

from ..core import quaternion as quat
from ..core.config import SensorID


class DriftCorrector:

    def __init__(
        self,
        correction_rate: float = 0.003,    # fraction to pull per frame
        zupt_gyro_thresh: float = 0.08,    # rad/s — below = "stationary"
        zupt_accel_thresh: float = 0.25,   # m/s² deviation from g
    ):
        self.correction_rate = correction_rate
        self.zupt_gyro_thresh  = zupt_gyro_thresh
        self.zupt_accel_thresh = zupt_accel_thresh
        self._reference_yaw: Optional[float] = None

    def set_reference_yaw(self, yaw: float):
        self._reference_yaw = yaw

    def correct(
        self,
        orientations: Dict[str, np.ndarray],
        gyros:        Optional[Dict[str, np.ndarray]] = None,
        accels:       Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Apply drift corrections and return updated orientation dict.

        orientations: sensor_id → world-frame quaternion
        gyros:        sensor_id → body-frame angular velocity (rad/s)
        accels:       sensor_id → body-frame acceleration (m/s²)
        """
        result = {k: v.copy() for k, v in orientations.items()}

        # ── 1. ZUPT — freeze yaw when clearly stationary ──────────────
        if gyros and accels:
            for sid, q in result.items():
                g = gyros.get(sid)
                a = accels.get(sid)
                if g is None or a is None:
                    continue
                if (np.linalg.norm(g) < self.zupt_gyro_thresh and
                        abs(np.linalg.norm(a) - 9.81) < self.zupt_accel_thresh):
                    # Snap yaw to current reference
                    result[sid] = self._zero_yaw_component(q, result[sid])

        # ── 2. Pelvis yaw soft reference ──────────────────────────────
        if self._reference_yaw is not None:
            q_pel = result.get(SensorID.PELVIS)
            if q_pel is not None:
                result[SensorID.PELVIS] = self._blend_yaw(
                    q_pel, self._reference_yaw, self.correction_rate
                )

        # ── 3. Other sensors: limit yaw drift relative to pelvis ───────
        q_pel = result.get(SensorID.PELVIS)
        if q_pel is not None:
            pel_yaw = quat.to_euler_yup(q_pel)[0]
            for sid, q in result.items():
                if sid == SensorID.PELVIS:
                    continue
                euler = quat.to_euler_yup(q)
                # Allow up to ±45° yaw difference from pelvis
                diff = _angle_diff(euler[0], pel_yaw)
                if abs(diff) > np.radians(45.0):
                    target_yaw = pel_yaw + np.sign(diff) * np.radians(45.0)
                    result[sid] = self._blend_yaw(q, target_yaw, 0.05)

        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _blend_yaw(q: np.ndarray, target_yaw: float, rate: float) -> np.ndarray:
        euler = quat.to_euler_yup(q)
        current_yaw = euler[0]
        diff = _angle_diff(target_yaw, current_yaw)
        new_yaw = current_yaw + rate * diff
        return quat.from_euler_yup(new_yaw, euler[1], euler[2])

    @staticmethod
    def _zero_yaw_component(q_ref: np.ndarray, q: np.ndarray) -> np.ndarray:
        euler = quat.to_euler_yup(q)
        ref_euler = quat.to_euler_yup(q_ref)
        rate = 0.02
        new_yaw = euler[0] + rate * _angle_diff(ref_euler[0], euler[0])
        return quat.from_euler_yup(new_yaw, euler[1], euler[2])


def _angle_diff(a: float, b: float) -> float:
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi
