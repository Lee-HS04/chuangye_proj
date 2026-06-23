"""
Root motion solver — estimates world-space pelvis position from the pelvis
IMU orientation and lower-body kinematics.

Without external positioning (GPS / UWB), absolute position drifts.
This solver uses:
  - Step detection from vertical pelvis oscillation
  - Estimated stride length from leg-swing amplitude
  - Direction from pelvis yaw
  - Height from simple biomechanical model (pelvis height ≈ leg_length × cos(avg_knee_angle))
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from ..core import quaternion as quat
from ..core.skeleton import Bone, LEG_LENGTH


_G = 9.81          # m/s²
_WALK_SPEED = 1.4  # m/s default estimated walk speed


@dataclass
class RootMotionState:
    position:         np.ndarray = field(default_factory=lambda: np.array([0.0, LEG_LENGTH, 0.0]))
    velocity:         np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw:              float      = 0.0
    step_count:       int        = 0
    is_moving:        bool       = False
    speed_estimate:   float      = 0.0


class RootMotionSolver:
    """
    Stateful root-motion estimator.  Call update() every frame.
    """

    def __init__(self, body_height: float = 1.75, dt: float = 0.01):
        self.body_height = body_height
        self.dt = dt
        self._leg_length = LEG_LENGTH * (body_height / 1.75)
        self._state = RootMotionState(
            position=np.array([0.0, self._leg_length, 0.0])
        )
        # Step detection state
        self._prev_pelvis_y_local: float = 0.0
        self._step_phase: float = 0.0
        self._velocity_lp: np.ndarray = np.zeros(3)

        # Drift-correction reference height
        self._nominal_height = self._leg_length

    # ------------------------------------------------------------------
    def reset(self):
        self._state = RootMotionState(
            position=np.array([0.0, self._leg_length, 0.0])
        )
        self._velocity_lp[:] = 0.0

    # ------------------------------------------------------------------
    def update(
        self,
        pelvis_orientation: np.ndarray,          # world-frame quat from Madgwick
        l_thigh_orientation: Optional[np.ndarray] = None,
        r_thigh_orientation: Optional[np.ndarray] = None,
        l_shin_orientation:  Optional[np.ndarray] = None,
        r_shin_orientation:  Optional[np.ndarray] = None,
        pelvis_accel_world:  Optional[np.ndarray] = None,  # gravity-removed
    ) -> np.ndarray:
        """Returns updated world-space pelvis position."""
        s = self._state

        # ── 1. Yaw from pelvis sensor ──────────────────────────────────
        euler = quat.to_euler_yup(pelvis_orientation)
        target_yaw = euler[0]
        s.yaw = s.yaw + 0.1 * _angle_diff(target_yaw, s.yaw)  # smooth yaw

        # ── 2. Motion detection from average thigh flexion ─────────────
        avg_flex = 0.0
        n_legs = 0
        for q in [l_thigh_orientation, r_thigh_orientation]:
            if q is not None:
                e = quat.to_euler_yup(q)
                avg_flex += abs(e[1])   # pitch = forward swing
                n_legs += 1
        if n_legs:
            avg_flex /= n_legs

        moving = avg_flex > np.radians(5.0)
        s.is_moving = moving

        # ── 3. Speed estimate from swing amplitude ──────────────────────
        if moving:
            # Rough empirical: speed ≈ 0.8 * stride_freq * stride_length
            # swing angle → stride length heuristic
            stride_m = np.clip(avg_flex * 1.8, 0.1, 2.5)
            s.speed_estimate = np.clip(stride_m * 0.7, 0.0, 8.0)
        else:
            s.speed_estimate = s.speed_estimate * 0.95   # decay to zero

        # ── 4. Horizontal position integration ─────────────────────────
        fwd = np.array([np.sin(s.yaw), 0.0, np.cos(s.yaw)])
        horiz_vel = fwd * s.speed_estimate

        if pelvis_accel_world is not None:
            # Blend dead-reckoning with accelerometer for responsiveness
            acc_horiz = pelvis_accel_world.copy()
            acc_horiz[1] = 0.0
            horiz_vel = horiz_vel * 0.7 + acc_horiz * 0.3

        self._velocity_lp = self._velocity_lp * 0.85 + horiz_vel * 0.15
        s.velocity = self._velocity_lp.copy()
        s.position[0] += s.velocity[0] * self.dt
        s.position[2] += s.velocity[2] * self.dt

        # ── 5. Vertical position (height correction) ────────────────────
        target_h = self._estimate_height(
            l_thigh_orientation, r_thigh_orientation,
            l_shin_orientation,  r_shin_orientation,
        )
        s.position[1] = s.position[1] * 0.9 + target_h * 0.1

        return s.position.copy()

    # ------------------------------------------------------------------
    def _estimate_height(
        self,
        lth: Optional[np.ndarray],
        rth: Optional[np.ndarray],
        lsh: Optional[np.ndarray],
        rsh: Optional[np.ndarray],
    ) -> float:
        """Pelvis height from average knee flexion angle."""
        angles = []
        for th, sh in [(lth, lsh), (rth, rsh)]:
            if th is not None and sh is not None:
                e_th = quat.to_euler_yup(th)
                e_sh = quat.to_euler_yup(sh)
                knee_flex = abs(e_sh[1] - e_th[1])
                angles.append(knee_flex)
        if not angles:
            return self._nominal_height
        avg_flex = np.mean(angles)
        height = self._leg_length * (0.7 + 0.3 * np.cos(np.clip(avg_flex, 0, np.pi * 0.4)))
        return float(np.clip(height, self._leg_length * 0.55, self._leg_length * 1.05))

    @property
    def state(self) -> RootMotionState:
        return self._state


def _angle_diff(a: float, b: float) -> float:
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi
