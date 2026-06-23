"""
Foot contact and gait phase detector.

In consumer mode (no foot sensors) contact is inferred from shin orientation
and pelvis vertical oscillation.  In professional mode (with foot sensors)
accelerometer magnitude and vertical orientation provide a direct signal.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict
import numpy as np

from ..core import quaternion as quat
from ..core.skeleton import Bone, LEG_LENGTH


class GaitPhase(Enum):
    STANCE   = "stance"    # foot planted
    HEEL_OFF = "heel_off"  # rising from stance
    SWING    = "swing"     # foot in the air
    HEEL_ON  = "heel_on"   # approaching ground


@dataclass
class FootState:
    phase:          GaitPhase = GaitPhase.STANCE
    contact:        bool      = True
    ground_height:  float     = 0.0
    phase_progress: float     = 0.0   # 0–1 within current phase


class FootContactDetector:
    """Separate state machines for left and right foot."""

    # Shin near-vertical (small forward angle) ↔ stance phase
    _STANCE_SHIN_PITCH_MAX  = np.radians(20.0)
    _SWING_SHIN_PITCH_MIN   = np.radians(10.0)
    _ACCEL_CONTACT_THRESH   = 1.5 * 9.81  # m/s² spike on heel strike

    def __init__(self):
        self._left  = FootState()
        self._right = FootState()
        # Phase timers (frames)
        self._l_timer = 0
        self._r_timer = 0

    def update(
        self,
        l_shin_q:    Optional[np.ndarray] = None,
        r_shin_q:    Optional[np.ndarray] = None,
        l_foot_q:    Optional[np.ndarray] = None,
        r_foot_q:    Optional[np.ndarray] = None,
        l_foot_accel: Optional[np.ndarray] = None,  # world-space, gravity removed
        r_foot_accel: Optional[np.ndarray] = None,
    ) -> Dict[str, FootState]:
        self._left  = self._update_side(
            self._left, l_shin_q, l_foot_q, l_foot_accel, "left"
        )
        self._right = self._update_side(
            self._right, r_shin_q, r_foot_q, r_foot_accel, "right"
        )
        return {"left": self._left, "right": self._right}

    def _update_side(
        self,
        state:      FootState,
        shin_q:     Optional[np.ndarray],
        foot_q:     Optional[np.ndarray],
        foot_accel: Optional[np.ndarray],
        side:       str,
    ) -> FootState:
        contact_signal = self._infer_contact(shin_q, foot_q, foot_accel)
        prev_contact = state.contact
        state.contact = contact_signal

        if not prev_contact and contact_signal:
            state.phase = GaitPhase.HEEL_ON
        elif prev_contact and not contact_signal:
            state.phase = GaitPhase.HEEL_OFF
        elif contact_signal:
            state.phase = GaitPhase.STANCE
        else:
            state.phase = GaitPhase.SWING

        return state

    def _infer_contact(
        self,
        shin_q:     Optional[np.ndarray],
        foot_q:     Optional[np.ndarray],
        foot_accel: Optional[np.ndarray],
    ) -> bool:
        # If foot accelerometer available — spike indicates contact
        if foot_accel is not None:
            mag = float(np.linalg.norm(foot_accel))
            if mag > self._ACCEL_CONTACT_THRESH:
                return True

        # Fall back to shin-angle heuristic (consumer mode)
        if shin_q is not None:
            euler = quat.to_euler_yup(shin_q)
            shin_pitch = abs(euler[1])  # forward/back tilt of shin
            if shin_pitch < self._STANCE_SHIN_PITCH_MAX:
                return True
            if shin_pitch > self._STANCE_SHIN_PITCH_MAX * 1.5:
                return False

        if foot_q is not None:
            euler = quat.to_euler_yup(foot_q)
            # Foot relatively flat → standing
            if abs(euler[1]) < np.radians(15.0):
                return True

        return False

    @property
    def left(self) -> FootState:
        return self._left

    @property
    def right(self) -> FootState:
        return self._right
