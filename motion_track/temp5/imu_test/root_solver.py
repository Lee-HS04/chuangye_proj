"""
root_solver.py  —  Root Motion Solver
======================================

Solves the pelvis root transform: world position, velocity, heading,
and balance-shift estimation.

Responsibility split vs motion_detector.py
------------------------------------------
motion_detector.py  — WHAT the user is doing (action classification)
root_solver.py      — WHERE the body is (position, velocity, CoM)

The root solver integrates the detector's directional output into a stable
world-space root position, applying:
  - Heading-aware direction conversion (body frame → world frame)
  - Speed scaling per action type
  - Vertical offset for squats, jumps, and forward lean
  - Balance-shift estimation (CoM lateral displacement during weight transfer)
  - Drift clamping so the figure never walks off to infinity

CoM estimation
--------------
The true CoM is at roughly 55–57% of height (just above the navel).
With only a pelvis sensor we approximate CoM directly from the pelvis
position plus a small vertical correction based on the pitch angle —
when the body leans forward the pelvis drops relative to the CoM.

Locomotion integration
----------------------
We do NOT double-integrate acceleration (drifts in <1 s).
Instead we use:
  detected.move_dir  × speed_per_action × dt → Δ world position
The result is a plausible locomotion velocity without drift, gated by
the travel/in_place flags from the detector.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from config import GRAVITY, MOVE_SPEED, ROOT_MAX_METRES


# ── math ─────────────────────────────────────────────────────────────────────

def _qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    qv = np.array([0.0, v[0], v[1], v[2]])
    def mul(a, b):
        aw,ax,ay,az = a; bw,bx,by,bz = b
        return np.array([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                         aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])
    r = mul(q, mul(qv, np.array([w,-x,-y,-z])))
    return r[1:]


def _yaw_rad(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2*(w*y + z*x), 1 - 2*(x*x + y*y)))


# ── output types ─────────────────────────────────────────────────────────────

@dataclass
class RootState:
    position:        np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity:        np.ndarray = field(default_factory=lambda: np.zeros(3))
    y_offset:        float      = 0.0   # vertical displacement (squat/jump)
    com_offset:      np.ndarray = field(default_factory=lambda: np.zeros(3))
    heading_deg:     float      = 0.0   # yaw in degrees
    speed_ms:        float      = 0.0   # estimated locomotion speed m/s
    balance_shift:   float      = 0.0   # lateral CoM shift (-1=left, +1=right)

    def to_dict(self) -> dict:
        return {
            'position':      [round(float(v), 4) for v in self.position],
            'velocity':      [round(float(v), 4) for v in self.velocity],
            'y_offset':      round(self.y_offset, 4),
            'heading_deg':   round(self.heading_deg, 1),
            'speed_ms':      round(self.speed_ms, 3),
            'balance_shift': round(self.balance_shift, 3),
        }


# ── solver ────────────────────────────────────────────────────────────────────

class RootSolver:
    """
    Integrates the pelvis root position frame-by-frame.

    Parameters
    ----------
    max_metres : float
        Clamp radius — the figure won't walk beyond this distance from origin.
        Used in the test viewer; set to float('inf') for production pipelines.
    """

    # Vertical response rates (m/s)
    SQUAT_SINK_RATE   =  0.25
    SQUAT_RISE_RATE   =  0.40
    SQUAT_MAX_DEPTH   = -0.35
    JUMP_RISE_RATE    =  2.0
    JUMP_FALL_RATE    =  4.0
    JUMP_MAX_HEIGHT   =  0.55
    LEAN_MAX_DROP     = -0.06   # slight pelvis drop when leaning forward

    def __init__(self, max_metres: float = ROOT_MAX_METRES):
        self._max   = max_metres
        self._state = RootState()
        self._prev_pos   = np.zeros(3)
        self._prev_ts    = 0.0
        # Vertical state machine
        self._y_target   = 0.0
        self._in_jump    = False
        self._jump_phase = 'none'   # ascent | apex | descent | impact | none
        # Balance smoothing
        self._balance_win: deque = deque(maxlen=20)

    # ── main update ───────────────────────────────────────────────────────────

    def update(
        self,
        q_pelvis: np.ndarray,
        detected,              # MotionState from motion_detector
        dt: float,
    ) -> RootState:
        s = self._state
        action     = detected.action
        move_dir   = detected.move_dir or [0.0, 0.0]
        in_place   = detected.in_place
        is_squat   = getattr(detected, 'is_squat', False)
        is_jump    = getattr(detected, 'is_jump', False)
        jump_phase = getattr(detected, 'jump_phase', 'none')

        # ── horizontal position ───────────────────────────────────────────
        if not in_place and action in MOVE_SPEED and (move_dir[0] or move_dir[1]):
            speed     = MOVE_SPEED[action]
            yaw       = _yaw_rad(q_pelvis)
            sinY, cosY = np.sin(yaw), np.cos(yaw)
            bR =  float(move_dir[0])
            bF = -float(move_dir[1])   # detector forward = +Y; world forward = -Z
            wX =  bR*cosY + bF*sinY
            wZ = -bR*sinY + bF*cosY
            s.position[0] = float(np.clip(s.position[0] + wX*speed*dt, -self._max, self._max))
            s.position[2] = float(np.clip(s.position[2] + wZ*speed*dt, -self._max, self._max))
            s.speed_ms = speed * float(np.hypot(move_dir[0], move_dir[1]))
        else:
            s.speed_ms = max(0.0, s.speed_ms - 4.0*dt)   # decelerate smoothly

        # ── vertical (Y) offset ───────────────────────────────────────────
        if is_jump:
            self._update_jump(jump_phase, dt)
        elif is_squat:
            self._y_target = self.SQUAT_MAX_DEPTH
            s.y_offset = max(self.SQUAT_MAX_DEPTH,
                             s.y_offset - self.SQUAT_SINK_RATE * dt)
        else:
            # Forward lean causes slight pelvis drop
            euler_pitch = self._pitch_from_q(q_pelvis)
            lean_drop   = max(self.LEAN_MAX_DROP, -abs(euler_pitch) * 0.001)
            self._y_target = lean_drop
            # Rise back toward target
            rate = self.SQUAT_RISE_RATE if s.y_offset < lean_drop else 4.0
            if s.y_offset < self._y_target:
                s.y_offset = min(self._y_target, s.y_offset + rate * dt)
            elif s.y_offset > self._y_target:
                s.y_offset = max(self._y_target, s.y_offset - rate * dt)

        s.position[1] = s.y_offset

        # ── velocity estimate ─────────────────────────────────────────────
        s.velocity = (s.position - self._prev_pos) / max(dt, 1e-4)
        self._prev_pos = s.position.copy()

        # ── heading ───────────────────────────────────────────────────────
        s.heading_deg = float(np.degrees(_yaw_rad(q_pelvis)))

        # ── balance shift (lateral CoM displacement) ──────────────────────
        # Estimate from pelvis lateral accel during stance transitions.
        # Positive = shifted right, negative = shifted left.
        # We simply use the pelvis roll angle as a proxy.
        roll = self._roll_from_q(q_pelvis)
        self._balance_win.append(roll)
        s.balance_shift = float(np.mean(self._balance_win)) / 15.0   # normalise to ±1

        return s

    def reset(self) -> None:
        self._state    = RootState()
        self._prev_pos = np.zeros(3)
        self._in_jump  = False
        self._jump_phase = 'none'
        self._balance_win.clear()

    @property
    def state(self) -> RootState:
        return self._state

    # ── jump vertical update ──────────────────────────────────────────────────

    def _update_jump(self, phase: str, dt: float) -> None:
        s = self._state
        self._jump_phase = phase
        if phase in ('ascent', 'apex'):
            s.y_offset = min(self.JUMP_MAX_HEIGHT,
                             s.y_offset + self.JUMP_RISE_RATE * dt)
        elif phase in ('descent', 'impact', 'none'):
            s.y_offset = max(0.0, s.y_offset - self.JUMP_FALL_RATE * dt)

    # ── euler helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _pitch_from_q(q: np.ndarray) -> float:
        w, x, y, z = q
        sp = np.clip(2*(w*y - z*x), -1, 1)
        return float(np.degrees(np.arcsin(sp)))

    @staticmethod
    def _roll_from_q(q: np.ndarray) -> float:
        w, x, y, z = q
        return float(np.degrees(np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))))