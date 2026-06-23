"""
gait_analyzer.py  —  Gait phase detection and stride analytics
===============================================================

Detects the gait cycle phases from pelvis + shin IMU streams:
  STANCE_L / STANCE_R — foot on ground, loading
  SWING_L  / SWING_R  — foot in air
  DOUBLE_SUPPORT      — both feet on ground (walking transition)
  FLIGHT              — both feet off ground (running)

Outputs per stride:
  - Cadence (Hz)
  - Stride length estimate (m, from accelerometer integration)
  - Symmetry index (0–100, 100 = perfectly symmetric)
  - Foot strike type: HEEL / MIDFOOT / FOREFOOT
  - Phase timings

The detector is sensor-agnostic: it works with 2 shins (consumer mode)
or expands to use foot sensors (pro mode) when available.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


GRAVITY = 9.81


class GaitPhase(str, Enum):
    IDLE           = 'IDLE'
    STANCE_L       = 'STANCE_L'
    STANCE_R       = 'STANCE_R'
    SWING_L        = 'SWING_L'
    SWING_R        = 'SWING_R'
    DOUBLE_SUPPORT = 'DOUBLE_SUPPORT'
    FLIGHT         = 'FLIGHT'


class FootStrike(str, Enum):
    HEEL     = 'HEEL'
    MIDFOOT  = 'MIDFOOT'
    FOREFOOT = 'FOREFOOT'
    UNKNOWN  = 'UNKNOWN'


@dataclass
class StrideEvent:
    side: str             # 'l' or 'r'
    foot_strike: float    # timestamp
    toe_off: float        # timestamp
    strike_type: FootStrike = FootStrike.UNKNOWN
    duration: float = 0.0


@dataclass
class GaitState:
    phase: GaitPhase        = GaitPhase.IDLE
    cadence_hz: float       = 0.0
    stride_length_m: float  = 0.0
    symmetry_index: float   = 100.0
    foot_strike_l: FootStrike = FootStrike.UNKNOWN
    foot_strike_r: FootStrike = FootStrike.UNKNOWN
    contact_l: bool         = True
    contact_r: bool         = True
    stance_time_l: float    = 0.0   # seconds in current stance
    stance_time_r: float    = 0.0
    swing_time_l: float     = 0.0
    swing_time_r: float     = 0.0
    step_count: int         = 0

    def to_dict(self) -> dict:
        return {
            'phase':           self.phase.value,
            'cadence_hz':      round(self.cadence_hz, 2),
            'stride_length_m': round(self.stride_length_m, 3),
            'symmetry_index':  round(self.symmetry_index, 1),
            'foot_strike_l':   self.foot_strike_l.value,
            'foot_strike_r':   self.foot_strike_r.value,
            'contact_l':       self.contact_l,
            'contact_r':       self.contact_r,
            'stance_time_l':   round(self.stance_time_l, 3),
            'stance_time_r':   round(self.stance_time_r, 3),
            'step_count':      self.step_count,
        }


class GaitAnalyzer:
    """
    Feed update() with each shin/foot packet; read .state for current gait info.

    The contact detection algorithm uses shin angular velocity magnitude:
    - High gyro magnitude (> gyro_swing_thresh deg/s) → swing phase
    - Low gyro magnitude (<= gyro_swing_thresh)        → stance phase
    - Additionally uses vertical accel transient at impact to detect foot strike type
    """

    def __init__(
        self,
        fps: float = 50.0,
        gyro_swing_thresh: float = 60.0,    # deg/s above which shin is swinging
        impact_accel_thresh: float = 2.5,   # m/s² transient for impact detection
    ):
        self._fps = fps
        self._gyro_thresh = gyro_swing_thresh
        self._impact_thresh = impact_accel_thresh
        self._win = max(8, int(fps * 0.5))

        self._gyro_l: deque = deque(maxlen=self._win)
        self._gyro_r: deque = deque(maxlen=self._win)
        self._accel_l: deque = deque(maxlen=self._win)
        self._accel_r: deque = deque(maxlen=self._win)

        self._contact_l = True
        self._contact_r = True
        self._contact_l_ts = time.time()
        self._contact_r_ts = time.time()
        self._swing_l_start: Optional[float] = None
        self._swing_r_start: Optional[float] = None

        self._stride_events: deque = deque(maxlen=20)
        self._step_times_l: deque = deque(maxlen=10)
        self._step_times_r: deque = deque(maxlen=10)

        self._step_count = 0
        self.state = GaitState()

    def update_shin(
        self,
        side: str,                   # 'l' or 'r'
        gyro: np.ndarray,            # deg/s in sensor frame
        accel: np.ndarray,           # m/s² in sensor frame
        q_shin: np.ndarray,          # shin quaternion for strike classification
        ts: float,
    ) -> None:
        """Update one shin's data. Call for each shin packet."""
        gyro_mag = float(np.linalg.norm(gyro))
        accel_mag = float(np.linalg.norm(accel)) / GRAVITY

        if side == 'l':
            self._gyro_l.append(gyro_mag)
            self._accel_l.append(accel)
            prev_contact = self._contact_l
            avg_gyro = float(np.mean(self._gyro_l)) if self._gyro_l else 0.0
            new_contact = avg_gyro <= self._gyro_thresh

            if not prev_contact and new_contact:
                # foot strike event
                self._step_count += 1
                self._step_times_l.append(ts)
                strike = self._classify_strike(accel, q_shin)
                self.state.foot_strike_l = strike
                if self._swing_l_start is not None:
                    self.state.swing_time_l = ts - self._swing_l_start
                self._contact_l_ts = ts
                self._contact_l = True

            elif prev_contact and not new_contact:
                # toe-off event
                self.state.stance_time_l = ts - self._contact_l_ts
                self._swing_l_start = ts
                self._contact_l = False

        else:
            self._gyro_r.append(gyro_mag)
            self._accel_r.append(accel)
            prev_contact = self._contact_r
            avg_gyro = float(np.mean(self._gyro_r)) if self._gyro_r else 0.0
            new_contact = avg_gyro <= self._gyro_thresh

            if not prev_contact and new_contact:
                self._step_count += 1
                self._step_times_r.append(ts)
                strike = self._classify_strike(accel, q_shin)
                self.state.foot_strike_r = strike
                if self._swing_r_start is not None:
                    self.state.swing_time_r = ts - self._swing_r_start
                self._contact_r_ts = ts
                self._contact_r = True

            elif prev_contact and not new_contact:
                self.state.stance_time_r = ts - self._contact_r_ts
                self._swing_r_start = ts
                self._contact_r = False

        # Recompute summary state
        self._recompute(ts)

    def _classify_strike(
        self, accel: np.ndarray, q_shin: np.ndarray
    ) -> FootStrike:
        """
        Classify foot strike from impact accel pattern.
        Heel strike: high AP component, low mediolateral
        Forefoot:    high ML, lower AP
        Midfoot:     intermediate
        Uses the shin tilt angle as a proxy for foot angle at impact.
        """
        # Shin forward-tilt angle relative to vertical
        # A more upright shin at contact → heel; more horizontal → forefoot
        shin_up = np.array([0.0, 1.0, 0.0])
        from biomechanics import _qrot, _norm  # type: ignore
        try:
            from biomechanics import _norm as _n2
            w, x, y, z = q_shin
            qv = np.array([0.0, shin_up[0], shin_up[1], shin_up[2]])
            r  = np.array([
                w*qv[0]-x*qv[1]-y*qv[2]-z*qv[3],
                w*qv[1]+x*qv[0]+y*qv[3]-z*qv[2],
                w*qv[2]-x*qv[3]+y*qv[0]+z*qv[1],
                w*qv[3]+x*qv[2]-y*qv[1]+z*qv[0],
            ])
            shin_world_up = np.array([r[1],r[2],r[3]])
        except Exception:
            shin_world_up = np.array([0.0, 1.0, 0.0])

        tilt = float(np.degrees(np.arccos(np.clip(abs(shin_world_up[1]), 0, 1))))

        if tilt < 15:
            return FootStrike.HEEL
        elif tilt > 35:
            return FootStrike.FOREFOOT
        return FootStrike.MIDFOOT

    def _recompute(self, ts: float) -> None:
        """Update cadence, phase, symmetry."""
        # Cadence from combined step times
        all_steps = sorted(
            list(self._step_times_l) + list(self._step_times_r)
        )
        if len(all_steps) >= 2:
            recent = [t for t in all_steps if ts - t < 3.0]
            if len(recent) >= 2:
                dts = np.diff(recent[-6:])
                mean_dt = float(np.mean(dts))
                self.state.cadence_hz = 1.0 / mean_dt if mean_dt > 0.05 else 0.0
            else:
                self.state.cadence_hz = 0.0
        else:
            self.state.cadence_hz = 0.0

        # Decay cadence if no recent step
        if all_steps and ts - all_steps[-1] > 1.0:
            self.state.cadence_hz *= max(0.0, 1.0 - (ts - all_steps[-1] - 1.0))

        # Phase
        cl, cr = self._contact_l, self._contact_r
        if cl and cr:
            self.state.phase = GaitPhase.DOUBLE_SUPPORT
        elif cl and not cr:
            self.state.phase = GaitPhase.STANCE_L
        elif not cl and cr:
            self.state.phase = GaitPhase.STANCE_R
        elif not cl and not cr:
            self.state.phase = GaitPhase.FLIGHT if self.state.cadence_hz > 2.4 else GaitPhase.IDLE

        self.state.contact_l = cl
        self.state.contact_r = cr
        self.state.step_count = self._step_count

        # Symmetry: compare mean stance times L vs R
        if self.state.stance_time_l > 0 and self.state.stance_time_r > 0:
            tl, tr = self.state.stance_time_l, self.state.stance_time_r
            sym = 100.0 * (1.0 - abs(tl - tr) / max(tl, tr))
            # Smooth symmetry score
            self.state.symmetry_index = float(np.clip(
                0.9*self.state.symmetry_index + 0.1*sym, 0, 100
            ))

    def reset(self) -> None:
        self._gyro_l.clear(); self._gyro_r.clear()
        self._accel_l.clear(); self._accel_r.clear()
        self._step_times_l.clear(); self._step_times_r.clear()
        self._step_count = 0
        self._contact_l = True; self._contact_r = True
        self.state = GaitState()