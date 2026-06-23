"""
analytics.py  —  Biomechanical performance scores
==================================================

Computes the five analytics scores defined in the system spec:
  stability_score    — resistance to sway and postural perturbation   (0–100)
  explosive_score    — capacity for rapid acceleration/deceleration    (0–100)
  fatigue_score      — estimated accumulated fatigue                   (0–100)
  mobility_score     — range of motion across lower-body joints        (0–100)
  balance_score      — CoM stability over base of support              (0–100)
  symmetry_score     — left/right kinematic symmetry                   (0–100)

All scores use rolling windows so they update in real time without
requiring complete stride cycles.

Design intent
-------------
Scores are heuristic biomechanical estimates, not clinical measurements.
They are calibrated to feel meaningful in the range of natural human
athletic movement at the sensor update rates (60–240 Hz).

The fatigue model uses a dual-component approach:
  acute load:  accumulates quickly from high-intensity motion (gyro, accel peaks)
  recovery:    decays slowly during low-activity periods
  fatigue = acute load component, mapped to 0–100 where 0 = fresh

Stability uses a Centre-of-Mass (CoM) approximation derived from the
pelvis IMU: sway is estimated from mediolateral acceleration variance
and yaw rate inconsistency over a short window.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


GRAVITY = 9.81


@dataclass
class AnalyticsState:
    stability_score:  float = 80.0
    explosive_score:  float = 0.0
    fatigue_score:    float = 0.0
    mobility_score:   float = 80.0
    balance_score:    float = 80.0
    symmetry_score:   float = 100.0

    def to_dict(self) -> dict:
        return {
            'stability_score':  round(self.stability_score,  1),
            'explosive_score':  round(self.explosive_score,  1),
            'fatigue_score':    round(self.fatigue_score,     1),
            'mobility_score':   round(self.mobility_score,    1),
            'balance_score':    round(self.balance_score,     1),
            'symmetry_score':   round(self.symmetry_score,    1),
        }


class AnalyticsEngine:
    """
    Call update() for each pelvis packet (primary driver).
    Also call update_limb() for each limb sensor packet to feed
    joint range-of-motion and symmetry estimates.

    Parameters
    ----------
    window_s : float
        Rolling window size in seconds for short-term metrics.
    fatigue_decay : float
        How quickly fatigue decays during rest (per second).
    fatigue_gain : float
        How quickly fatigue builds from sustained high-intensity motion.
    """

    def __init__(
        self,
        window_s: float = 3.0,
        fps: float = 50.0,
        fatigue_decay: float = 0.008,
        fatigue_gain: float = 0.006,
    ):
        self._win = max(8, int(fps * window_s))
        self._fps = fps
        self._fatigue_decay = fatigue_decay
        self._fatigue_gain  = fatigue_gain

        # Pelvis dynamics windows
        self._pel_accel_ml:  deque = deque(maxlen=self._win)  # mediolateral (X) accel
        self._pel_accel_vert:deque = deque(maxlen=self._win)  # vertical (Y) accel
        self._pel_gyro_yaw:  deque = deque(maxlen=self._win)  # yaw rate
        self._pel_gyro_mag:  deque = deque(maxlen=self._win)  # total gyro magnitude
        self._accel_peak_window: deque = deque(maxlen=self._win)  # for explosive score

        # Limb ROM tracking: {side: [max_angle_seen]}
        self._thigh_range_l: float = 0.0
        self._thigh_range_r: float = 0.0
        self._shin_range_l:  float = 0.0
        self._shin_range_r:  float = 0.0

        # Symmetry: rolling difference between L and R thigh gyro magnitudes
        self._gyro_thigh_l: deque = deque(maxlen=self._win)
        self._gyro_thigh_r: deque = deque(maxlen=self._win)

        # Fatigue: dual-component
        self._acute_load:   float = 0.0
        self._last_ts:      float = 0.0
        self._session_start:float = time.time()

        self.state = AnalyticsState()

    # ── pelvis update (primary) ───────────────────────────────────────────────

    def update(
        self,
        q_pelvis: np.ndarray,
        accel: np.ndarray,     # m/s² sensor frame
        gyro: np.ndarray,      # deg/s sensor frame
        ts: float,
        action: str = 'idle',
    ) -> AnalyticsState:
        dt = min(ts - self._last_ts, 0.1) if self._last_ts > 0 else 0.02
        self._last_ts = ts

        a = accel.astype(float)
        g = gyro.astype(float)

        # Rotate accel to world frame, remove gravity to get linear accel
        from sensor_fusion import _qrot, _norm  # type: ignore
        try:
            a_world = _qrot(q_pelvis, a)
        except Exception:
            a_world = a.copy()
        a_lin = a_world - np.array([0.0, GRAVITY, 0.0])
        a_ml   = float(a_lin[0])      # mediolateral
        a_vert = float(a_lin[1])      # vertical
        g_yaw  = float(g[1])          # yaw rate
        g_mag  = float(np.linalg.norm(g))
        a_mag  = float(np.linalg.norm(a_lin))

        self._pel_accel_ml.append(a_ml)
        self._pel_accel_vert.append(a_vert)
        self._pel_gyro_yaw.append(g_yaw)
        self._pel_gyro_mag.append(g_mag)
        self._accel_peak_window.append(a_mag)

        # ── stability score ───────────────────────────────────────────────
        # Low sway (low ML accel variance) + consistent heading = high stability
        ml_var = float(np.var(self._pel_accel_ml)) if len(self._pel_accel_ml) > 2 else 0.0
        yaw_var = float(np.var(self._pel_gyro_yaw)) if len(self._pel_gyro_yaw) > 2 else 0.0
        sway = ml_var * 2.0 + yaw_var * 0.05
        # Map: sway=0 → 100, sway=10 → 0
        raw_stab = max(0.0, 100.0 * (1.0 - sway / 10.0))
        # Low-pass smooth
        self.state.stability_score = _ewma(self.state.stability_score, raw_stab, 0.05)

        # ── explosive score ───────────────────────────────────────────────
        # Peak horizontal acceleration over the window, mapped to 0–100
        if len(self._accel_peak_window) > 1:
            peak_a = float(np.percentile(self._accel_peak_window, 95))
            # 0 m/s² → 0,  15 m/s² → 100
            raw_exp = min(100.0, peak_a / 15.0 * 100.0)
        else:
            raw_exp = 0.0
        self.state.explosive_score = _ewma(self.state.explosive_score, raw_exp, 0.08)

        # ── fatigue score ─────────────────────────────────────────────────
        # Acute load builds from combined accel + gyro intensity
        intensity = (g_mag / 300.0) * 0.5 + (a_mag / GRAVITY) * 0.5
        if intensity > 0.3:
            # Active: load accumulates
            self._acute_load += intensity * self._fatigue_gain * dt
        else:
            # Rest: load decays
            self._acute_load -= self._fatigue_decay * dt
        self._acute_load = float(np.clip(self._acute_load, 0.0, 1.0))
        raw_fatigue = self._acute_load * 100.0
        self.state.fatigue_score = _ewma(self.state.fatigue_score, raw_fatigue, 0.02)

        # ── balance score ─────────────────────────────────────────────────
        # Similar to stability but also penalizes large vertical oscillation
        vert_var = float(np.var(self._pel_accel_vert)) if len(self._pel_accel_vert) > 2 else 0.0
        balance_pen = ml_var * 1.5 + vert_var * 0.5
        raw_bal = max(0.0, 100.0 * (1.0 - balance_pen / 12.0))
        self.state.balance_score = _ewma(self.state.balance_score, raw_bal, 0.04)

        return self.state

    # ── limb update ───────────────────────────────────────────────────────────

    def update_limb(
        self,
        sensor_id: str,
        gyro: np.ndarray,
        q: np.ndarray,
        ts: float,
    ) -> None:
        """Call for each thigh/shin packet to update mobility and symmetry."""
        g_mag = float(np.linalg.norm(gyro))
        sid = sensor_id.lower()

        if 'thigh_l' in sid:
            self._gyro_thigh_l.append(g_mag)
            # Track ROM from euler pitch
            angle = abs(_euler_pitch(q))
            self._thigh_range_l = max(self._thigh_range_l * 0.999, angle)
        elif 'thigh_r' in sid:
            self._gyro_thigh_r.append(g_mag)
            angle = abs(_euler_pitch(q))
            self._thigh_range_r = max(self._thigh_range_r * 0.999, angle)
        elif 'shin_l' in sid:
            angle = abs(_euler_pitch(q))
            self._shin_range_l = max(self._shin_range_l * 0.999, angle)
        elif 'shin_r' in sid:
            angle = abs(_euler_pitch(q))
            self._shin_range_r = max(self._shin_range_r * 0.999, angle)

        # ── mobility score ────────────────────────────────────────────────
        # Based on observed joint ROM vs expected healthy ROM
        # Expected: thigh ~40°, shin ~90° (walking); higher is better up to limits
        thigh_score = min(100.0, (self._thigh_range_l + self._thigh_range_r) / 2.0 / 40.0 * 100.0)
        shin_score  = min(100.0, (self._shin_range_l  + self._shin_range_r)  / 2.0 / 90.0 * 100.0)
        raw_mob = 0.5 * thigh_score + 0.5 * shin_score
        self.state.mobility_score = _ewma(self.state.mobility_score, raw_mob, 0.01)

        # ── symmetry score ────────────────────────────────────────────────
        if len(self._gyro_thigh_l) > 2 and len(self._gyro_thigh_r) > 2:
            mean_l = float(np.mean(self._gyro_thigh_l))
            mean_r = float(np.mean(self._gyro_thigh_r))
            if max(mean_l, mean_r) > 1e-3:
                sym = 100.0 * (1.0 - abs(mean_l - mean_r) / max(mean_l, mean_r))
                self.state.symmetry_score = _ewma(self.state.symmetry_score, sym, 0.03)

    def reset(self) -> None:
        for q in (self._pel_accel_ml, self._pel_accel_vert, self._pel_gyro_yaw,
                  self._pel_gyro_mag, self._accel_peak_window,
                  self._gyro_thigh_l, self._gyro_thigh_r):
            q.clear()
        self._acute_load = 0.0
        self.state = AnalyticsState()


# ── helpers ───────────────────────────────────────────────────────────────────

def _ewma(prev: float, new: float, alpha: float) -> float:
    """Exponential weighted moving average: larger alpha = more responsive."""
    return float(np.clip(prev + alpha * (new - prev), 0.0, 100.0))


def _euler_pitch(q: np.ndarray) -> float:
    """Extract pitch angle in degrees from quaternion [w,x,y,z]."""
    w, x, y, z = q.astype(float)
    sp = np.clip(2.0 * (w*x - y*z), -1.0, 1.0)
    return float(np.degrees(np.arcsin(sp)))


def _qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.array([0.0, v[0], v[1], v[2]])
    w, x, y, z = q
    def mul(a, b):
        w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
        return np.array([w1*w2-x1*x2-y1*y2-z1*z2,w1*x2+x1*w2+y1*z2-z1*y2,
                         w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2])
    r = mul(q, mul(qv, np.array([w,-x,-y,-z])))
    return r[1:]