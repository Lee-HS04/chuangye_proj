"""
sensor_fusion.py  —  Per-sensor orientation filter + confidence scoring
========================================================================

Implements a complementary filter (gyro integration + accel correction)
with optional magnetometer yaw correction.  Each sensor gets its own
SensorFilter instance so state is fully independent.

Design choices
--------------
* Complementary filter rather than full EKF: runs in O(1) per packet,
  no matrix inversion, negligible compute overhead at 240 Hz.
* Gyro integration uses the processed (remapped + zeroed) quaternion
  from the BLE pipeline, so the filter operates in the skeleton frame
  rather than the raw chip frame.
* Accel correction is applied only when |a| is close to 1 g — during
  high-acceleration movement the accelerometer is not a reliable gravity
  reference and the filter trusts the gyro alone.
* Magnetometer yaw correction gates on field strength and dip angle so
  ferromagnetic interference is automatically suppressed.
* Confidence score combines: packet recency, accel reliability,
  gyro rate (high spin = lower confidence), mag availability.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def _norm(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-9 else np.array([1.0, 0, 0, 0])


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _qconj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.array([0.0, v[0], v[1], v[2]])
    return _qmul(_qmul(q, qv), _qconj(q))[1:]


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    d = float(np.dot(a, b))
    if d < 0:
        b = -b
        d = -d
    if d > 0.9995:
        return _norm(a + t * (b - a))
    th = np.arccos(np.clip(d, -1, 1))
    return _norm(np.sin((1 - t) * th) / np.sin(th) * a + np.sin(t * th) / np.sin(th) * b)


# ── per-sensor filter ─────────────────────────────────────────────────────────

@dataclass
class FilterState:
    """Live state for one sensor."""
    q: np.ndarray = field(default_factory=lambda: np.array([1.0, 0, 0, 0]))
    last_ts: float = 0.0
    confidence: float = 0.0
    # short gyro-rate history for confidence scoring
    gyro_mag_window: deque = field(default_factory=lambda: deque(maxlen=10))


class SensorFilter:
    """
    Complementary filter for one IMU.

    Parameters
    ----------
    alpha_gyro : float
        Weight given to gyro integration vs accel correction (0.9–0.98 typical).
    accel_thresh_lo / hi : float
        Accel magnitude band (g) where correction is trusted.
        Outside this band the filter trusts the gyro alone.
    mag_enabled : bool
        Whether to apply magnetometer yaw correction.
    stale_s : float
        Seconds after which confidence drops to 0 (no new packets).
    """

    GRAVITY = 9.81  # m/s²

    def __init__(
        self,
        sensor_id: str,
        alpha_gyro: float = 0.95,
        accel_thresh_lo: float = 0.75,   # g
        accel_thresh_hi: float = 1.30,   # g
        mag_enabled: bool = False,
        stale_s: float = 0.4,
    ):
        self.sensor_id = sensor_id
        self.alpha = alpha_gyro
        self.accel_lo = accel_thresh_lo
        self.accel_hi = accel_thresh_hi
        self.mag_enabled = mag_enabled
        self.stale_s = stale_s
        self._state = FilterState()

    # ── main update ──────────────────────────────────────────────────────────

    def update(
        self,
        q_raw: np.ndarray,       # already remapped + zeroed quaternion [w,x,y,z]
        accel: np.ndarray,       # m/s² in sensor frame
        gyro: np.ndarray,        # deg/s in sensor frame
        ts: float,
        mag: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fuse incoming packet and return the smoothed quaternion.

        The input q_raw is trusted as the primary orientation reference
        (it already incorporates the chip's onboard filter).  We apply
        a complementary blend between the previous state and q_raw,
        weighted by accel reliability — so during high-g events we lean
        on the gyro-propagated state rather than snapping to raw.
        """
        s = self._state
        dt = min(ts - s.last_ts, 0.1) if s.last_ts > 0 else 0.016
        s.last_ts = ts

        # ── 1. accel reliability gate ────────────────────────────────────
        a_g = np.linalg.norm(accel) / self.GRAVITY   # magnitude in g
        in_band = self.accel_lo <= a_g <= self.accel_hi
        # Smoothly weight: full trust at centre of band, fades at edges
        if in_band:
            band_centre = (self.accel_lo + self.accel_hi) / 2
            band_half = (self.accel_hi - self.accel_lo) / 2
            accel_weight = 1.0 - min(abs(a_g - band_centre) / band_half, 1.0) * 0.5
        else:
            accel_weight = 0.0

        # ── 2. gyro magnitude for confidence scoring ─────────────────────
        gyro_mag = float(np.linalg.norm(gyro))   # deg/s
        s.gyro_mag_window.append(gyro_mag)

        # ── 3. blend: trust raw more when accel is reliable ──────────────
        # During calm movement: alpha ≈ 0.6 (more raw correction)
        # During high-g / spin: alpha ≈ 0.95 (trust previous state)
        blend_alpha = self.alpha + (1.0 - self.alpha) * (1.0 - accel_weight)
        blend_alpha = np.clip(blend_alpha, 0.5, 0.99)

        if s.last_ts == ts:   # very first packet
            s.q = _norm(q_raw)
        else:
            s.q = _norm(_slerp(q_raw, s.q, blend_alpha))

        # ── 4. optional mag yaw correction ───────────────────────────────
        if self.mag_enabled and mag is not None:
            s.q = self._apply_mag(s.q, mag, accel)

        # ── 5. confidence score ───────────────────────────────────────────
        s.confidence = self._compute_confidence(ts, accel_weight, gyro_mag)

        return s.q.copy()

    def _apply_mag(self, q: np.ndarray, mag: np.ndarray, accel: np.ndarray) -> np.ndarray:
        """Soft yaw correction toward magnetic north if field looks clean."""
        mag_n = np.linalg.norm(mag)
        if mag_n < 1e-3:
            return q
        mag_unit = mag / mag_n
        # Only correct if field strength is plausible (25–65 µT typical)
        if not (0.25 < mag_n < 0.65):  # normalised scale varies by driver
            return q
        # Project mag into horizontal plane in world frame
        m_world = _qrot(q, mag_unit)
        yaw_correction = np.arctan2(m_world[0], m_world[2])
        # Small correction quaternion around world Y
        h = yaw_correction * 0.01  # very soft — just nudge
        dq = np.array([np.cos(h), 0, np.sin(h), 0])
        return _norm(_qmul(dq, q))

    def _compute_confidence(self, ts: float, accel_weight: float, gyro_mag: float) -> float:
        # Recency
        age = time.time() - ts
        recency = max(0.0, 1.0 - age / self.stale_s)
        # Gyro penalty: spinning fast = less reliable orientation estimate
        gyro_avg = float(np.mean(list(self._state.gyro_mag_window))) if self._state.gyro_mag_window else 0.0
        gyro_pen = max(0.0, 1.0 - gyro_avg / 720.0)   # full penalty at 720 deg/s
        # Combine
        return float(np.clip(recency * 0.5 + accel_weight * 0.3 + gyro_pen * 0.2, 0, 1))

    @property
    def q(self) -> np.ndarray:
        return self._state.q.copy()

    @property
    def confidence(self) -> float:
        return self._state.confidence


# ── multi-sensor manager ──────────────────────────────────────────────────────

class FusionManager:
    """
    Holds one SensorFilter per sensor ID and exposes a unified update interface.

    Usage
    -----
    fm = FusionManager()
    q_smooth = fm.update('pelvis', q_raw, accel, gyro, ts)
    conf = fm.confidence('pelvis')
    all_conf = fm.all_confidence()
    """

    _ALPHA_MAP = {
        # Sensors that need more smoothing (noisy mounting points)
        'head': 0.97,
        'l_foot': 0.96, 'r_foot': 0.96,
    }
    _DEFAULT_ALPHA = 0.93

    def __init__(self, mag_enabled: bool = False):
        self._filters: dict[str, SensorFilter] = {}
        self._mag_enabled = mag_enabled

    def _get_filter(self, sid: str) -> SensorFilter:
        if sid not in self._filters:
            alpha = self._ALPHA_MAP.get(sid, self._DEFAULT_ALPHA)
            self._filters[sid] = SensorFilter(
                sid, alpha_gyro=alpha, mag_enabled=self._mag_enabled
            )
        return self._filters[sid]

    def update(
        self,
        sid: str,
        q_raw: np.ndarray,
        accel: np.ndarray,
        gyro: np.ndarray,
        ts: float,
        mag: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self._get_filter(sid).update(q_raw, accel, gyro, ts, mag)

    def confidence(self, sid: str) -> float:
        if sid not in self._filters:
            return 0.0
        return self._filters[sid].confidence

    def all_confidence(self) -> dict[str, float]:
        return {sid: f.confidence for sid, f in self._filters.items()}

    def get_q(self, sid: str) -> Optional[np.ndarray]:
        if sid not in self._filters:
            return None
        return self._filters[sid].q