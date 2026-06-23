"""
biomechanics.py  —  Anatomical joint constraint enforcement
============================================================

Provides constraint functions for every major joint in the R2P skeleton.
All inputs/outputs are quaternions in [w, x, y, z] order.

Constraint philosophy
---------------------
* Constraints are SOFT: they blend the raw sensor quaternion toward the
  nearest valid pose rather than hard-clamping.  This avoids violent
  snapping when a sensor reading briefly crosses a limit.
* Blend weight `hardness` (0–1) controls how strongly the constraint is
  enforced.  Default 0.85 gives good realism without over-constraining.
* All constraints operate in LOCAL joint space (relative to parent bone),
  not world space — so they remain valid regardless of the body's heading.

Joint limit reference (approximate anatomical ranges)
------------------------------------------------------
Knee:        flexion 0–140°, extension 0°, varus/valgus ±5°
Elbow:       flexion 0–145°, extension 0°, pronation/supination ±90°
Spine:       flexion/extension ±40°, lateral bend ±30°, axial ±45°
Hip:         flexion 120°, extension 20°, abduction 45°, adduction 30°
Ankle:       dorsiflexion 20°, plantarflexion 50°, inversion 35°, eversion 20°
Shoulder:    elevation 180°, extension 60°, abduction 180°
Neck:        flexion/extension ±60°, lateral ±45°, axial ±80°
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


# ── math helpers ──────────────────────────────────────────────────────────────

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


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    d = float(np.dot(a, b))
    if d < 0:
        b = -b; d = -d
    if d > 0.9995:
        return _norm(a + t * (b - a))
    th = np.arccos(np.clip(d, -1, 1))
    return _norm(np.sin((1-t)*th)/np.sin(th)*a + np.sin(t*th)/np.sin(th)*b)


def _quat_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
    """Returns (yaw, pitch, roll) in degrees."""
    w, x, y, z = q
    yaw   = np.degrees(np.arctan2(2*(w*y + z*x), 1 - 2*(x*x + y*y)))
    sp    = np.clip(2*(w*x - y*z), -1, 1)
    pitch = np.degrees(np.arcsin(float(sp)))
    roll  = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(x*x + z*z)))
    return float(yaw), float(pitch), float(roll)


def _euler_to_quat(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """ZYX Euler → quaternion [w,x,y,z]."""
    y, p, r = np.radians(yaw_deg), np.radians(pitch_deg), np.radians(roll_deg)
    cy, sy = np.cos(y/2), np.sin(y/2)
    cp, sp = np.cos(p/2), np.sin(p/2)
    cr, sr = np.cos(r/2), np.sin(r/2)
    return _norm(np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ]))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ── generic swing-twist decomposition ────────────────────────────────────────

def swing_twist_decompose(q: np.ndarray, twist_axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose quaternion q into swing (perpendicular to twist_axis) and
    twist (rotation around twist_axis).  Returns (swing, twist).
    """
    axis = np.array([q[1], q[2], q[3]])
    proj = np.dot(axis, twist_axis) * twist_axis
    twist = _norm(np.array([q[0], proj[0], proj[1], proj[2]]))
    swing = _norm(_qmul(q, _qconj(twist)))
    return swing, twist


def twist_angle(q: np.ndarray, axis: np.ndarray) -> float:
    """Extract the twist angle (degrees) around `axis`."""
    _, t = swing_twist_decompose(q, axis)
    return float(np.degrees(2 * np.arccos(np.clip(abs(t[0]), 0, 1))))


# ── soft clamping helper ──────────────────────────────────────────────────────

def _soft_clamp_twist(
    q: np.ndarray,
    axis: np.ndarray,
    min_deg: float,
    max_deg: float,
    hardness: float = 0.85,
) -> np.ndarray:
    """
    Clamp the twist component around `axis` to [min_deg, max_deg].
    Returns a blended quaternion (soft, not hard-clamped).
    """
    swing, twist = swing_twist_decompose(q, axis)
    # Get signed twist angle
    t_axis = np.array([twist[1], twist[2], twist[3]])
    sign = 1.0 if np.dot(t_axis, axis) >= 0 else -1.0
    angle = sign * float(np.degrees(2 * np.arccos(np.clip(abs(twist[0]), 0, 1))))
    clamped = _clamp(angle, min_deg, max_deg)
    if abs(clamped - angle) < 0.1:
        return q   # already within limits, no change needed
    # Rebuild clamped twist
    h = np.radians(clamped) / 2
    clamped_twist = _norm(np.array([np.cos(h), axis[0]*np.sin(h), axis[1]*np.sin(h), axis[2]*np.sin(h)]))
    q_clamped = _norm(_qmul(swing, clamped_twist))
    return _norm(_slerp(q, q_clamped, hardness))


# ══════════════════════════════════════════════════════════════════════════════
# JOINT CONSTRAINTS
# ══════════════════════════════════════════════════════════════════════════════

class JointConstraints:
    """
    All constraint methods take a LOCAL quaternion (relative to parent bone)
    and return the constrained version.

    The `hardness` parameter (0–1) controls blend strength:
      1.0 = fully enforced (hard)
      0.0 = no constraint applied
    Default is 0.85 for most joints.
    """

    # ── Knee ─────────────────────────────────────────────────────────────────

    @staticmethod
    def knee(q_local: np.ndarray, side: str = 'l', hardness: float = 0.85) -> np.ndarray:
        """
        Knee is primarily a hinge around the medial-lateral axis (X in local frame).
        Flexion: 0–140°  (positive flexion = bending)
        Extension: 0°    (no hyperextension)
        Varus/valgus (Z twist): ±5°
        Axial rotation (Y twist): ±10° (small amount allowed during flexion)
        """
        q = _norm(q_local)
        # Primary flexion/extension around local X
        q = _soft_clamp_twist(q, np.array([1.0, 0.0, 0.0]),
                               min_deg=0.0, max_deg=140.0, hardness=hardness)
        # Prevent valgus/varus twisting
        q = _soft_clamp_twist(q, np.array([0.0, 0.0, 1.0]),
                               min_deg=-5.0, max_deg=5.0, hardness=hardness)
        # Limit axial rotation
        q = _soft_clamp_twist(q, np.array([0.0, 1.0, 0.0]),
                               min_deg=-10.0, max_deg=10.0, hardness=hardness)
        return _norm(q)

    # ── Elbow ─────────────────────────────────────────────────────────────────

    @staticmethod
    def elbow(q_local: np.ndarray, side: str = 'l', hardness: float = 0.85) -> np.ndarray:
        """
        Elbow is a hinge + limited pronation/supination.
        Flexion: 0–145°
        Extension: 0° (no hyperextension)
        Pronation/supination (Y): ±90°
        Varus/valgus (Z): ±5°
        """
        q = _norm(q_local)
        q = _soft_clamp_twist(q, np.array([1.0, 0.0, 0.0]),
                               min_deg=0.0, max_deg=145.0, hardness=hardness)
        q = _soft_clamp_twist(q, np.array([0.0, 1.0, 0.0]),
                               min_deg=-90.0, max_deg=90.0, hardness=hardness)
        q = _soft_clamp_twist(q, np.array([0.0, 0.0, 1.0]),
                               min_deg=-5.0, max_deg=5.0, hardness=hardness)
        return _norm(q)

    # ── Hip ──────────────────────────────────────────────────────────────────

    @staticmethod
    def hip(q_local: np.ndarray, side: str = 'l', hardness: float = 0.80) -> np.ndarray:
        """
        Hip is a ball-and-socket with wide but bounded range.
        Flexion (+X):    0–120°
        Extension (-X):  0–20°
        Abduction (Z):   0–45° (outward)
        Adduction (Z):   0–30° (inward)
        Internal rotation (Y): 0–45°
        External rotation (Y): 0–45°
        Softer constraint (0.80) — hips have genuine wide range.
        """
        q = _norm(q_local)
        q = _soft_clamp_twist(q, np.array([1.0, 0.0, 0.0]),
                               min_deg=-20.0, max_deg=120.0, hardness=hardness)
        ab_sign = 1.0 if side == 'l' else -1.0
        q = _soft_clamp_twist(q, np.array([0.0, 0.0, 1.0]),
                               min_deg=-30.0, max_deg=45.0 * ab_sign, hardness=hardness * 0.8)
        q = _soft_clamp_twist(q, np.array([0.0, 1.0, 0.0]),
                               min_deg=-45.0, max_deg=45.0, hardness=hardness * 0.7)
        return _norm(q)

    # ── Spine / Lumbar ────────────────────────────────────────────────────────

    @staticmethod
    def spine(q_local: np.ndarray, hardness: float = 0.80) -> np.ndarray:
        """
        Lumbar spine: moderate flex/extend, limited lateral and axial rotation.
        Flexion (+X):   0–40°
        Extension (-X): 0–30°
        Lateral (Z):    ±30°
        Axial (Y):      ±45°
        """
        q = _norm(q_local)
        q = _soft_clamp_twist(q, np.array([1.0, 0.0, 0.0]),
                               min_deg=-30.0, max_deg=40.0, hardness=hardness)
        q = _soft_clamp_twist(q, np.array([0.0, 0.0, 1.0]),
                               min_deg=-30.0, max_deg=30.0, hardness=hardness)
        q = _soft_clamp_twist(q, np.array([0.0, 1.0, 0.0]),
                               min_deg=-45.0, max_deg=45.0, hardness=hardness)
        return _norm(q)

    # ── Thoracic (chest) ──────────────────────────────────────────────────────

    @staticmethod
    def thoracic(q_local: np.ndarray, hardness: float = 0.75) -> np.ndarray:
        """
        Thoracic / chest: slightly more restricted than lumbar.
        Flexion/extension: ±25°
        Lateral: ±20°
        Axial: ±35°
        """
        q = _norm(q_local)
        q = _soft_clamp_twist(q, np.array([1.0, 0.0, 0.0]),
                               min_deg=-25.0, max_deg=25.0, hardness=hardness)
        q = _soft_clamp_twist(q, np.array([0.0, 0.0, 1.0]),
                               min_deg=-20.0, max_deg=20.0, hardness=hardness)
        q = _soft_clamp_twist(q, np.array([0.0, 1.0, 0.0]),
                               min_deg=-35.0, max_deg=35.0, hardness=hardness)
        return _norm(q)

    # ── Neck ─────────────────────────────────────────────────────────────────

    @staticmethod
    def neck(q_local: np.ndarray, hardness: float = 0.80) -> np.ndarray:
        """
        Neck: moderate range, stabilized against extreme positions.
        Flexion (+X):   0–60°
        Extension (-X): 0–50°
        Lateral (Z):    ±45°
        Axial (Y):      ±80°
        """
        q = _norm(q_local)
        q = _soft_clamp_twist(q, np.array([1.0, 0.0, 0.0]),
                               min_deg=-50.0, max_deg=60.0, hardness=hardness)
        q = _soft_clamp_twist(q, np.array([0.0, 0.0, 1.0]),
                               min_deg=-45.0, max_deg=45.0, hardness=hardness)
        q = _soft_clamp_twist(q, np.array([0.0, 1.0, 0.0]),
                               min_deg=-80.0, max_deg=80.0, hardness=hardness)
        return _norm(q)

    # ── Ankle ─────────────────────────────────────────────────────────────────

    @staticmethod
    def ankle(q_local: np.ndarray, side: str = 'l', hardness: float = 0.75) -> np.ndarray:
        """
        Ankle: dorsi/plantarflexion + limited inversion/eversion.
        Dorsiflexion (+X):   0–20°
        Plantarflexion (-X): 0–50°
        Inversion (Z):       ±35°
        Eversion (-Z):       ±20°
        """
        q = _norm(q_local)
        q = _soft_clamp_twist(q, np.array([1.0, 0.0, 0.0]),
                               min_deg=-50.0, max_deg=20.0, hardness=hardness)
        q = _soft_clamp_twist(q, np.array([0.0, 0.0, 1.0]),
                               min_deg=-20.0, max_deg=35.0, hardness=hardness * 0.7)
        return _norm(q)

    # ── Shoulder ──────────────────────────────────────────────────────────────

    @staticmethod
    def shoulder(q_local: np.ndarray, side: str = 'l', hardness: float = 0.70) -> np.ndarray:
        """
        Shoulder: wide range — softly constrained to prevent impossible states.
        Elevation (X):      0–180°
        Extension (-X):     0–60°
        Abduction (Z):      0–180°
        Internal rotation:  ±90°
        Hardness is lower (0.70) because athletic motion needs full range.
        """
        q = _norm(q_local)
        q = _soft_clamp_twist(q, np.array([1.0, 0.0, 0.0]),
                               min_deg=-60.0, max_deg=180.0, hardness=hardness)
        q = _soft_clamp_twist(q, np.array([0.0, 1.0, 0.0]),
                               min_deg=-90.0, max_deg=90.0, hardness=hardness * 0.8)
        return _norm(q)

    # ── Apply all constraints by sensor ID ───────────────────────────────────

    @classmethod
    def apply(cls, sensor_id: str, q_local: np.ndarray, hardness: float = 0.85) -> np.ndarray:
        """
        Dispatch to the correct constraint function by sensor/bone ID.
        Returns constrained quaternion, or the original if no constraint defined.
        """
        sid = sensor_id.lower()
        if 'knee' in sid or 'shin' in sid:
            side = 'l' if '_l' in sid else 'r'
            return cls.knee(q_local, side=side, hardness=hardness)
        elif 'elbow' in sid or 'forearm' in sid:
            side = 'l' if '_l' in sid else 'r'
            return cls.elbow(q_local, side=side, hardness=hardness)
        elif 'hip' in sid or 'thigh' in sid:
            side = 'l' if '_l' in sid else 'r'
            return cls.hip(q_local, side=side, hardness=hardness)
        elif 'lumbar' in sid or 'spine' in sid:
            return cls.spine(q_local, hardness=hardness)
        elif 'chest' in sid or 'thoracic' in sid:
            return cls.thoracic(q_local, hardness=hardness)
        elif 'neck' in sid:
            return cls.neck(q_local, hardness=hardness)
        elif 'ankle' in sid or 'foot' in sid:
            side = 'l' if '_l' in sid else 'r'
            return cls.ankle(q_local, side=side, hardness=hardness)
        elif 'shoulder' in sid:
            side = 'l' if '_l' in sid else 'r'
            return cls.shoulder(q_local, side=side, hardness=hardness)
        return _norm(q_local)