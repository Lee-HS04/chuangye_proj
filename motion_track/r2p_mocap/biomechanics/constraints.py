"""
Biomechanical constraint engine.

Clamps local joint rotations to anatomically plausible ranges.
Ranges are stored as (min_angle, max_angle) in radians for each Euler axis.

Convention: YXZ Euler (yaw, pitch, roll) extracted from local quaternion.
  pitch = flex/extension (nodding / forward-back)
  yaw   = axial rotation (twisting)
  roll  = ab/adduction (sideways)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

from ..core.skeleton import Bone
from ..core import quaternion as quat

_DEG = np.pi / 180.0

# (yaw_min, yaw_max, pitch_min, pitch_max, roll_min, roll_max) — degrees
# Positive pitch = flexion (forward / upward), negative = extension
_LIMITS_DEG: Dict[Bone, Tuple[float, ...]] = {
    Bone.HEAD:        (-70,  70,  -45,  45,  -30,  30),
    Bone.NECK:        (-50,  50,  -30,  30,  -20,  20),
    Bone.SPINE_1:     (-20,  20,  -25,  10,  -10,  10),
    Bone.SPINE_2:     (-20,  20,  -25,  10,  -10,  10),
    Bone.CHEST:       (-25,  25,  -30,  15,  -15,  15),
    Bone.L_SHOULDER:  (-60,  60,  -90, 180,  -40,  40),
    Bone.R_SHOULDER:  (-60,  60, -180,  90,  -40,  40),
    Bone.L_UPPER_ARM: (-90,  90,  -90,  90,  -30,  30),
    Bone.R_UPPER_ARM: (-90,  90,  -90,  90,  -30,  30),
    Bone.L_FOREARM:   (-80,  80,    0, 145,  -10,  10),   # hinge dominant
    Bone.R_FOREARM:   (-80,  80,    0, 145,  -10,  10),
    Bone.L_HAND:      (-30,  30,  -70,  70,  -10,  10),
    Bone.R_HAND:      (-30,  30,  -70,  70,  -10,  10),
    Bone.L_THIGH:     (-30,  30, -120,  20,  -40,  30),
    Bone.R_THIGH:     (-30,  30, -120,  20,  -30,  40),
    Bone.L_SHIN:      ( -5,   5,   -5, 135,  -10,  10),   # hinge dominant
    Bone.R_SHIN:      ( -5,   5,   -5, 135,  -10,  10),
    Bone.L_FOOT:      (-15,  15,  -40,  30,  -10,  10),
    Bone.R_FOOT:      (-15,  15,  -40,  30,  -10,  10),
    Bone.L_TOE:       (  0,   0,   -5,  40,   -5,   5),
    Bone.R_TOE:       (  0,   0,   -5,  40,   -5,   5),
}


def _to_limits(t: Tuple[float, ...]) -> np.ndarray:
    """Convert degrees tuple to radians array [[min,max], [min,max], [min,max]]."""
    arr = np.array(t, dtype=float) * _DEG
    return arr.reshape(3, 2)


_LIMITS_RAD: Dict[Bone, np.ndarray] = {
    b: _to_limits(t) for b, t in _LIMITS_DEG.items()
}


class BiomechanicalConstraints:
    """
    Apply anatomical joint-angle limits to a dictionary of local rotations.
    """

    def __init__(self, strength: float = 1.0):
        """strength=1 → hard clamp, 0 → pass-through."""
        self.strength = strength

    def apply(
        self,
        local_rotations: Dict[Bone, np.ndarray],
    ) -> Dict[Bone, np.ndarray]:
        result = {}
        for bone, q in local_rotations.items():
            limits = _LIMITS_RAD.get(bone)
            if limits is None or self.strength == 0.0:
                result[bone] = q
            else:
                result[bone] = self._clamp(q, limits)
        return result

    def _clamp(self, q: np.ndarray, limits: np.ndarray) -> np.ndarray:
        euler = quat.to_euler_yup(q)               # [yaw, pitch, roll]
        clamped = np.clip(euler, limits[:, 0], limits[:, 1])
        if self.strength < 1.0:
            clamped = euler + self.strength * (clamped - euler)
        return quat.from_euler_yup(*clamped)
