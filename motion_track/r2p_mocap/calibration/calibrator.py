"""
T-pose / A-pose calibration for the R2P suit.

The calibrator records sensor orientations in a known neutral pose, then
computes per-sensor offsets so that subsequent readings are expressed
relative to that calibration reference.

T-pose: arms horizontal, palms down, facing +Z.
A-pose: arms at ~45° (relaxed down-diagonal).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import numpy as np
from ..core import quaternion as quat
from ..core.config import SensorID


@dataclass
class CalibrationData:
    """Immutable result of a calibration pass."""
    pose_name: str
    offsets:   Dict[str, np.ndarray]   # sensor_id → quaternion offset
    forward:   np.ndarray              # estimated body forward vector in world frame
    timestamp: float = 0.0

    def apply(self, sensor_id: str, q_world: np.ndarray) -> np.ndarray:
        """Remove calibration offset from a world-frame orientation."""
        offset = self.offsets.get(sensor_id)
        if offset is None:
            return quat.normalize(q_world)
        return quat.normalize(quat.multiply(quat.inverse(offset), q_world))


class Calibrator:
    """
    Accumulates N frames of sensor data while the subject holds a neutral
    pose, then computes the average orientation for each sensor and derives
    a forward-vector estimate.
    """

    # T-pose reference: what each sensor's orientation *should* read after
    # removing the offset.  IDENTITY means "align body axes with world axes".
    T_POSE_REFERENCES: Dict[str, np.ndarray] = {
        sid: quat.IDENTITY.copy()
        for sid in [
            SensorID.PELVIS, SensorID.CHEST, SensorID.HEAD,
            SensorID.L_THIGH, SensorID.R_THIGH,
            SensorID.L_SHIN,  SensorID.R_SHIN,
            SensorID.L_FOOT,  SensorID.R_FOOT,
        ]
    }
    # Arms horizontal in T-pose: left arm points −X, right arm points +X
    T_POSE_REFERENCES[SensorID.L_UPPER_ARM] = quat.from_euler_yup(0, 0, np.radians(-90))
    T_POSE_REFERENCES[SensorID.R_UPPER_ARM] = quat.from_euler_yup(0, 0, np.radians( 90))
    T_POSE_REFERENCES[SensorID.L_FOREARM]   = quat.from_euler_yup(0, 0, np.radians(-90))
    T_POSE_REFERENCES[SensorID.R_FOREARM]   = quat.from_euler_yup(0, 0, np.radians( 90))
    T_POSE_REFERENCES[SensorID.L_SHOULDER]  = quat.IDENTITY.copy()
    T_POSE_REFERENCES[SensorID.R_SHOULDER]  = quat.IDENTITY.copy()

    def __init__(self, accumulation_frames: int = 60):
        self.accumulation_frames = accumulation_frames
        self._buffers: Dict[str, List[np.ndarray]] = {}
        self._frame_count = 0
        self.is_complete = False
        self._result: Optional[CalibrationData] = None

    def reset(self):
        self._buffers.clear()
        self._frame_count = 0
        self.is_complete = False
        self._result = None

    def add_frame(self, sensor_orientations: Dict[str, np.ndarray]) -> bool:
        """
        Feed one frame of sensor world-orientations.
        Returns True when enough frames have been collected and calibration
        can be finalised via compute().
        """
        if self.is_complete:
            return True
        for sid, q in sensor_orientations.items():
            self._buffers.setdefault(sid, []).append(quat.normalize(q))
        self._frame_count += 1
        return self._frame_count >= self.accumulation_frames

    def compute(self, pose_name: str = "tpose", timestamp: float = 0.0) -> CalibrationData:
        """Average accumulated orientations and build CalibrationData."""
        offsets: Dict[str, np.ndarray] = {}

        for sid, qs in self._buffers.items():
            avg = _average_quaternions(qs)
            ref = self.T_POSE_REFERENCES.get(sid, quat.IDENTITY)
            # offset such that: q_calibrated = inv(offset) * q_world = ref
            # → offset = q_world_avg * inv(ref)
            offsets[sid] = quat.normalize(quat.multiply(avg, quat.inverse(ref)))

        # Forward vector from pelvis offset
        pelvis_offset = offsets.get(SensorID.PELVIS, quat.IDENTITY)
        forward = quat.rotate_vector(pelvis_offset, np.array([0.0, 0.0, 1.0]))
        forward[1] = 0.0
        fn = np.linalg.norm(forward)
        forward = forward / fn if fn > 1e-6 else np.array([0.0, 0.0, 1.0])

        self._result = CalibrationData(
            pose_name=pose_name,
            offsets=offsets,
            forward=forward,
            timestamp=timestamp,
        )
        self.is_complete = True
        return self._result

    @property
    def result(self) -> Optional[CalibrationData]:
        return self._result

    @staticmethod
    def from_identity(sensor_ids: List[str]) -> CalibrationData:
        """Return a no-op calibration (identity offsets)."""
        return CalibrationData(
            pose_name="identity",
            offsets={sid: quat.IDENTITY.copy() for sid in sensor_ids},
            forward=np.array([0.0, 0.0, 1.0]),
        )


def _average_quaternions(qs: List[np.ndarray]) -> np.ndarray:
    """Simple quaternion averaging via eigen-decomposition of the outer-product sum."""
    M = np.zeros((4, 4))
    ref = qs[0]
    for q in qs:
        if np.dot(q, ref) < 0:
            q = -q
        M += np.outer(q, q)
    M /= len(qs)
    eigvals, eigvecs = np.linalg.eigh(M)
    return quat.normalize(eigvecs[:, np.argmax(eigvals)])
