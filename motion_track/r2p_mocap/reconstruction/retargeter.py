"""
Sensor-to-skeleton retargeter.

Converts global sensor orientations (post-Madgwick, post-calibration) into
local bone rotations that drive the humanoid FK rig.

Strategy
--------
1.  Apply calibration offset to each sensor's world orientation.
2.  Map each sensor directly to its corresponding bone.
3.  Compute local rotation = inv(parent_world) * child_world.
4.  Fill unsensed bones procedurally:
      - Spine bones → interpolate between PELVIS and CHEST
      - Arm/shoulder bones → follow CHEST with a rest-pose offset
      - NECK → interpolate between CHEST and HEAD
      - TOE bones → follow FOOT orientation
"""

from __future__ import annotations
from typing import Dict, Optional
import numpy as np

from ..core.skeleton import Bone, BONE_PARENTS, BONE_OFFSETS
from ..core import quaternion as quat
from ..core.config import SensorID
from ..calibration.calibrator import CalibrationData


# Sensor → primary driven bone
_SENSOR_TO_BONE: Dict[str, Bone] = {
    SensorID.PELVIS:      Bone.PELVIS,
    SensorID.CHEST:       Bone.CHEST,
    SensorID.L_THIGH:     Bone.L_THIGH,
    SensorID.R_THIGH:     Bone.R_THIGH,
    SensorID.L_SHIN:      Bone.L_SHIN,
    SensorID.R_SHIN:      Bone.R_SHIN,
    SensorID.HEAD:        Bone.HEAD,
    SensorID.L_SHOULDER:  Bone.L_SHOULDER,
    SensorID.R_SHOULDER:  Bone.R_SHOULDER,
    SensorID.L_UPPER_ARM: Bone.L_UPPER_ARM,
    SensorID.R_UPPER_ARM: Bone.R_UPPER_ARM,
    SensorID.L_FOREARM:   Bone.L_FOREARM,
    SensorID.R_FOREARM:   Bone.R_FOREARM,
    SensorID.L_FOOT:      Bone.L_FOOT,
    SensorID.R_FOOT:      Bone.R_FOOT,
}


class SensorToSkeletonRetargeter:

    def __init__(self, calibration: Optional[CalibrationData] = None):
        self.calibration = calibration

    def retarget(
        self,
        sensor_orientations: Dict[str, np.ndarray],
    ) -> Dict[Bone, np.ndarray]:
        """
        Main entry point.
        sensor_orientations: sensor_id → world-frame quaternion from Madgwick.
        Returns: Bone → local rotation quaternion for the FK rig.
        """
        # 1. Apply calibration
        world_bone: Dict[Bone, np.ndarray] = {}
        for sid, q_world in sensor_orientations.items():
            bone = _SENSOR_TO_BONE.get(sid)
            if bone is None:
                continue
            if self.calibration:
                q_world = self.calibration.apply(sid, q_world)
            world_bone[bone] = quat.normalize(q_world)

        # 2. Fill unsensed bones
        world_bone = self._fill_unsensed(world_bone)

        # 3. Convert world → local rotations
        return self._world_to_local(world_bone)

    # ------------------------------------------------------------------
    def _fill_unsensed(
        self, wb: Dict[Bone, np.ndarray]
    ) -> Dict[Bone, np.ndarray]:
        """Procedurally infer bones that have no direct sensor."""

        pelvis = wb.get(Bone.PELVIS, quat.IDENTITY)
        chest  = wb.get(Bone.CHEST,  quat.IDENTITY)

        # Spine chain: even distribution between pelvis and chest
        for frac, bone in [(0.33, Bone.SPINE_1), (0.66, Bone.SPINE_2)]:
            if bone not in wb:
                wb[bone] = quat.slerp(pelvis, chest, frac)

        # Neck: midway between CHEST and HEAD
        head = wb.get(Bone.HEAD, chest)
        if Bone.NECK not in wb:
            wb[Bone.NECK] = quat.slerp(chest, head, 0.5)
        if Bone.HEAD not in wb:
            wb[Bone.HEAD] = chest

        # Shoulders follow CHEST
        for bone in (Bone.L_SHOULDER, Bone.R_SHOULDER):
            if bone not in wb:
                wb[bone] = chest.copy()

        # Upper arms follow their shoulder (or CHEST) — rest pose = hanging down
        for bone, parent in [
            (Bone.L_UPPER_ARM, Bone.L_SHOULDER),
            (Bone.R_UPPER_ARM, Bone.R_SHOULDER),
        ]:
            if bone not in wb:
                wb[bone] = wb.get(parent, chest).copy()

        # Forearms follow upper arms
        for bone, parent in [
            (Bone.L_FOREARM, Bone.L_UPPER_ARM),
            (Bone.R_FOREARM, Bone.R_UPPER_ARM),
        ]:
            if bone not in wb:
                wb[bone] = wb.get(parent, chest).copy()

        # Hands follow forearms
        for bone, parent in [
            (Bone.L_HAND, Bone.L_FOREARM),
            (Bone.R_HAND, Bone.R_FOREARM),
        ]:
            if bone not in wb:
                wb[bone] = wb.get(parent, chest).copy()

        # Feet follow shins when no foot sensor
        for bone, shin in [(Bone.L_FOOT, Bone.L_SHIN), (Bone.R_FOOT, Bone.R_SHIN)]:
            if bone not in wb:
                wb[bone] = wb.get(shin, quat.IDENTITY).copy()

        # Toes follow feet
        for bone, foot in [(Bone.L_TOE, Bone.L_FOOT), (Bone.R_TOE, Bone.R_FOOT)]:
            if bone not in wb:
                wb[bone] = wb.get(foot, quat.IDENTITY).copy()

        return wb

    # ------------------------------------------------------------------
    @staticmethod
    def _world_to_local(
        wb: Dict[Bone, np.ndarray],
    ) -> Dict[Bone, np.ndarray]:
        local: Dict[Bone, np.ndarray] = {}
        for bone in Bone:
            q_world = wb.get(bone, quat.IDENTITY)
            parent  = BONE_PARENTS[bone]
            if parent is None:
                local[bone] = q_world
            else:
                q_parent = wb.get(parent, quat.IDENTITY)
                local[bone] = quat.relative_rotation(q_parent, q_world)
        return local
