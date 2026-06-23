"""
Humanoid skeletal hierarchy for the R2P mocap engine.

Coordinate system:  Y-up  |  X-right  |  Z-forward  (right-handed)
Bone offsets are in the parent's local T-pose space.
"""

from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np
from . import quaternion as quat


# ── Bone names ────────────────────────────────────────────────────────────────

class Bone(Enum):
    PELVIS      = "pelvis"
    SPINE_1     = "spine_1"
    SPINE_2     = "spine_2"
    CHEST       = "chest"
    NECK        = "neck"
    HEAD        = "head"
    L_SHOULDER  = "l_shoulder"
    L_UPPER_ARM = "l_upper_arm"
    L_FOREARM   = "l_forearm"
    L_HAND      = "l_hand"
    R_SHOULDER  = "r_shoulder"
    R_UPPER_ARM = "r_upper_arm"
    R_FOREARM   = "r_forearm"
    R_HAND      = "r_hand"
    L_THIGH     = "l_thigh"
    L_SHIN      = "l_shin"
    L_FOOT      = "l_foot"
    L_TOE       = "l_toe"
    R_THIGH     = "r_thigh"
    R_SHIN      = "r_shin"
    R_FOOT      = "r_foot"
    R_TOE       = "r_toe"


# ── Parent hierarchy (None = root) ────────────────────────────────────────────

BONE_PARENTS: Dict[Bone, Optional[Bone]] = {
    Bone.PELVIS:      None,
    Bone.SPINE_1:     Bone.PELVIS,
    Bone.SPINE_2:     Bone.SPINE_1,
    Bone.CHEST:       Bone.SPINE_2,
    Bone.NECK:        Bone.CHEST,
    Bone.HEAD:        Bone.NECK,
    Bone.L_SHOULDER:  Bone.CHEST,
    Bone.L_UPPER_ARM: Bone.L_SHOULDER,
    Bone.L_FOREARM:   Bone.L_UPPER_ARM,
    Bone.L_HAND:      Bone.L_FOREARM,
    Bone.R_SHOULDER:  Bone.CHEST,
    Bone.R_UPPER_ARM: Bone.R_SHOULDER,
    Bone.R_FOREARM:   Bone.R_UPPER_ARM,
    Bone.R_HAND:      Bone.R_FOREARM,
    Bone.L_THIGH:     Bone.PELVIS,
    Bone.L_SHIN:      Bone.L_THIGH,
    Bone.L_FOOT:      Bone.L_SHIN,
    Bone.L_TOE:       Bone.L_FOOT,
    Bone.R_THIGH:     Bone.PELVIS,
    Bone.R_SHIN:      Bone.R_THIGH,
    Bone.R_FOOT:      Bone.R_SHIN,
    Bone.R_TOE:       Bone.R_FOOT,
}

# ── T-pose local offsets from parent joint (metres) ───────────────────────────
# Scaled for a 1.75 m subject.  Positive Y = up, positive X = right, +Z = fwd.

_s = 1.0  # scale — multiply everything if body_height changes
BONE_OFFSETS: Dict[Bone, np.ndarray] = {
    Bone.PELVIS:      np.array([ 0.00,  0.00,  0.00]),
    Bone.SPINE_1:     np.array([ 0.00,  0.12,  0.00]),
    Bone.SPINE_2:     np.array([ 0.00,  0.13,  0.00]),
    Bone.CHEST:       np.array([ 0.00,  0.11,  0.00]),
    Bone.NECK:        np.array([ 0.00,  0.20,  0.00]),
    Bone.HEAD:        np.array([ 0.00,  0.12,  0.00]),
    Bone.L_SHOULDER:  np.array([-0.19,  0.15,  0.00]),
    Bone.L_UPPER_ARM: np.array([-0.07,  0.00,  0.00]),
    Bone.L_FOREARM:   np.array([-0.28,  0.00,  0.00]),
    Bone.L_HAND:      np.array([-0.26,  0.00,  0.00]),
    Bone.R_SHOULDER:  np.array([ 0.19,  0.15,  0.00]),
    Bone.R_UPPER_ARM: np.array([ 0.07,  0.00,  0.00]),
    Bone.R_FOREARM:   np.array([ 0.28,  0.00,  0.00]),
    Bone.R_HAND:      np.array([ 0.26,  0.00,  0.00]),
    Bone.L_THIGH:     np.array([-0.09, -0.07,  0.00]),
    Bone.L_SHIN:      np.array([ 0.00, -0.43,  0.00]),
    Bone.L_FOOT:      np.array([ 0.00, -0.40,  0.00]),
    Bone.L_TOE:       np.array([ 0.00, -0.04,  0.15]),
    Bone.R_THIGH:     np.array([ 0.09, -0.07,  0.00]),
    Bone.R_SHIN:      np.array([ 0.00, -0.43,  0.00]),
    Bone.R_FOOT:      np.array([ 0.00, -0.40,  0.00]),
    Bone.R_TOE:       np.array([ 0.00, -0.04,  0.15]),
}

# Leg length used for pelvis-height estimation (pelvis→ankle)
LEG_LENGTH = 0.07 + 0.43 + 0.40  # = 0.90 m


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class BoneTransform:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: quat.IDENTITY.copy())


@dataclass
class SkeletonPose:
    bones:           Dict[Bone, BoneTransform]  = field(default_factory=dict)
    local_rotations: Dict[Bone, np.ndarray]     = field(default_factory=dict)
    root_position:   np.ndarray                 = field(default_factory=lambda: np.array([0.0, LEG_LENGTH, 0.0]))
    timestamp:       float                      = 0.0
    frame_id:        int                        = 0

    def world_position(self, bone: Bone) -> np.ndarray:
        return self.bones[bone].position

    def world_rotation(self, bone: Bone) -> np.ndarray:
        return self.bones[bone].rotation


# ── Skeleton class (FK) ───────────────────────────────────────────────────────

class Skeleton:
    """Humanoid rig — runs forward kinematics from local joint rotations."""

    def __init__(self, body_height: float = 1.75):
        self.body_height = body_height
        self._scale = body_height / 1.75
        self._offsets = {b: v * self._scale for b, v in BONE_OFFSETS.items()}
        self._ordered = self._topological_sort()

    def _topological_sort(self) -> List[Bone]:
        order: List[Bone] = []
        visited: set = set()

        def visit(b: Bone):
            if b in visited:
                return
            p = BONE_PARENTS[b]
            if p is not None:
                visit(p)
            visited.add(b)
            order.append(b)

        for b in Bone:
            visit(b)
        return order

    def forward_kinematics(
        self,
        local_rotations: Dict[Bone, np.ndarray],
        root_position: Optional[np.ndarray] = None,
    ) -> SkeletonPose:
        if root_position is None:
            root_position = np.array([0.0, LEG_LENGTH * self._scale, 0.0])

        pose = SkeletonPose(root_position=root_position.copy())

        for bone in self._ordered:
            local_rot = quat.normalize(local_rotations.get(bone, quat.IDENTITY))
            parent = BONE_PARENTS[bone]

            if parent is None:
                pose.bones[bone] = BoneTransform(
                    position=root_position.copy(),
                    rotation=local_rot,
                )
            else:
                pt = pose.bones[parent]
                offset = self._offsets[bone]
                world_pos = pt.position + quat.rotate_vector(pt.rotation, offset)
                world_rot = quat.normalize(quat.multiply(pt.rotation, local_rot))
                pose.bones[bone] = BoneTransform(position=world_pos, rotation=world_rot)

            pose.local_rotations[bone] = local_rot

        return pose

    def get_children(self, bone: Bone) -> List[Bone]:
        return [b for b, p in BONE_PARENTS.items() if p == bone]

    def get_bone_length(self, bone: Bone) -> float:
        return float(np.linalg.norm(self._offsets[bone]))
