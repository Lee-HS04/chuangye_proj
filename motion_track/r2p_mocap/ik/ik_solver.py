"""
Two-bone analytical IK + foot grounding.

Analytical two-bone IK solves the exact hip→knee→ankle chain given a target
ankle position.  Foot grounding pins the ankle to max(FK ankle, ground_y)
and re-solves the leg.
"""

from __future__ import annotations
from typing import Optional, Dict, Tuple
import numpy as np

from ..core.skeleton import Bone, BoneTransform, SkeletonPose, BONE_OFFSETS
from ..core import quaternion as quat


_EPS = 1e-8


class TwoBoneIK:
    """
    Analytical IK for a root → mid → end chain.
    All inputs / outputs are in world space.
    """

    def solve(
        self,
        root_pos:  np.ndarray,   # hip
        mid_pos:   np.ndarray,   # knee
        end_pos:   np.ndarray,   # ankle (current FK)
        target:    np.ndarray,   # desired ankle world position
        pole_vec:  Optional[np.ndarray] = None,  # hint for knee bend direction
        upper_len: Optional[float] = None,
        lower_len: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (new_root_pos, new_mid_pos, new_end_pos).
        root_pos is fixed; mid and end are repositioned.
        """
        if upper_len is None:
            upper_len = float(np.linalg.norm(mid_pos - root_pos))
        if lower_len is None:
            lower_len = float(np.linalg.norm(end_pos - mid_pos))
        if pole_vec is None:
            pole_vec = np.array([0.0, 0.0, -1.0])  # knee bends forward

        to_target = target - root_pos
        dist = float(np.linalg.norm(to_target))
        dist = max(dist, _EPS)

        total = upper_len + lower_len
        dist = min(dist, total * 0.9999)  # prevent full extension singularity

        # Law of cosines — angle at root
        cos_a = (upper_len**2 + dist**2 - lower_len**2) / (2 * upper_len * dist)
        cos_a = float(np.clip(cos_a, -1.0, 1.0))
        angle_a = np.arccos(cos_a)

        dir_to_target = to_target / dist

        # Build a perpendicular vector (pole plane)
        perp = pole_vec - np.dot(pole_vec, dir_to_target) * dir_to_target
        perp_n = np.linalg.norm(perp)
        if perp_n < _EPS:
            perp = np.array([0.0, 1.0, 0.0])
            perp -= np.dot(perp, dir_to_target) * dir_to_target
            perp_n = np.linalg.norm(perp)
        perp /= max(perp_n, _EPS)

        # Mid joint position
        mid_dir = dir_to_target * np.cos(angle_a) + perp * np.sin(angle_a)
        new_mid  = root_pos + mid_dir * upper_len
        new_end  = target

        return root_pos.copy(), new_mid, new_end


class FootGroundingIK:
    """
    Applies foot grounding: when FK places the ankle below ground, pin it to
    ground level, re-solve the leg chain, and adjust the pelvis height.
    """

    def __init__(self):
        self._ik = TwoBoneIK()

    def apply(
        self,
        pose: SkeletonPose,
        ground_y: float = 0.0,
        left_contact:  bool = True,
        right_contact: bool = True,
    ) -> SkeletonPose:
        """Modify pose in-place and return it."""
        if left_contact:
            self._ground_leg(
                pose,
                hip=Bone.L_THIGH, knee=Bone.L_SHIN, ankle=Bone.L_FOOT,
                ground_y=ground_y,
                pole=np.array([0.0, 0.0, 1.0]),
            )
        if right_contact:
            self._ground_leg(
                pose,
                hip=Bone.R_THIGH, knee=Bone.R_SHIN, ankle=Bone.R_FOOT,
                ground_y=ground_y,
                pole=np.array([0.0, 0.0, 1.0]),
            )
        return pose

    def _ground_leg(
        self,
        pose: SkeletonPose,
        hip: Bone, knee: Bone, ankle: Bone,
        ground_y: float,
        pole: np.ndarray,
    ):
        hip_pos   = pose.bones[hip].position
        knee_pos  = pose.bones[knee].position
        ankle_pos = pose.bones[ankle].position

        if ankle_pos[1] >= ground_y:
            return  # already above ground, no correction needed

        target = ankle_pos.copy()
        target[1] = ground_y

        _, new_knee, new_ankle = self._ik.solve(
            root_pos=hip_pos,
            mid_pos=knee_pos,
            end_pos=ankle_pos,
            target=target,
            pole_vec=pole,
        )

        # Compute corrected rotations from new positions
        # Hip rotation: align bone vector to point toward new_knee
        self._align_bone(pose, hip,   hip_pos,   new_knee)
        self._align_bone(pose, knee,  new_knee,  new_ankle)
        pose.bones[knee].position  = new_knee
        pose.bones[ankle].position = new_ankle

    @staticmethod
    def _align_bone(pose: SkeletonPose, bone: Bone, from_pos: np.ndarray, to_pos: np.ndarray):
        """Rotate bone so its +Y axis points from from_pos toward to_pos."""
        desired_dir = to_pos - from_pos
        dn = np.linalg.norm(desired_dir)
        if dn < _EPS:
            return
        desired_dir /= dn
        # Current bone Y axis in world space
        current_rot = pose.bones[bone].rotation
        current_y = quat.rotate_vector(current_rot, np.array([0.0, -1.0, 0.0]))  # bone points -Y
        cross = np.cross(current_y, desired_dir)
        cn = np.linalg.norm(cross)
        if cn < _EPS:
            return
        angle = np.arcsin(np.clip(cn, -1.0, 1.0))
        axis = cross / cn
        correction = quat.from_axis_angle(axis, angle)
        pose.bones[bone].rotation = quat.normalize(quat.multiply(correction, current_rot))
