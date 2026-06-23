"""
Output data structures for the R2P mocap engine.

MotionCapFrame holds one frame of solved skeleton data.
SkeletonOutput accumulates frames and can serialise to JSON / dict.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import numpy as np

from ..core.skeleton import Bone, SkeletonPose
from ..core import quaternion as quat
from ..biomechanics.foot_contact import FootState, GaitPhase


@dataclass
class BoneData:
    position:       List[float]  # world xyz
    rotation:       List[float]  # quaternion [w, x, y, z]
    local_rotation: List[float]  # local quaternion


@dataclass
class MotionCapFrame:
    frame_id:       int
    timestamp:      float
    bones:          Dict[str, BoneData]  = field(default_factory=dict)
    root_position:  List[float]          = field(default_factory=lambda: [0.0, 0.9, 0.0])
    root_velocity:  List[float]          = field(default_factory=lambda: [0.0, 0.0, 0.0])
    motion_state:   str                  = "idle"
    foot_contact:   Dict[str, bool]      = field(default_factory=dict)
    tracking_conf:  float                = 1.0

    @staticmethod
    def from_skeleton_pose(
        pose:       SkeletonPose,
        root_vel:   Optional[np.ndarray]      = None,
        foot_states: Optional[Dict[str, FootState]] = None,
        motion_state: str = "unknown",
    ) -> "MotionCapFrame":
        bones = {}
        for bone, bt in pose.bones.items():
            lr = pose.local_rotations.get(bone, quat.IDENTITY)
            bones[bone.value] = BoneData(
                position=bt.position.tolist(),
                rotation=bt.rotation.tolist(),
                local_rotation=lr.tolist(),
            )
        foot_contact = {}
        if foot_states:
            for side, fs in foot_states.items():
                foot_contact[side] = fs.contact

        return MotionCapFrame(
            frame_id=pose.frame_id,
            timestamp=pose.timestamp,
            bones=bones,
            root_position=pose.root_position.tolist(),
            root_velocity=(root_vel.tolist() if root_vel is not None else [0.0, 0.0, 0.0]),
            motion_state=motion_state,
            foot_contact=foot_contact,
        )

    def to_dict(self) -> dict:
        return {
            "frame_id":      self.frame_id,
            "timestamp":     self.timestamp,
            "root_position": self.root_position,
            "root_velocity": self.root_velocity,
            "motion_state":  self.motion_state,
            "foot_contact":  self.foot_contact,
            "bones": {
                name: {
                    "position":       bd.position,
                    "rotation":       bd.rotation,
                    "local_rotation": bd.local_rotation,
                }
                for name, bd in self.bones.items()
            },
        }


class SkeletonOutput:
    """Accumulates frames; serialises to JSON or returns as list of dicts."""

    def __init__(self, sample_rate: float = 100.0, body_height: float = 1.75):
        self.sample_rate = sample_rate
        self.body_height = body_height
        self._frames: List[MotionCapFrame] = []

    def add_frame(self, frame: MotionCapFrame):
        self._frames.append(frame)

    def to_json(self, indent: int = 2) -> str:
        payload = {
            "metadata": {
                "sample_rate": self.sample_rate,
                "body_height": self.body_height,
                "frame_count": len(self._frames),
                "duration":    len(self._frames) / max(self.sample_rate, 1),
            },
            "frames": [f.to_dict() for f in self._frames],
        }
        return json.dumps(payload, indent=indent)

    def save(self, path: str):
        with open(path, "w") as fh:
            fh.write(self.to_json())

    def clear(self):
        self._frames.clear()

    @property
    def frames(self) -> List[MotionCapFrame]:
        return self._frames

    @property
    def frame_count(self) -> int:
        return len(self._frames)
