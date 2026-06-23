"""
BVH (Biovision Hierarchy) exporter.

Outputs a standard .bvh file compatible with Blender, Maya, MotionBuilder,
and Unity's BVH import.  Uses ZXY Euler channels, Y-up, Z-forward.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import numpy as np

from ..core.skeleton import Bone, BONE_PARENTS, BONE_OFFSETS
from ..core import quaternion as quat
from .skeleton_output import SkeletonOutput, MotionCapFrame


# Bones that appear in the BVH hierarchy (subset — BVH doesn't need toes for basic rigs)
_BVH_BONES: List[Bone] = [
    Bone.PELVIS,
    Bone.SPINE_1, Bone.SPINE_2, Bone.CHEST,
    Bone.NECK, Bone.HEAD,
    Bone.L_SHOULDER, Bone.L_UPPER_ARM, Bone.L_FOREARM, Bone.L_HAND,
    Bone.R_SHOULDER, Bone.R_UPPER_ARM, Bone.R_FOREARM, Bone.R_HAND,
    Bone.L_THIGH, Bone.L_SHIN, Bone.L_FOOT,
    Bone.R_THIGH, Bone.R_SHIN, Bone.R_FOOT,
]

_CHILDREN: Dict[Bone, List[Bone]] = {b: [] for b in Bone}
for _b, _p in BONE_PARENTS.items():
    if _p is not None:
        _CHILDREN[_p].append(_b)


class BVHExporter:
    """Convert SkeletonOutput to BVH text format."""

    CM = 100.0  # metres → centimetres conversion

    def export(
        self,
        output:       SkeletonOutput,
        body_height:  float = 1.75,
        scale:        float = 1.0,
    ) -> str:
        scale_cm = scale * self.CM
        lines: List[str] = ["HIERARCHY"]
        self._write_joint(lines, Bone.PELVIS, 0, scale_cm, is_root=True)
        lines.append("")
        lines.append("MOTION")
        lines.append(f"Frames: {output.frame_count}")
        lines.append(f"Frame Time: {1.0 / max(output.sample_rate, 1):.6f}")
        for frame in output.frames:
            lines.append(self._frame_data(frame, scale_cm))
        return "\n".join(lines)

    def save(self, output: SkeletonOutput, path: str, **kwargs):
        with open(path, "w") as fh:
            fh.write(self.export(output, **kwargs))

    # ------------------------------------------------------------------
    def _write_joint(
        self,
        lines: List[str],
        bone:  Bone,
        depth: int,
        scale_cm: float,
        is_root: bool = False,
    ):
        indent = "  " * depth
        name = bone.value.upper()
        offset = BONE_OFFSETS[bone] * scale_cm

        if is_root:
            lines.append(f"{indent}ROOT {name}")
        else:
            lines.append(f"{indent}JOINT {name}")
        lines.append(f"{indent}{{")
        lines.append(f"{indent}  OFFSET {offset[0]:.4f} {offset[1]:.4f} {offset[2]:.4f}")

        children_in_set = [c for c in _CHILDREN[bone] if c in set(_BVH_BONES)]

        if is_root:
            lines.append(f"{indent}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
        else:
            lines.append(f"{indent}  CHANNELS 3 Zrotation Xrotation Yrotation")

        if children_in_set:
            for child in children_in_set:
                self._write_joint(lines, child, depth + 1, scale_cm)
        else:
            # End site
            tip = BONE_OFFSETS.get(bone, np.zeros(3)) * scale_cm * 0.5
            lines.append(f"{indent}  End Site")
            lines.append(f"{indent}  {{")
            lines.append(f"{indent}    OFFSET {tip[0]:.4f} {tip[1]:.4f} {tip[2]:.4f}")
            lines.append(f"{indent}  }}")

        lines.append(f"{indent}}}")

    # ------------------------------------------------------------------
    def _frame_data(self, frame: MotionCapFrame, scale_cm: float) -> str:
        values: List[float] = []

        # Root translation (pelvis position in cm)
        rp = frame.root_position
        values += [rp[0]*scale_cm, rp[1]*scale_cm, rp[2]*scale_cm]

        for bone in _BVH_BONES:
            bd = frame.bones.get(bone.value)
            if bd is None:
                values += [0.0, 0.0, 0.0]
                continue
            q = np.array(bd.local_rotation)
            # Convert quaternion to ZXY Euler (BVH convention)
            R = quat.to_rotation_matrix(q)
            z, x, y = _mat_to_euler_zxy(R)
            z_deg = np.degrees(z)
            x_deg = np.degrees(x)
            y_deg = np.degrees(y)
            if bone == Bone.PELVIS:
                values += [z_deg, x_deg, y_deg]
            else:
                values += [z_deg, x_deg, y_deg]

        return " ".join(f"{v:.4f}" for v in values)


def _mat_to_euler_zxy(R: np.ndarray) -> Tuple[float, float, float]:
    """Decompose rotation matrix into ZXY Euler angles."""
    x = np.arcsin(np.clip(R[1, 2], -1.0, 1.0))
    if abs(R[1, 2]) < 0.9999:
        z = np.arctan2(-R[0, 2], R[2, 2])
        y = np.arctan2(-R[1, 0], R[1, 1])
    else:
        z = np.arctan2(R[2, 0], R[0, 0])
        y = 0.0
    return z, x, y
