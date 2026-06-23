"""
retargeting.py  —  Motion Retargeting & Output Formatting
==========================================================

Converts the R2P skeleton's world-space joint positions and world-space
bone rotations into the LOCAL parent-relative rotations expected by
animation rigs in Unity, Unreal Engine, and Blender.

Why retargeting is non-trivial
-------------------------------
The R2P FK solver works in world space for simplicity and stability.
Animation engines expect PARENT-RELATIVE quaternions — each bone's rotation
is expressed relative to its parent bone's orientation.

Additionally, each target engine has a different:
  - Coordinate system convention (Y-up vs Z-up)
  - Bone axis convention (which local axis points "along" the bone)
  - Quaternion component order ([w,x,y,z] vs [x,y,z,w])

This module handles the conversions so downstream consumers receive
correctly-formatted data without needing to know R2P internals.

Output formats
--------------
  'unity'   — Y-up, Z-forward, quaternions [x,y,z,w], Unity humanoid mapping
  'unreal'  — Z-up, X-forward, quaternions [x,y,z,w], Unreal skeleton bone names
  'blender' — Z-up, -Y-forward, quaternions [w,x,y,z], Blender armature format
  'webgl'   — Y-up, -Z-forward, quaternions [w,x,y,z] (same as R2P internal)
  'raw'     — R2P internal format, no conversion

Sensor-to-bone local transform
--------------------------------
Given:
  q_parent_world = world-space rotation of the parent bone
  q_child_world  = world-space rotation of the child bone (from sensor or FK)

Local rotation = conjugate(q_parent_world) * q_child_world

This produces the rotation you'd apply TO the parent frame to arrive at the
child frame — exactly what Unity's Transform.localRotation expects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import UNITY_BONE_MAP, R2P_TO_UNITY


# ── math ─────────────────────────────────────────────────────────────────────

def _norm(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-9 else np.array([1.0, 0, 0, 0])


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1,x1,y1,z1 = a;  w2,x2,y2,z2 = b
    return np.array([
        w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
        w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2,
    ])


def _qconj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


# ── coordinate-system converters ──────────────────────────────────────────────

def _yup_to_zup(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion from Y-up (R2P/Unity) to Z-up (Unreal/Blender) space.
    Rotation: -90° around X axis applied to the orientation.
    """
    rot_x_neg90 = np.array([0.7071068, -0.7071068, 0.0, 0.0])  # [w, x, y, z]
    return _norm(_qmul(rot_x_neg90, q))


def _yup_to_blender(q: np.ndarray) -> np.ndarray:
    """Blender uses Z-up, -Y forward."""
    rot = np.array([0.7071068, -0.7071068, 0.0, 0.0])
    return _norm(_qmul(rot, q))


# ── bone hierarchy for parent-relative solving ────────────────────────────────

# Maps each bone to its parent bone ID (R2P naming)
BONE_PARENT: Dict[str, Optional[str]] = {
    'pelvis':      None,
    'lumbar':      'pelvis',
    'chest':       'lumbar',
    'neck':        'chest',
    'head':        'neck',
    'head_top':    'head',
    'hip_l':       'pelvis',
    'thigh_l':     'hip_l',
    'shin_l':      'thigh_l',
    'ankle_l':     'shin_l',
    'foot_l':      'ankle_l',
    'hip_r':       'pelvis',
    'thigh_r':     'hip_r',
    'shin_r':      'thigh_r',
    'ankle_r':     'shin_r',
    'foot_r':      'ankle_r',
    'shoulder_l':  'chest',
    'upper_arm_l': 'shoulder_l',
    'elbow_l':     'upper_arm_l',
    'forearm_l':   'elbow_l',
    'wrist_l':     'forearm_l',
    'hand_l':      'wrist_l',
    'shoulder_r':  'chest',
    'upper_arm_r': 'shoulder_r',
    'elbow_r':     'upper_arm_r',
    'forearm_r':   'elbow_r',
    'wrist_r':     'forearm_r',
    'hand_r':      'wrist_r',
}


# ── main retargeter ───────────────────────────────────────────────────────────

class Retargeter:
    """
    Converts R2P world-space bone rotations to engine-specific local rotations.

    Usage
    -----
    retargeter = Retargeter(target='unity')
    output = retargeter.retarget(world_rotations, root_pos, root_rot)

    Parameters
    ----------
    target : str
        One of 'unity', 'unreal', 'blender', 'webgl', 'raw'.
    """

    def __init__(self, target: str = 'webgl'):
        self._target = target

    def retarget(
        self,
        world_rotations: Dict[str, np.ndarray],   # {bone_id: [w,x,y,z]} world space
        root_pos: np.ndarray,
        root_rot: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Returns a dict of local bone rotations in the target engine format.
        """
        local_rots = self._solve_local(world_rotations)

        if self._target == 'raw' or self._target == 'webgl':
            return self._format_raw(local_rots, root_pos, root_rot)
        elif self._target == 'unity':
            return self._format_unity(local_rots, root_pos, root_rot)
        elif self._target == 'unreal':
            return self._format_unreal(local_rots, root_pos, root_rot)
        elif self._target == 'blender':
            return self._format_blender(local_rots, root_pos, root_rot)
        return self._format_raw(local_rots, root_pos, root_rot)

    # ── local rotation solve ──────────────────────────────────────────────────

    def _solve_local(
        self, world_rots: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        For each bone: local_rot = conj(parent_world_rot) * child_world_rot
        Bones without a parent in world_rots fall back to identity.
        """
        local: Dict[str, np.ndarray] = {}
        I = np.array([1.0, 0, 0, 0])

        for bone, parent in BONE_PARENT.items():
            q_child  = _norm(world_rots.get(bone, I))
            if parent is None:
                # Root bone: world rotation IS the local rotation
                local[bone] = q_child
            else:
                q_parent = _norm(world_rots.get(parent, I))
                local[bone] = _norm(_qmul(_qconj(q_parent), q_child))
        return local

    # ── format helpers ────────────────────────────────────────────────────────

    def _format_raw(
        self, local: Dict[str, np.ndarray],
        root_pos: np.ndarray, root_rot: np.ndarray
    ) -> Dict[str, Any]:
        """R2P format: [w,x,y,z], Y-up."""
        bones: Dict[str, Dict] = {}
        for bone, q in local.items():
            bones[bone] = {
                'rotation': [round(float(v), 5) for v in q],
            }
        return {
            'format':   'webgl',
            'coord':    'y_up',
            'root_pos': [round(float(v), 4) for v in root_pos],
            'root_rot': [round(float(v), 5) for v in root_rot],
            'bones':    bones,
        }

    def _format_unity(
        self, local: Dict[str, np.ndarray],
        root_pos: np.ndarray, root_rot: np.ndarray
    ) -> Dict[str, Any]:
        """
        Unity humanoid format.
        - Quaternion order: [x, y, z, w]
        - Coordinate: Y-up, Z-forward (same as R2P, no conversion needed)
        - Bone names from UNITY_BONE_MAP
        """
        bones: Dict[str, Dict] = {}
        for r2p_id, q in local.items():
            unity_name = R2P_TO_UNITY.get(r2p_id)
            if unity_name is None:
                continue
            # Unity quat order: x, y, z, w
            bones[unity_name] = {
                'localRotation': {
                    'x': round(float(q[1]), 5),
                    'y': round(float(q[2]), 5),
                    'z': round(float(q[3]), 5),
                    'w': round(float(q[0]), 5),
                }
            }
        # Root (Hips) position
        hp = [round(float(v), 4) for v in root_pos]
        return {
            'format': 'unity',
            'coord':  'y_up_z_fwd',
            'humanoid': {
                'Hips': {
                    'position': {'x': hp[0], 'y': hp[1], 'z': hp[2]},
                    **bones.get('Hips', {}),
                },
                **{k: v for k, v in bones.items() if k != 'Hips'},
            }
        }

    def _format_unreal(
        self, local: Dict[str, np.ndarray],
        root_pos: np.ndarray, root_rot: np.ndarray
    ) -> Dict[str, Any]:
        """
        Unreal Engine format.
        - Coordinate: Z-up, X-forward (left-handed in UE)
        - Quaternion order: [x, y, z, w]
        - Position scale: centimetres (× 100)
        """
        bones: Dict[str, Dict] = {}
        for bone, q in local.items():
            q_ue = _yup_to_zup(q)
            bones[bone] = {
                'rotation': [round(float(q_ue[1]),5), round(float(q_ue[2]),5),
                             round(float(q_ue[3]),5), round(float(q_ue[0]),5)]
            }
        rp_cm = [round(float(v)*100, 2) for v in root_pos]
        rr_ue = _yup_to_zup(root_rot)
        return {
            'format':   'unreal',
            'coord':    'z_up_x_fwd',
            'root_pos': {'x': rp_cm[0], 'y': rp_cm[2], 'z': rp_cm[1]},
            'root_rot': [round(float(rr_ue[1]),5), round(float(rr_ue[2]),5),
                         round(float(rr_ue[3]),5), round(float(rr_ue[0]),5)],
            'bones':    bones,
        }

    def _format_blender(
        self, local: Dict[str, np.ndarray],
        root_pos: np.ndarray, root_rot: np.ndarray
    ) -> Dict[str, Any]:
        """
        Blender armature format.
        - Coordinate: Z-up, -Y-forward
        - Quaternion order: [w, x, y, z] (same as R2P, but coordinate differs)
        """
        bones: Dict[str, Dict] = {}
        for bone, q in local.items():
            q_bl = _yup_to_blender(q)
            bones[bone] = {
                'rotation_quaternion': [round(float(v), 5) for v in q_bl]
            }
        rr_bl = _yup_to_blender(root_rot)
        rp_bl = [round(float(root_pos[0]),4), round(float(root_pos[2]),4), round(float(root_pos[1]),4)]
        return {
            'format':   'blender',
            'coord':    'z_up_neg_y_fwd',
            'root_pos': rp_bl,
            'root_rot': [round(float(v),5) for v in rr_bl],
            'bones':    bones,
        }