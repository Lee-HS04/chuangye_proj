"""
skeleton.py  —  Hierarchical FK solver + IK foot planting
===========================================================

Solves the full 15-joint humanoid skeleton from IMU quaternions.

Architecture
------------
1. FK PASS — propagate transforms top-down through the hierarchy.
   Each bone's world rotation = parent_world_rotation * local_rotation.
   Each bone's world position = parent_position + parent_rotation * offset.

2. CONSTRAINT PASS — after FK, soft-clamp joint angles to anatomical limits
   using biomechanics.JointConstraints.  The constrained local rotation
   replaces the raw sensor rotation for the next frame.

3. IK FOOT PLANTING — two-bone IK on each leg (hip → knee → ankle) to
   pin the foot to the ground plane during stance phase.

4. CONSUMER ESTIMATION — when only 6 IMUs are live, estimate missing
   upper-body joints using procedural biomechanics (arm swing from
   pelvis/chest dynamics, head from chest + neck heuristic).

Multi-sensor fusion
-------------------
Each bone maps to a primary sensor.  If the sensor is live, its smoothed
quaternion drives the bone rotation.  If it is absent, a fallback strategy
is applied:
  - Spine chain (lumbar, chest): pelvis quaternion with damped twist
  - Head: chest quaternion with stabilization
  - Arms: procedural swing derived from pelvis acceleration
  - Feet: extrapolated from shin orientation

Output
------
computeJoints() returns a dict of {joint_name: [x, y, z]} world positions
for every named joint — identical interface to the browser-side FK so the
viewer can be driven by either.

It also returns a SkeletonState with full rotation, velocity, and analytics
data matching the output specification in the system prompt.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from biomechanics import JointConstraints


# ── math ─────────────────────────────────────────────────────────────────────

def _norm(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-9 else np.array([1.0, 0, 0, 0])


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
        w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2,
    ])


def _qconj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.array([0.0, v[0], v[1], v[2]])
    return _qmul(_qmul(q, qv), _qconj(q))[1:]


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    d = float(np.dot(a, b))
    if d < 0: b = -b; d = -d
    if d > 0.9995: return _norm(a + t*(b-a))
    th = np.arccos(np.clip(d, -1, 1))
    return _norm(np.sin((1-t)*th)/np.sin(th)*a + np.sin(t*th)/np.sin(th)*b)


def _ax_angle(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    h = np.radians(angle_deg) / 2
    s = np.sin(h)
    return _norm(np.array([np.cos(h), axis[0]*s, axis[1]*s, axis[2]*s]))


def _quat_yaw(q: np.ndarray) -> float:
    """Extract yaw angle in degrees."""
    w, x, y, z = q
    return float(np.degrees(np.arctan2(2*(w*y + z*x), 1 - 2*(x*x + y*y))))


def _yaw_only(q: np.ndarray) -> np.ndarray:
    """Return yaw-only quaternion (rotation about world Y)."""
    return _ax_angle(np.array([0.0, 1.0, 0.0]), _quat_yaw(q))


def _tilt_from(q: np.ndarray) -> np.ndarray:
    """Return the tilt component: q with yaw removed."""
    return _norm(_qmul(_qconj(_yaw_only(q)), q))


# ── limb lengths (metres) ─────────────────────────────────────────────────────

LL = dict(
    pelvis_h      = 0.92,    # root height above floor
    pelvis_lumbar = 0.10,
    lumbar_chest  = 0.15,
    chest_neck    = 0.20,
    neck_head     = 0.10,
    head_top      = 0.13,
    hip_offset_x  = 0.105,
    hip_offset_y  = -0.05,
    thigh         = 0.42,
    shin          = 0.40,
    foot          = 0.17,
    clavicle      = 0.175,
    upper_arm     = 0.30,
    forearm       = 0.26,
    hand          = 0.09,
)


# ── sensor → bone mapping ────────────────────────────────────────────────────

# Maps sensor_id to the bone it primarily controls (local rotation).
SENSOR_BONE_MAP = {
    'pelvis':      'pelvis',
    'chest':       'chest',
    'head':        'head',
    'thigh_l':     'thigh_l',
    'thigh_r':     'thigh_r',
    'shin_l':      'shin_l',
    'shin_r':      'shin_r',
    'l_upper_arm': 'upper_arm_l',
    'r_upper_arm': 'upper_arm_r',
    'l_forearm':   'forearm_l',
    'r_forearm':   'forearm_r',
    'l_foot':      'foot_l',
    'r_foot':      'foot_r',
    'l_shoulder':  'shoulder_l',
    'r_shoulder':  'shoulder_r',
}


# ── output types ─────────────────────────────────────────────────────────────

@dataclass
class BoneState:
    world_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    world_rot: np.ndarray = field(default_factory=lambda: np.array([1.0,0,0,0]))
    local_rot: np.ndarray = field(default_factory=lambda: np.array([1.0,0,0,0]))
    velocity:  np.ndarray = field(default_factory=lambda: np.zeros(3))
    is_estimated: bool = False


@dataclass
class SkeletonState:
    joints: Dict[str, np.ndarray]   = field(default_factory=dict)   # {name: [x,y,z]}
    bones:  Dict[str, BoneState]    = field(default_factory=dict)
    root_pos: np.ndarray            = field(default_factory=lambda: np.zeros(3))
    root_rot: np.ndarray            = field(default_factory=lambda: np.array([1.0,0,0,0]))
    movement_state: str             = 'IDLE'
    tracking_confidence: float      = 1.0
    live_sensors: List[str]         = field(default_factory=list)
    timestamp: float                = 0.0

    def to_dict(self) -> dict:
        out = {}
        for name, pos in self.joints.items():
            out[name] = [round(float(v), 4) for v in pos]
        return {
            'joints': out,
            'root_pos': [round(float(v), 4) for v in self.root_pos],
            'root_rot': [round(float(v), 5) for v in self.root_rot],
            'movement_state': self.movement_state,
            'tracking_confidence': round(self.tracking_confidence, 3),
            'live_sensors': self.live_sensors,
            'timestamp': self.timestamp,
        }


# ── main solver ───────────────────────────────────────────────────────────────

class SkeletonSolver:
    """
    Hierarchical FK solver with constraint enforcement, foot IK,
    and consumer-mode upper-body estimation.

    Usage
    -----
    solver = SkeletonSolver()
    state = solver.solve(
        sensor_quats  = {'pelvis': q_pelvis, 'thigh_l': q_tl, ...},
        root_pos      = np.array([x, 0, z]),
        root_y_offset = 0.0,
        live_sensors  = ['pelvis', 'thigh_l', ...],
        dt            = 0.016,
    )
    joints_dict = state.joints   # {joint_name: [x, y, z]}
    """

    def __init__(self, constraint_hardness: float = 0.82, ik_enabled: bool = True):
        self._hardness = constraint_hardness
        self._ik_enabled = ik_enabled
        self._prev_bones: Dict[str, BoneState] = {}
        self._t = 0.0
        # Smoothed procedural arm swing state
        self._arm_swing_phase = 0.0
        self._prev_pelvis_yaw = 0.0
        self._stance_l = True
        self._stance_r = True
        # velocity estimation
        self._prev_joints: Dict[str, np.ndarray] = {}
        self._prev_ts = 0.0

    def solve(
        self,
        sensor_quats: Dict[str, np.ndarray],
        root_pos: np.ndarray,
        root_y_offset: float,
        live_sensors: List[str],
        dt: float,
        action: str = 'idle',
        cadence_hz: float = 0.0,
    ) -> SkeletonState:
        self._t += dt

        qI = np.array([1.0, 0, 0, 0])
        qp = _norm(sensor_quats.get('pelvis', qI))
        qc = _norm(sensor_quats.get('chest',  qI))

        # Decompose pelvis
        qp_yaw  = _yaw_only(qp)
        qp_tilt = _tilt_from(qp)

        # ── Build local rotations ─────────────────────────────────────────

        local = {}

        # Pelvis = full orientation (heading + tilt)
        local['pelvis'] = qp

        # Spine: if chest sensor live, use it directly; otherwise derive from pelvis
        if 'chest' in live_sensors:
            local['chest'] = qc
        else:
            # Procedurally: chest inherits pelvis with slight independent sway
            local['chest'] = _norm(_slerp(qp, qI, 0.15))

        # Head: head sensor if live, else stabilized chest
        if 'head' in live_sensors:
            local['head'] = _norm(sensor_quats.get('head', qI))
        else:
            # Head tends to lag behind chest rotation, slightly stabilized
            local['head'] = _norm(_slerp(local['chest'], qI, 0.25))

        # Legs — thigh and shin use yaw-only as their orientation base
        # so they stand vertical regardless of forward lean
        for side in ('l', 'r'):
            thigh_id = f'thigh_{side}'
            shin_id  = f'shin_{side}'
            foot_id  = f'l_foot' if side == 'l' else 'r_foot'

            if thigh_id in live_sensors:
                qth = _norm(sensor_quats[thigh_id])
                # Constrain hip (thigh) joint
                q_local_th = _norm(_qmul(_qconj(qp_yaw), qth))
                q_local_th = JointConstraints.hip(q_local_th, side=side, hardness=self._hardness)
                local[f'thigh_{side}'] = _norm(_qmul(qp_yaw, q_local_th))
            else:
                local[f'thigh_{side}'] = qp_yaw

            if shin_id in live_sensors:
                qsh = _norm(sensor_quats[shin_id])
                q_parent_th = local[f'thigh_{side}']
                q_local_sh = _norm(_qmul(_qconj(q_parent_th), qsh))
                q_local_sh = JointConstraints.knee(q_local_sh, side=side, hardness=self._hardness)
                local[f'shin_{side}'] = _norm(_qmul(q_parent_th, q_local_sh))
            else:
                local[f'shin_{side}'] = local[f'thigh_{side}']

            if foot_id in live_sensors:
                qft = _norm(sensor_quats[foot_id])
                q_parent_sh = local[f'shin_{side}']
                q_local_ft = _norm(_qmul(_qconj(q_parent_sh), qft))
                q_local_ft = JointConstraints.ankle(q_local_ft, side=side, hardness=self._hardness)
                local[f'foot_{side}'] = _norm(_qmul(q_parent_sh, q_local_ft))
            else:
                local[f'foot_{side}'] = local[f'shin_{side}']

        # Arms — use real sensors if live, else procedural estimation
        for side, sign in (('l', -1), ('r', 1)):
            ua_id = f'l_upper_arm' if side == 'l' else 'r_upper_arm'
            fa_id = f'l_forearm'   if side == 'l' else 'r_forearm'

            if ua_id in live_sensors:
                qua = _norm(sensor_quats[ua_id])
                q_local_ua = _norm(_qmul(_qconj(local['chest']), qua))
                q_local_ua = JointConstraints.shoulder(q_local_ua, side=side, hardness=self._hardness)
                local[f'upper_arm_{side}'] = _norm(_qmul(local['chest'], q_local_ua))
            else:
                # Consumer estimation: arm swings opposite to same-side leg
                swing = self._estimate_arm_swing(side, cadence_hz, local)
                local[f'upper_arm_{side}'] = swing

            if fa_id in live_sensors:
                qfa = _norm(sensor_quats[fa_id])
                q_parent_ua = local[f'upper_arm_{side}']
                q_local_fa = _norm(_qmul(_qconj(q_parent_ua), qfa))
                q_local_fa = JointConstraints.elbow(q_local_fa, side=side, hardness=self._hardness)
                local[f'forearm_{side}'] = _norm(_qmul(q_parent_ua, q_local_fa))
            else:
                local[f'forearm_{side}'] = local[f'upper_arm_{side}']

        # ── FK pass ───────────────────────────────────────────────────────
        J = {}
        px = float(root_pos[0])
        pz = float(root_pos[2]) if len(root_pos) > 2 else 0.0
        py = LL['pelvis_h'] + root_y_offset

        UP = np.array([0.0, 1.0, 0.0])
        DN = np.array([0.0,-1.0, 0.0])

        qpe = local['pelvis']
        qch = local['chest']
        qhd = local['head']

        J['pelvis']   = np.array([px, py, pz])
        J['lumbar']   = J['pelvis']   + _qrot(qpe, UP * LL['pelvis_lumbar'])
        J['chest']    = J['lumbar']   + _qrot(qpe, UP * LL['lumbar_chest'])
        J['neck']     = J['chest']    + _qrot(qch, UP * LL['chest_neck'])
        J['head']     = J['neck']     + _qrot(qhd, UP * LL['neck_head'])
        J['head_top'] = J['head']     + _qrot(qhd, UP * LL['head_top'])

        for side, sx in (('l', -1), ('r', 1)):
            qth = local[f'thigh_{side}']
            qsh = local[f'shin_{side}']
            qft = local[f'foot_{side}']
            qua = local[f'upper_arm_{side}']
            qfa = local[f'forearm_{side}']

            hip_off = np.array([sx * LL['hip_offset_x'], LL['hip_offset_y'], 0.0])
            J[f'hip_{side}']      = J['pelvis']        + _qrot(qpe, hip_off)
            J[f'knee_{side}']     = J[f'hip_{side}']   + _qrot(qth, DN * LL['thigh'])
            J[f'ankle_{side}']    = J[f'knee_{side}']  + _qrot(qsh, DN * LL['shin'])
            J[f'foot_{side}']     = J[f'ankle_{side}'] + _qrot(qft, np.array([0.0,-0.05,LL['foot']]))

            clav_off = np.array([sx * LL['clavicle'], 0.0, 0.0])
            J[f'shoulder_{side}'] = J['chest']             + _qrot(qch, clav_off)
            J[f'elbow_{side}']    = J[f'shoulder_{side}']  + _qrot(qua, DN * LL['upper_arm'])
            J[f'wrist_{side}']    = J[f'elbow_{side}']     + _qrot(qfa, DN * LL['forearm'])
            J[f'hand_{side}']     = J[f'wrist_{side}']     + _qrot(qfa, DN * LL['hand'])

        # ── IK foot planting ──────────────────────────────────────────────
        if self._ik_enabled:
            for side in ('l', 'r'):
                J = self._ik_foot_plant(J, local, side)

        # ── velocity estimation ───────────────────────────────────────────
        bones: Dict[str, BoneState] = {}
        for jname, jpos in J.items():
            prev_pos = self._prev_joints.get(jname, jpos)
            vel = (jpos - prev_pos) / max(dt, 1e-4)
            bones[jname] = BoneState(world_pos=jpos, velocity=vel)
        self._prev_joints = {k: v.copy() for k, v in J.items()}

        conf = min(1.0, len(live_sensors) / 6.0) if live_sensors else 0.0

        return SkeletonState(
            joints=J,
            bones=bones,
            root_pos=root_pos.copy(),
            root_rot=qpe.copy(),
            live_sensors=live_sensors,
            tracking_confidence=conf,
            timestamp=time.time(),
        )

    # ── IK: two-bone leg solver ───────────────────────────────────────────────

    def _ik_foot_plant(
        self, J: Dict[str, np.ndarray], local: dict, side: str
    ) -> Dict[str, np.ndarray]:
        """
        If the foot joint is below the floor plane (y < 0), use two-bone IK
        to pin it at y=0 and solve the knee position accordingly.
        This eliminates foot skating and penetration.
        """
        foot_key  = f'foot_{side}'
        ankle_key = f'ankle_{side}'
        knee_key  = f'knee_{side}'
        hip_key   = f'hip_{side}'

        foot_pos  = J.get(foot_key)
        hip_pos   = J.get(hip_key)
        if foot_pos is None or hip_pos is None:
            return J

        # Only plant when foot is at or below floor
        floor_y = 0.0
        if float(foot_pos[1]) > floor_y + 0.02:
            return J   # foot is in the air, no IK needed

        # Target: foot pinned at floor height
        target_foot = foot_pos.copy()
        target_foot[1] = floor_y

        # Two-bone IK: hip → knee → ankle/foot
        L1 = LL['thigh']   # hip to knee
        L2 = LL['shin']    # knee to ankle

        hip_to_target = target_foot - hip_pos
        dist = float(np.linalg.norm(hip_to_target))
        dist = np.clip(dist, abs(L1 - L2) + 0.001, L1 + L2 - 0.001)

        # Cosine rule for knee angle
        cos_angle = (L1**2 + dist**2 - L2**2) / (2 * L1 * dist)
        knee_angle = np.arccos(np.clip(cos_angle, -1, 1))

        # Knee bends forward (slightly forward of the hip-ankle axis)
        dir_vec = hip_to_target / (dist + 1e-9)
        # Perpendicular in the sagittal plane (forward direction = -Z world)
        perp = np.cross(dir_vec, np.array([0.0, 0.0, -1.0]))
        if np.linalg.norm(perp) < 1e-6:
            perp = np.array([1.0, 0.0, 0.0])
        perp = perp / np.linalg.norm(perp)

        knee_pos = (
            hip_pos
            + dir_vec * L1 * np.cos(knee_angle)
            + perp    * L1 * np.sin(knee_angle) * 0.5
        )

        J[knee_key]  = knee_pos
        J[ankle_key] = target_foot + np.array([0.0, LL['foot']*0.05, 0.0])
        J[foot_key]  = target_foot

        return J

    # ── Consumer estimation: procedural arm swing ─────────────────────────────

    def _estimate_arm_swing(
        self, side: str, cadence_hz: float, local: dict
    ) -> np.ndarray:
        """
        Generate a believable arm swing quaternion when arm IMUs are absent.
        Arm swing is contra-lateral to the same-side thigh:
          left arm swings forward when right thigh swings forward.
        """
        cad = max(cadence_hz, 0.5)
        phi = 2 * np.pi * cad * self._t
        amp = 0.28

        if side == 'l':
            swing = amp * np.sin(phi + np.pi)   # contra to left thigh
        else:
            swing = amp * np.sin(phi)

        # Compose: chest orientation + local arm swing around X (forward/back)
        q_chest = local.get('chest', np.array([1.0, 0, 0, 0]))
        q_swing = _ax_angle(np.array([1.0, 0.0, 0.0]), np.degrees(swing))
        # Small constant out-angle (arms hang slightly away from body)
        sign = -1 if side == 'l' else 1
        q_abduct = _ax_angle(np.array([0.0, 0.0, 1.0]), sign * 8.0)
        return _norm(_qmul(q_chest, _qmul(q_abduct, q_swing)))