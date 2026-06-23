# """
# backend/skeleton.py
# -------------------
# 3D human skeleton reconstruction from IMU quaternion data.

# Architecture:
#   - Kinematic chain: pelvis → spine → chest → neck → head
#                      pelvis → hip → thigh → shin → foot  (×2)
#                      chest  → shoulder → upper_arm → forearm → hand (×2)

#   - Each sensor stores a rotation quaternion Q_world (orientation in world frame)
#   - Joint positions are computed forward-kinematically from the pelvis root
#     using segment vectors rotated by the sensor quaternion for that segment.

#   - Before calibration (T-pose), each sensor's first reading is stored as
#     Q_offset (the reference orientation). All subsequent rotations are expressed
#     relative to this offset so that the neutral T-pose = standard anatomical pose.

#   - Limb lengths default to average adult proportions and are overridden by
#     GVHMR calibration output.

#   - Root translation (X/Z horizontal, Y vertical offset) is accumulated via
#     update_root_translation() — call each frame with integrated accelerometer
#     displacement or GVHMR-derived positional delta. All joint positions are
#     offset by this root so the skeleton moves through world space.

# Usage:
#     engine = SkeletonEngine()
#     engine.update("pelvis", quaternion_array, timestamp)
#     engine.update_root_translation(dx, dz)      # optional: world-space movement
#     joints = engine.get_joints()   # dict[joint_name -> [x,y,z]]
# """

# from __future__ import annotations

# import time
# from typing import Optional
# import numpy as np


# # ──────────────────────────────────────────────────────────────
# # QUATERNION HELPERS
# # ──────────────────────────────────────────────────────────────

# def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
#     """Hamilton product q1 * q2.  Both are [w, x, y, z]."""
#     w1, x1, y1, z1 = q1
#     w2, x2, y2, z2 = q2
#     return np.array([
#         w1*w2 - x1*x2 - y1*y2 - z1*z2,
#         w1*x2 + x1*w2 + y1*z2 - z1*y2,
#         w1*y2 - x1*z2 + y1*w2 + z1*x2,
#         w1*z2 + x1*y2 - y1*x2 + z1*w2,
#     ])


# def quat_conjugate(q: np.ndarray) -> np.ndarray:
#     return np.array([q[0], -q[1], -q[2], -q[3]])


# def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
#     """Rotate vector v by quaternion q  (q must be normalised)."""
#     qv = np.array([0.0, v[0], v[1], v[2]])
#     rotated = quat_mul(quat_mul(q, qv), quat_conjugate(q))
#     return rotated[1:]


# def normalize_quat(q: np.ndarray) -> np.ndarray:
#     n = np.linalg.norm(q)
#     return q / n if n > 1e-9 else np.array([1.0, 0, 0, 0])


# # ──────────────────────────────────────────────────────────────
# # DEFAULT LIMB LENGTHS  (metres, average adult ~1.75 m)
# # ──────────────────────────────────────────────────────────────

# DEFAULT_LIMB_LENGTHS: dict[str, float] = {
#     # Spine — segment lengths along the chain
#     "spine_pelvis_to_lumbar": 0.10,  # pelvis centre → lumbar root
#     "lumbar":         0.15,   # lumbar → chest
#     "chest":          0.20,   # chest → neck
#     "neck":           0.10,   # neck → head centre
#     "head":           0.12,   # head radius (top of skull offset)

#     # Standing height — pelvis centre above floor (used as Y origin)
#     "pelvis_height":  0.95,

#     # Lower limb
#     "thigh_l":        0.42,
#     "thigh_r":        0.42,
#     "shin_l":         0.40,
#     "shin_r":         0.40,
#     "foot_l":         0.18,   # heel → toe length
#     "foot_r":         0.18,

#     # Upper limb
#     "clavicle":       0.16,   # chest centre → shoulder (half shoulder width extra)
#     "l_upper_arm":    0.30,
#     "r_upper_arm":    0.30,
#     "l_forearm":      0.26,
#     "r_forearm":      0.26,
#     "l_hand":         0.08,
#     "r_hand":         0.08,
# }

# # ──────────────────────────────────────────────────────────────
# # SENSOR IDs (must match ble_receiver.py LOC_TO_SENSOR_ID values)
# # ──────────────────────────────────────────────────────────────

# SENSOR_IDS = [
#     "pelvis", "chest",
#     "thigh_l", "thigh_r",
#     "shin_l",  "shin_r",
#     "l_upper_arm", "r_upper_arm",
#     "l_forearm",   "r_forearm",
#     "head",
#     "l_foot",  "r_foot",
# ]

# # How long before a sensor is considered stale / inactive
# # 200 ms = 10 missed packets at 50 Hz — matches original skeleton.py
# STALE_TIMEOUT = 0.20  # seconds


# # ──────────────────────────────────────────────────────────────
# # SKELETON ENGINE
# # ──────────────────────────────────────────────────────────────

# class SkeletonEngine:
#     """
#     Maintains per-sensor quaternion state and reconstructs 3-D joint
#     positions every time a new IMU packet arrives.

#     Root translation
#     ----------------
#     Call update_root_translation(dx, dz) each frame with the world-space
#     horizontal displacement for that timestep (metres). The engine
#     accumulates these into _root_position and applies them as an offset
#     to every joint, so the skeleton moves through world space while the
#     kinematic chain remains relative-rotation-based.

#     Vertical (Y) displacement can be supplied via the dy parameter of
#     update_root_translation — useful if you have barometric / GVHMR
#     height data.  If not supplied, pelvis height stays at pelvis_height.
#     """

#     SENSOR_IDS = SENSOR_IDS

#     def __init__(self, limb_lengths: Optional[dict] = None):
#         self._ll: dict[str, float] = {**DEFAULT_LIMB_LENGTHS}
#         if limb_lengths:
#             self._ll.update(limb_lengths)

#         # Latest world-frame quaternion per sensor  [w,x,y,z]
#         self._quats: dict[str, np.ndarray] = {}

#         # T-pose calibration offset per sensor  [w,x,y,z]
#         # Q_relative = Q_offset_inv * Q_world
#         self._calibration_offsets: dict[str, np.ndarray] = {
#             sid: np.array([1.0, 0, 0, 0]) for sid in SENSOR_IDS
#         }
#         self._calibrated = False

#         # Timestamps for staleness check
#         self._last_update: dict[str, float] = {}

#         # Estimated (interpolated) flags for joints we synthesised
#         self._estimated: dict[str, bool] = {}

#         # ── Root translation (world-space position of pelvis origin) ──
#         # X = lateral (right +), Y = vertical delta, Z = forward (+)
#         self._root_position: np.ndarray = np.zeros(3, dtype=float)

#     # ── public API ──────────────────────────────────────────────

#     def update(self, sensor_id: str, quaternion: np.ndarray, timestamp: float):
#         if sensor_id not in SENSOR_IDS:
#             return
#         self._quats[sensor_id] = normalize_quat(quaternion)
#         self._last_update[sensor_id] = timestamp

#     def update_root_translation(self, dx: float = 0.0, dz: float = 0.0, dy: float = 0.0):
#         """
#         Accumulate world-space positional delta for the pelvis root.

#         Parameters
#         ----------
#         dx : float
#             Lateral displacement this frame (metres, right = +X).
#         dz : float
#             Forward/back displacement this frame (metres, forward = +Z).
#         dy : float
#             Vertical displacement this frame (metres, up = +Y).
#             Leave at 0 to keep pelvis_height as the sole Y reference.

#         Typical usage
#         -------------
#         From accelerometer double-integration (noisy, drift quickly):
#             engine.update_root_translation(dx=ax*dt*dt, dz=az*dt*dt)

#         From GVHMR / video pose estimator delta:
#             engine.update_root_translation(dx=pos_delta[0], dz=pos_delta[2], dy=pos_delta[1])
#         """
#         self._root_position[0] += dx
#         self._root_position[1] += dy
#         self._root_position[2] += dz

#     def reset_root_translation(self):
#         """Reset accumulated root position to origin (e.g. on session start)."""
#         self._root_position[:] = 0.0

#     def get_root_position(self) -> list[float]:
#         """Return current root [x, y, z] as a plain list."""
#         return self._root_position.tolist()

#     def calibrate_tpose(self):
#         """
#         Store current quaternions as the neutral (T-pose) reference.
#         All joint angles will be computed relative to this snapshot.
#         Also resets root translation so the T-pose position is the origin.
#         """
#         for sid, q in self._quats.items():
#             self._calibration_offsets[sid] = normalize_quat(q)
#         self._calibrated = True
#         self.reset_root_translation()
#         print("[Skeleton] T-pose calibration captured.")

#     def set_limb_lengths(self, ll: Optional[dict]):
#         if ll:
#             self._ll.update(ll)
#         else:
#             self._ll = {**DEFAULT_LIMB_LENGTHS}

#     def is_ready(self) -> bool:
#         """True once the pelvis sensor has sent a recent packet (within stale window)."""
#         return time.time() - self._last_update.get("pelvis", 0.0) < STALE_TIMEOUT

#     def is_ready_for(self, required_sensors: list) -> bool:
#         """True when every sensor in required_sensors has sent data recently."""
#         now = time.time()
#         return all(now - self._last_update.get(s, 0.0) < STALE_TIMEOUT for s in required_sensors)

#     def get_active_sensors(self) -> list[str]:
#         now = time.time()
#         return [
#             sid for sid, ts in self._last_update.items()
#             if now - ts < STALE_TIMEOUT
#         ]

#     def get_joints(self) -> dict[str, list[float]]:
#         """
#         Compute and return all joint positions as {name: [x, y, z]} in metres.

#         The pelvis root is placed at:
#             X = _root_position[0]   (accumulated lateral translation)
#             Y = pelvis_height + _root_position[1]   (standing height + vertical delta)
#             Z = _root_position[2]   (accumulated forward translation)

#         All child joints are offset from this root, so the entire skeleton
#         moves through world space when update_root_translation() is called.
#         Y-axis points up.
#         """
#         joints: dict[str, np.ndarray] = {}
#         self._estimated = {}

#         ll = self._ll

#         # ── helper: get relative quaternion for a sensor ──────────
#         def get_q(sensor_id: str) -> np.ndarray:
#             q_world  = self._quats.get(sensor_id, np.array([1.0, 0, 0, 0]))
#             q_offset = self._calibration_offsets.get(sensor_id, np.array([1.0, 0, 0, 0]))
#             # relative = inv(offset) * world
#             q_rel = quat_mul(quat_conjugate(q_offset), q_world)
#             return normalize_quat(q_rel)

#         def estimated(sensor_id: str) -> bool:
#             return sensor_id not in self._quats

#         # ── PELVIS (root) ─────────────────────────────────────────
#         # Pelvis Y = standing pelvis_height PLUS any accumulated vertical delta.
#         # X and Z come entirely from accumulated root translation.
#         root_x = float(self._root_position[0])
#         root_y = float(self._root_position[1])
#         root_z = float(self._root_position[2])

#         joints["pelvis"] = np.array([
#             root_x,
#             ll.get("pelvis_height", 0.95) + root_y,
#             root_z,
#         ])

#         q_pelvis = get_q("pelvis")

#         # ── SPINE chain ───────────────────────────────────────────
#         # Default segment direction: +Y (upward along spine)
#         up = np.array([0.0, 1.0, 0.0])

#         lumbar = joints["pelvis"] + quat_rotate(q_pelvis, up * ll["spine_pelvis_to_lumbar"])
#         joints["lumbar"] = lumbar

#         # Use chest sensor if available, else continue pelvis rotation
#         q_chest = get_q("chest") if "chest" in self._quats else q_pelvis
#         self._estimated["chest_est"] = estimated("chest")

#         chest = lumbar + quat_rotate(q_chest, up * ll["lumbar"])
#         joints["chest"] = chest

#         neck = chest + quat_rotate(q_chest, up * ll["chest"])
#         joints["neck"] = neck

#         q_head = get_q("head") if "head" in self._quats else q_chest
#         head = neck + quat_rotate(q_head, up * ll["neck"])
#         joints["head"] = head
#         joints["head_top"] = head + quat_rotate(q_head, up * ll["head"])

#         # ── LEFT LOWER LIMB ───────────────────────────────────────
#         # Hip offset from pelvis: lateral ±X, slightly down
#         hip_offset_l = np.array([-0.10, -0.05, 0.0])
#         hip_l = joints["pelvis"] + quat_rotate(q_pelvis, hip_offset_l)
#         joints["hip_l"] = hip_l

#         q_thigh_l = get_q("thigh_l")
#         knee_l = hip_l + quat_rotate(q_thigh_l, np.array([0, -1, 0]) * ll["thigh_l"])
#         joints["knee_l"] = knee_l

#         q_shin_l = get_q("shin_l") if "shin_l" in self._quats else q_thigh_l
#         self._estimated["shin_l_est"] = estimated("shin_l")
#         ankle_l = knee_l + quat_rotate(q_shin_l, np.array([0, -1, 0]) * ll["shin_l"])
#         joints["ankle_l"] = ankle_l

#         q_foot_l = get_q("l_foot") if "l_foot" in self._quats else q_shin_l
#         foot_l = ankle_l + quat_rotate(q_foot_l, np.array([0, -0.1, 1]) * ll["foot_l"])
#         joints["foot_l"] = foot_l

#         # ── RIGHT LOWER LIMB ──────────────────────────────────────
#         hip_offset_r = np.array([0.10, -0.05, 0.0])
#         hip_r = joints["pelvis"] + quat_rotate(q_pelvis, hip_offset_r)
#         joints["hip_r"] = hip_r

#         q_thigh_r = get_q("thigh_r")
#         knee_r = hip_r + quat_rotate(q_thigh_r, np.array([0, -1, 0]) * ll["thigh_r"])
#         joints["knee_r"] = knee_r

#         q_shin_r = get_q("shin_r") if "shin_r" in self._quats else q_thigh_r
#         self._estimated["shin_r_est"] = estimated("shin_r")
#         ankle_r = knee_r + quat_rotate(q_shin_r, np.array([0, -1, 0]) * ll["shin_r"])
#         joints["ankle_r"] = ankle_r

#         q_foot_r = get_q("r_foot") if "r_foot" in self._quats else q_shin_r
#         foot_r = ankle_r + quat_rotate(q_foot_r, np.array([0, -0.1, 1]) * ll["foot_r"])
#         joints["foot_r"] = foot_r

#         # ── LEFT UPPER LIMB ───────────────────────────────────────
#         shoulder_l = chest + quat_rotate(q_chest, np.array([-ll["clavicle"], 0, 0]))
#         joints["shoulder_l"] = shoulder_l

#         q_ua_l = get_q("l_upper_arm") if "l_upper_arm" in self._quats else q_chest
#         self._estimated["l_upper_arm_est"] = estimated("l_upper_arm")
#         elbow_l = shoulder_l + quat_rotate(q_ua_l, np.array([0, -1, 0]) * ll["l_upper_arm"])
#         joints["elbow_l"] = elbow_l

#         q_fa_l = get_q("l_forearm") if "l_forearm" in self._quats else q_ua_l
#         self._estimated["l_forearm_est"] = estimated("l_forearm")
#         wrist_l = elbow_l + quat_rotate(q_fa_l, np.array([0, -1, 0]) * ll["l_forearm"])
#         joints["wrist_l"] = wrist_l
#         joints["hand_l"] = wrist_l + quat_rotate(q_fa_l, np.array([0, -1, 0]) * ll["l_hand"])

#         # ── RIGHT UPPER LIMB ──────────────────────────────────────
#         shoulder_r = chest + quat_rotate(q_chest, np.array([ll["clavicle"], 0, 0]))
#         joints["shoulder_r"] = shoulder_r

#         q_ua_r = get_q("r_upper_arm") if "r_upper_arm" in self._quats else q_chest
#         self._estimated["r_upper_arm_est"] = estimated("r_upper_arm")
#         elbow_r = shoulder_r + quat_rotate(q_ua_r, np.array([0, -1, 0]) * ll["r_upper_arm"])
#         joints["elbow_r"] = elbow_r

#         q_fa_r = get_q("r_forearm") if "r_forearm" in self._quats else q_ua_r
#         self._estimated["r_forearm_est"] = estimated("r_forearm")
#         wrist_r = elbow_r + quat_rotate(q_fa_r, np.array([0, -1, 0]) * ll["r_forearm"])
#         joints["wrist_r"] = wrist_r
#         joints["hand_r"] = wrist_r + quat_rotate(q_fa_r, np.array([0, -1, 0]) * ll["r_hand"])

#         return {k: v.tolist() for k, v in joints.items()}

#     # ── Joint angle helpers (ported from original skeleton.py) ───────────────

#     def get_knee_angles(self) -> dict:
#         """
#         Returns knee flexion in degrees for left and right.
#         0° = fully extended, 90° = right-angle bend.
#         Requires get_joints() to have been called at least once.
#         """
#         j = self.get_joints()
#         if not j:
#             return {"knee_l": 0.0, "knee_r": 0.0}

#         def _angle(hip, knee, ankle):
#             hip, knee, ankle = np.array(hip), np.array(knee), np.array(ankle)
#             v1 = hip - knee
#             v2 = ankle - knee
#             cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
#             return 180.0 - float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

#         return {
#             "knee_l": _angle(j["hip_l"], j["knee_l"], j["ankle_l"]),
#             "knee_r": _angle(j["hip_r"], j["knee_r"], j["ankle_r"]),
#         }

#     def get_pelvis_tilt(self) -> dict:
#         """
#         Returns pelvis lateral tilt and anterior-posterior tilt in degrees
#         derived directly from the raw pelvis quaternion.
#         """
#         q = self._quats.get("pelvis", np.array([1.0, 0, 0, 0]))
#         roll = float(np.degrees(np.arctan2(
#             2 * (q[0]*q[1] + q[2]*q[3]),
#             1 - 2 * (q[1]**2 + q[2]**2)
#         )))
#         pitch = float(np.degrees(np.arcsin(
#             np.clip(2 * (q[0]*q[2] - q[3]*q[1]), -1.0, 1.0)
#         )))
#         return {"lateral_tilt": roll, "anterior_tilt": pitch}

"""
backend/skeleton.py
-------------------
3D human skeleton reconstruction from IMU quaternion data.

Architecture:
  - Kinematic chain: pelvis → spine → chest → neck → head
                     pelvis → hip → thigh → shin → foot  (×2)
                     chest  → shoulder → upper_arm → forearm → hand (×2)

  - Each sensor stores a rotation quaternion Q_world (orientation in world frame)
  - Joint positions are computed forward-kinematically from the pelvis root
    using segment vectors rotated by the sensor quaternion for that segment.

  - Before calibration (T-pose), each sensor's first reading is stored as
    Q_offset (the reference orientation). All subsequent rotations are expressed
    relative to this offset so that the neutral T-pose = standard anatomical pose.

  - Limb lengths default to average adult proportions and are overridden by
    GVHMR calibration output.

  - Root translation (X/Z horizontal, Y vertical offset) is accumulated via
    update_root_translation() — call each frame with integrated accelerometer
    displacement or GVHMR-derived positional delta. All joint positions are
    offset by this root so the skeleton moves through world space.

  v_remap: swap_yz_negz applied to all incoming sensor quaternions to correct
    the R2P firmware axis convention (Z-up → Y-up skeleton). Determined
    empirically via imu_test/skeleton_viewer.html diagnostic harness.

Usage:
    engine = SkeletonEngine()
    engine.update("pelvis", quaternion_array, timestamp)
    engine.update_root_translation(dx, dz)      # optional: world-space movement
    joints = engine.get_joints()   # dict[joint_name -> [x,y,z]]
"""

from __future__ import annotations

import time
from typing import Optional
import numpy as np


# ──────────────────────────────────────────────────────────────
# QUATERNION HELPERS
# ──────────────────────────────────────────────────────────────

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 * q2.  Both are [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q  (q must be normalised)."""
    qv = np.array([0.0, v[0], v[1], v[2]])
    rotated = quat_mul(quat_mul(q, qv), quat_conjugate(q))
    return rotated[1:]


def normalize_quat(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-9 else np.array([1.0, 0, 0, 0])


def _remap(q: np.ndarray) -> np.ndarray:
    """
    swap_yz_negz — corrects the R2P firmware quaternion axis convention.

    The IMU firmware outputs quaternions in a Z-up frame. The skeleton uses
    Y-up. Without this remap, a physical left/right turn arrives as a
    pitch (bow) instead of a yaw (turn). Determined empirically via the
    imu_test diagnostic harness using the pelvis sensor.

    Mapping:  output [w, x, y, z] = input [w, x, z, y]
    i.e. swap Y and Z axes only. This correctly maps all three rotation axes:
      sensor X → skeleton pitch  (tilt forward/back)
      sensor Y → skeleton roll   (tilt left/right)
      sensor Z → skeleton yaw    (turn left/right)

    Apply to ALL sensors — they all run the same firmware convention.
    """
    w, x, y, z = q
    return np.array([w, x, z, y])


# ──────────────────────────────────────────────────────────────
# DEFAULT LIMB LENGTHS  (metres, average adult ~1.75 m)
# ──────────────────────────────────────────────────────────────

DEFAULT_LIMB_LENGTHS: dict[str, float] = {
    # Spine — segment lengths along the chain
    "spine_pelvis_to_lumbar": 0.10,  # pelvis centre → lumbar root
    "lumbar":         0.15,   # lumbar → chest
    "chest":          0.20,   # chest → neck
    "neck":           0.10,   # neck → head centre
    "head":           0.12,   # head radius (top of skull offset)

    # Standing height — pelvis centre above floor (used as Y origin)
    "pelvis_height":  0.95,

    # Lower limb
    "thigh_l":        0.42,
    "thigh_r":        0.42,
    "shin_l":         0.40,
    "shin_r":         0.40,
    "foot_l":         0.18,   # heel → toe length
    "foot_r":         0.18,

    # Upper limb
    "clavicle":       0.16,   # chest centre → shoulder (half shoulder width extra)
    "l_upper_arm":    0.30,
    "r_upper_arm":    0.30,
    "l_forearm":      0.26,
    "r_forearm":      0.26,
    "l_hand":         0.08,
    "r_hand":         0.08,
}

# ──────────────────────────────────────────────────────────────
# SENSOR IDs (must match ble_receiver.py LOC_TO_SENSOR_ID values)
# ──────────────────────────────────────────────────────────────

SENSOR_IDS = [
    "pelvis", "chest",
    "thigh_l", "thigh_r",
    "shin_l",  "shin_r",
    "l_upper_arm", "r_upper_arm",
    "l_forearm",   "r_forearm",
    "head",
    "l_foot",  "r_foot",
]

# How long before a sensor is considered stale / inactive
# 200 ms = 10 missed packets at 50 Hz — matches original skeleton.py
STALE_TIMEOUT = 0.20  # seconds


# ──────────────────────────────────────────────────────────────
# SKELETON ENGINE
# ──────────────────────────────────────────────────────────────

class SkeletonEngine:
    """
    Maintains per-sensor quaternion state and reconstructs 3-D joint
    positions every time a new IMU packet arrives.

    Root translation
    ----------------
    Call update_root_translation(dx, dz) each frame with the world-space
    horizontal displacement for that timestep (metres). The engine
    accumulates these into _root_position and applies them as an offset
    to every joint, so the skeleton moves through world space while the
    kinematic chain remains relative-rotation-based.

    Vertical (Y) displacement can be supplied via the dy parameter of
    update_root_translation — useful if you have barometric / GVHMR
    height data.  If not supplied, pelvis height stays at pelvis_height.
    """

    SENSOR_IDS = SENSOR_IDS

    def __init__(self, limb_lengths: Optional[dict] = None):
        self._ll: dict[str, float] = {**DEFAULT_LIMB_LENGTHS}
        if limb_lengths:
            self._ll.update(limb_lengths)

        # Latest world-frame quaternion per sensor  [w,x,y,z]
        self._quats: dict[str, np.ndarray] = {}

        # T-pose calibration offset per sensor  [w,x,y,z]
        # Q_relative = Q_offset_inv * Q_world
        self._calibration_offsets: dict[str, np.ndarray] = {
            sid: np.array([1.0, 0, 0, 0]) for sid in SENSOR_IDS
        }
        self._calibrated = False

        # Timestamps for staleness check
        self._last_update: dict[str, float] = {}

        # Estimated (interpolated) flags for joints we synthesised
        self._estimated: dict[str, bool] = {}

        # ── Root translation (world-space position of pelvis origin) ──
        # X = lateral (right +), Y = vertical delta, Z = forward (+)
        self._root_position: np.ndarray = np.zeros(3, dtype=float)

    # ── public API ──────────────────────────────────────────────

    def update(self, sensor_id: str, quaternion: np.ndarray, timestamp: float):
        if sensor_id not in SENSOR_IDS:
            return
        # Apply axis remap before storing — corrects Z-up firmware convention
        # to Y-up skeleton convention for all sensors.
        self._quats[sensor_id] = normalize_quat(_remap(quaternion))
        self._last_update[sensor_id] = timestamp

    def update_root_translation(self, dx: float = 0.0, dz: float = 0.0, dy: float = 0.0):
        """
        Accumulate world-space positional delta for the pelvis root.

        Parameters
        ----------
        dx : float
            Lateral displacement this frame (metres, right = +X).
        dz : float
            Forward/back displacement this frame (metres, forward = +Z).
        dy : float
            Vertical displacement this frame (metres, up = +Y).
            Leave at 0 to keep pelvis_height as the sole Y reference.

        Typical usage
        -------------
        From accelerometer double-integration (noisy, drift quickly):
            engine.update_root_translation(dx=ax*dt*dt, dz=az*dt*dt)

        From GVHMR / video pose estimator delta:
            engine.update_root_translation(dx=pos_delta[0], dz=pos_delta[2], dy=pos_delta[1])
        """
        self._root_position[0] += dx
        self._root_position[1] += dy
        self._root_position[2] += dz

    def reset_root_translation(self):
        """Reset accumulated root position to origin (e.g. on session start)."""
        self._root_position[:] = 0.0

    def get_root_position(self) -> list[float]:
        """Return current root [x, y, z] as a plain list."""
        return self._root_position.tolist()

    def calibrate_tpose(self):
        """
        Store current quaternions as the neutral (T-pose) reference.
        All joint angles will be computed relative to this snapshot.
        Also resets root translation so the T-pose position is the origin.

        Note: quaternions stored in self._quats have already been remapped
        by _remap() in update(), so the calibration offset is captured in
        the corrected frame — no extra remap needed here.
        """
        for sid, q in self._quats.items():
            self._calibration_offsets[sid] = normalize_quat(q)
        self._calibrated = True
        self.reset_root_translation()
        print("[Skeleton] T-pose calibration captured.")

    def set_limb_lengths(self, ll: Optional[dict]):
        if ll:
            self._ll.update(ll)
        else:
            self._ll = {**DEFAULT_LIMB_LENGTHS}

    def is_ready(self) -> bool:
        """True once the pelvis sensor has sent a recent packet (within stale window)."""
        return time.time() - self._last_update.get("pelvis", 0.0) < STALE_TIMEOUT

    def is_ready_for(self, required_sensors: list) -> bool:
        """True when every sensor in required_sensors has sent data recently."""
        now = time.time()
        return all(now - self._last_update.get(s, 0.0) < STALE_TIMEOUT for s in required_sensors)

    def get_active_sensors(self) -> list[str]:
        now = time.time()
        return [
            sid for sid, ts in self._last_update.items()
            if now - ts < STALE_TIMEOUT
        ]

    def get_joints(self) -> dict[str, list[float]]:
        """
        Compute and return all joint positions as {name: [x, y, z]} in metres.

        The pelvis root is placed at:
            X = _root_position[0]   (accumulated lateral translation)
            Y = pelvis_height + _root_position[1]   (standing height + vertical delta)
            Z = _root_position[2]   (accumulated forward translation)

        All child joints are offset from this root, so the entire skeleton
        moves through world space when update_root_translation() is called.
        Y-axis points up.
        """
        joints: dict[str, np.ndarray] = {}
        self._estimated = {}

        ll = self._ll

        # ── helper: get relative quaternion for a sensor ──────────
        def get_q(sensor_id: str) -> np.ndarray:
            q_world  = self._quats.get(sensor_id, np.array([1.0, 0, 0, 0]))
            q_offset = self._calibration_offsets.get(sensor_id, np.array([1.0, 0, 0, 0]))
            # relative = inv(offset) * world
            q_rel = quat_mul(quat_conjugate(q_offset), q_world)
            return normalize_quat(q_rel)

        def estimated(sensor_id: str) -> bool:
            return sensor_id not in self._quats

        # ── PELVIS (root) ─────────────────────────────────────────
        root_x = float(self._root_position[0])
        root_y = float(self._root_position[1])
        root_z = float(self._root_position[2])

        joints["pelvis"] = np.array([
            root_x,
            ll.get("pelvis_height", 0.95) + root_y,
            root_z,
        ])

        q_pelvis = get_q("pelvis")

        # ── SPINE chain ───────────────────────────────────────────
        up = np.array([0.0, 1.0, 0.0])

        lumbar = joints["pelvis"] + quat_rotate(q_pelvis, up * ll["spine_pelvis_to_lumbar"])
        joints["lumbar"] = lumbar

        q_chest = get_q("chest") if "chest" in self._quats else q_pelvis
        self._estimated["chest_est"] = estimated("chest")

        chest = lumbar + quat_rotate(q_chest, up * ll["lumbar"])
        joints["chest"] = chest

        neck = chest + quat_rotate(q_chest, up * ll["chest"])
        joints["neck"] = neck

        q_head = get_q("head") if "head" in self._quats else q_chest
        head = neck + quat_rotate(q_head, up * ll["neck"])
        joints["head"] = head
        joints["head_top"] = head + quat_rotate(q_head, up * ll["head"])

        # ── LEFT LOWER LIMB ───────────────────────────────────────
        hip_offset_l = np.array([-0.10, -0.05, 0.0])
        hip_l = joints["pelvis"] + quat_rotate(q_pelvis, hip_offset_l)
        joints["hip_l"] = hip_l

        q_thigh_l = get_q("thigh_l")
        knee_l = hip_l + quat_rotate(q_thigh_l, np.array([0, -1, 0]) * ll["thigh_l"])
        joints["knee_l"] = knee_l

        q_shin_l = get_q("shin_l") if "shin_l" in self._quats else q_thigh_l
        self._estimated["shin_l_est"] = estimated("shin_l")
        ankle_l = knee_l + quat_rotate(q_shin_l, np.array([0, -1, 0]) * ll["shin_l"])
        joints["ankle_l"] = ankle_l

        q_foot_l = get_q("l_foot") if "l_foot" in self._quats else q_shin_l
        foot_l = ankle_l + quat_rotate(q_foot_l, np.array([0, -0.1, 1]) * ll["foot_l"])
        joints["foot_l"] = foot_l

        # ── RIGHT LOWER LIMB ──────────────────────────────────────
        hip_offset_r = np.array([0.10, -0.05, 0.0])
        hip_r = joints["pelvis"] + quat_rotate(q_pelvis, hip_offset_r)
        joints["hip_r"] = hip_r

        q_thigh_r = get_q("thigh_r")
        knee_r = hip_r + quat_rotate(q_thigh_r, np.array([0, -1, 0]) * ll["thigh_r"])
        joints["knee_r"] = knee_r

        q_shin_r = get_q("shin_r") if "shin_r" in self._quats else q_thigh_r
        self._estimated["shin_r_est"] = estimated("shin_r")
        ankle_r = knee_r + quat_rotate(q_shin_r, np.array([0, -1, 0]) * ll["shin_r"])
        joints["ankle_r"] = ankle_r

        q_foot_r = get_q("r_foot") if "r_foot" in self._quats else q_shin_r
        foot_r = ankle_r + quat_rotate(q_foot_r, np.array([0, -0.1, 1]) * ll["foot_r"])
        joints["foot_r"] = foot_r

        # ── LEFT UPPER LIMB ───────────────────────────────────────
        shoulder_l = chest + quat_rotate(q_chest, np.array([-ll["clavicle"], 0, 0]))
        joints["shoulder_l"] = shoulder_l

        q_ua_l = get_q("l_upper_arm") if "l_upper_arm" in self._quats else q_chest
        self._estimated["l_upper_arm_est"] = estimated("l_upper_arm")
        elbow_l = shoulder_l + quat_rotate(q_ua_l, np.array([0, -1, 0]) * ll["l_upper_arm"])
        joints["elbow_l"] = elbow_l

        q_fa_l = get_q("l_forearm") if "l_forearm" in self._quats else q_ua_l
        self._estimated["l_forearm_est"] = estimated("l_forearm")
        wrist_l = elbow_l + quat_rotate(q_fa_l, np.array([0, -1, 0]) * ll["l_forearm"])
        joints["wrist_l"] = wrist_l
        joints["hand_l"] = wrist_l + quat_rotate(q_fa_l, np.array([0, -1, 0]) * ll["l_hand"])

        # ── RIGHT UPPER LIMB ──────────────────────────────────────
        shoulder_r = chest + quat_rotate(q_chest, np.array([ll["clavicle"], 0, 0]))
        joints["shoulder_r"] = shoulder_r

        q_ua_r = get_q("r_upper_arm") if "r_upper_arm" in self._quats else q_chest
        self._estimated["r_upper_arm_est"] = estimated("r_upper_arm")
        elbow_r = shoulder_r + quat_rotate(q_ua_r, np.array([0, -1, 0]) * ll["r_upper_arm"])
        joints["elbow_r"] = elbow_r

        q_fa_r = get_q("r_forearm") if "r_forearm" in self._quats else q_ua_r
        self._estimated["r_forearm_est"] = estimated("r_forearm")
        wrist_r = elbow_r + quat_rotate(q_fa_r, np.array([0, -1, 0]) * ll["r_forearm"])
        joints["wrist_r"] = wrist_r
        joints["hand_r"] = wrist_r + quat_rotate(q_fa_r, np.array([0, -1, 0]) * ll["r_hand"])

        return {k: v.tolist() for k, v in joints.items()}

    # ── Joint angle helpers ──────────────────────────────────────

    def get_knee_angles(self) -> dict:
        """
        Returns knee flexion in degrees for left and right.
        0° = fully extended, 90° = right-angle bend.
        """
        j = self.get_joints()
        if not j:
            return {"knee_l": 0.0, "knee_r": 0.0}

        def _angle(hip, knee, ankle):
            hip, knee, ankle = np.array(hip), np.array(knee), np.array(ankle)
            v1 = hip - knee
            v2 = ankle - knee
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            return 180.0 - float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

        return {
            "knee_l": _angle(j["hip_l"], j["knee_l"], j["ankle_l"]),
            "knee_r": _angle(j["hip_r"], j["knee_r"], j["ankle_r"]),
        }

    def get_pelvis_tilt(self) -> dict:
        """
        Returns pelvis lateral tilt and anterior-posterior tilt in degrees
        derived directly from the raw pelvis quaternion.
        """
        q = self._quats.get("pelvis", np.array([1.0, 0, 0, 0]))
        roll = float(np.degrees(np.arctan2(
            2 * (q[0]*q[1] + q[2]*q[3]),
            1 - 2 * (q[1]**2 + q[2]**2)
        )))
        pitch = float(np.degrees(np.arcsin(
            np.clip(2 * (q[0]*q[2] - q[3]*q[1]), -1.0, 1.0)
        )))
        return {"lateral_tilt": roll, "anterior_tilt": pitch}