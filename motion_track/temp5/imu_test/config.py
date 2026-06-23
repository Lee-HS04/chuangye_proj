"""
config.py  —  R2P System Configuration
=======================================
Single source of truth for every constant used across the pipeline.
Import from here rather than duplicating values in individual modules.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ── update rates ──────────────────────────────────────────────────────────────
TARGET_FPS_MIN  = 60
TARGET_FPS_PREF = 120
TARGET_FPS_MAX  = 240
DEFAULT_FPS     = 50.0       # typical BLE throughput per sensor

# ── sensor staleness ──────────────────────────────────────────────────────────
STALE_S = 0.4                # seconds before a sensor is considered dropped

# ── sensor tier definitions ───────────────────────────────────────────────────
PELVIS_ONLY: List[str] = ['pelvis']

CONSUMER: List[str] = [
    'pelvis', 'chest',
    'thigh_l', 'thigh_r',
    'shin_l',  'shin_r',
]

PRO: List[str] = CONSUMER + [
    'head',
    'l_shoulder', 'r_shoulder',
    'l_upper_arm', 'r_upper_arm',
    'l_forearm',   'r_forearm',
    'l_foot',      'r_foot',
]

TIERS: Dict[str, List[str]] = {
    'pelvis_only': PELVIS_ONLY,
    'consumer':    CONSUMER,
    'pro':         PRO,
}

ALL_SENSORS = set(PRO)

# ── BLE firmware sensor location strings → canonical sensor IDs ───────────────
LOC_TO_SENSOR_ID: Dict[str, str] = {
    'PELVIS':      'pelvis',
    'CHEST':       'chest',
    'L_THIGH':     'thigh_l',
    'R_THIGH':     'thigh_r',
    'L_SHIN':      'shin_l',
    'R_SHIN':      'shin_r',
    'HEAD':        'head',
    'L_SHOULDER':  'l_shoulder',
    'R_SHOULDER':  'r_shoulder',
    'L_UPPER_ARM': 'l_upper_arm',
    'R_UPPER_ARM': 'r_upper_arm',
    'L_FOREARM':   'l_forearm',
    'R_FOREARM':   'r_forearm',
    'L_FOOT':      'l_foot',
    'R_FOOT':      'r_foot',
}

# ── axis remap presets ────────────────────────────────────────────────────────
# Each entry is a list of (source_index, sign) for [x, y, z] of the vector part.
# Applied after raw quaternion is received, before zeroing.
REMAP_PRESETS: Dict[str, List[Tuple[int, int]]] = {
    'identity':           [(0,+1),(1,+1),(2,+1)],
    'swap_yz':            [(0,+1),(2,+1),(1,+1)],
    'swap_yz_negz':       [(0,+1),(2,+1),(1,-1)],
    'swap_yz_negz_negx':  [(0,-1),(2,+1),(1,-1)],   # old default (lateral tilt flipped)
    'swap_yz_negz_negx_v2': [(0,-1),(2,+1),(1,+1)], # fixed: proc_qz = +raw_qy (lateral tilt correct)
    'swap_yz_negyz':      [(0,+1),(2,-1),(1,-1)],
    'swap_xy':            [(1,+1),(0,+1),(2,+1)],
    'swap_xz':            [(2,+1),(1,+1),(0,+1)],
    'neg_all':            [(0,-1),(1,-1),(2,-1)],
}
DEFAULT_REMAP = 'swap_yz_negz_negx_v2'

# ── limb lengths (metres) — average adult, adjustable per user ────────────────
@dataclass
class LimbLengths:
    pelvis_h:        float = 0.92    # pelvis height above floor in neutral stance
    pelvis_lumbar:   float = 0.10    # pelvis joint → lumbar vertebra
    lumbar_chest:    float = 0.15    # lumbar → chest (lower thoracic)
    chest_neck:      float = 0.20    # chest → neck base
    neck_head:       float = 0.10    # neck base → head centre
    head_top:        float = 0.13    # head centre → crown
    hip_offset_x:    float = 0.105   # lateral hip offset from pelvis centre
    hip_offset_y:    float = -0.05   # vertical hip offset (slightly below pelvis)
    thigh:           float = 0.42    # hip joint → knee joint
    shin:            float = 0.40    # knee joint → ankle joint
    foot:            float = 0.17    # ankle joint → toe tip
    foot_fwd:        float = 0.14    # ankle → ball of foot (forward component)
    clavicle:        float = 0.175   # chest centre → shoulder joint (half span)
    upper_arm:       float = 0.30    # shoulder → elbow
    forearm:         float = 0.26    # elbow → wrist
    hand:            float = 0.09    # wrist → fingertip midpoint

DEFAULT_LIMB_LENGTHS = LimbLengths()

# ── gravity ───────────────────────────────────────────────────────────────────
GRAVITY = 9.81   # m/s²

# ── motion detection tunables ─────────────────────────────────────────────────
@dataclass
class DetectorParams:
    travel_thresh:    float = 1.2    # m/s² horiz accel needed to count as travel
    bounce_thresh:    float = 1.5    # m/s² vertical oscillation per step
    run_cadence_hz:   float = 2.7    # step rate above which motion grades as run
    walk_cadence_hz:  float = 1.0    # step rate above which motion grades as walk
    dir_stability:    float = 0.6    # cos-similarity threshold for directional gate
    squat_vert_thresh:float = -0.5   # m/s² downward linear accel for squat detection
    squat_pitch_deg:  float = 8.0    # forward pitch (deg) required for squat confirm
    squat_hold_s:     float = 0.25   # seconds hold before squat is confirmed
    jump_launch_g:    float = 0.36   # vertical accel threshold (g) to detect push-off
    jump_flight_g:    float = 0.30   # accel magnitude (g) below which = airborne
    jump_impact_g:    float = 0.41   # downward accel (g) threshold for landing impact

DEFAULT_DETECTOR = DetectorParams()

# ── sensor fusion tunables ────────────────────────────────────────────────────
@dataclass
class FusionParams:
    alpha_default:     float = 0.93   # complementary filter gyro weight (general)
    alpha_head:        float = 0.97   # more smoothing for head (noisy)
    alpha_foot:        float = 0.96   # more smoothing for feet
    accel_lo_g:        float = 0.75   # lower g bound for accel trust band
    accel_hi_g:        float = 1.30   # upper g bound for accel trust band

DEFAULT_FUSION = FusionParams()

# ── biomechanical constraint hardness ─────────────────────────────────────────
# 1.0 = fully enforced, 0.0 = no constraint
CONSTRAINT_HARDNESS = 0.82

# ── IK ────────────────────────────────────────────────────────────────────────
IK_ENABLED       = True
IK_FLOOR_Y       = 0.0       # world-space Y of the ground plane
IK_FOOT_MARGIN   = 0.02      # metres above floor before IK activates

# ── analytics ─────────────────────────────────────────────────────────────────
@dataclass
class AnalyticsParams:
    window_s:        float = 3.0
    fatigue_decay:   float = 0.008
    fatigue_gain:    float = 0.006
    gyro_swing_thresh:float = 60.0   # deg/s above which shin counts as swinging

DEFAULT_ANALYTICS = AnalyticsParams()

# ── root motion ───────────────────────────────────────────────────────────────
ROOT_MAX_METRES  = 3.5       # clamp root travel radius before wrapping/resetting
MOVE_SPEED: Dict[str, float] = {
    'walk':         1.4,
    'walk_fwd':     1.4,
    'walk_back':    1.0,
    'strafe_left':  1.1,
    'strafe_right': 1.1,
    'run':          3.4,
    'march_in_place': 0.0,
}

# ── output format identifiers ─────────────────────────────────────────────────
OUTPUT_TARGETS = ['unity', 'unreal', 'blender', 'webgl', 'raw']

# ── Unity humanoid muscle names → R2P bone IDs ───────────────────────────────
UNITY_BONE_MAP: Dict[str, str] = {
    'Hips':           'pelvis',
    'Spine':          'lumbar',
    'Chest':          'chest',
    'Neck':           'neck',
    'Head':           'head',
    'LeftUpperLeg':   'thigh_l',
    'RightUpperLeg':  'thigh_r',
    'LeftLowerLeg':   'shin_l',
    'RightLowerLeg':  'shin_r',
    'LeftFoot':       'foot_l',
    'RightFoot':      'foot_r',
    'LeftShoulder':   'shoulder_l',
    'RightShoulder':  'shoulder_r',
    'LeftUpperArm':   'upper_arm_l',
    'RightUpperArm':  'upper_arm_r',
    'LeftLowerArm':   'forearm_l',
    'RightLowerArm':  'forearm_r',
}

# Reverse map
R2P_TO_UNITY: Dict[str, str] = {v: k for k, v in UNITY_BONE_MAP.items()}

# ── logging ───────────────────────────────────────────────────────────────────
CSV_HEADER = [
    'timestamp', 'sensor', 'label',
    'raw_qw', 'raw_qx', 'raw_qy', 'raw_qz',
    'proc_qw', 'proc_qx', 'proc_qy', 'proc_qz',
    'yaw_deg', 'pitch_deg', 'roll_deg',
    'accel_x', 'accel_y', 'accel_z',
    'gyro_x',  'gyro_y',  'gyro_z',
    'remap_preset', 'zeroed',
]