"""
Realistic IMU simulator for the R2P mocap suit.

Generates per-sensor gyro/accelerometer data that reproduces parametric
human gait cycles (walk, run, idle) and maps to each suit sensor.

Coordinate system: Y-up.
Accelerometer output includes gravity (as a real sensor does).
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional
import numpy as np

from ..core import quaternion as quat
from ..core.imu_data import IMUReading, SuitFrame
from ..core.config import SensorID, CONSUMER_SENSORS, PROFESSIONAL_SENSORS


_G = 9.81                      # m/s²
_DEG = np.pi / 180.0


class MotionType(Enum):
    IDLE         = "idle"
    WALK         = "walk"
    RUN          = "run"
    SQUAT        = "squat"
    LUNGE        = "lunge"
    SIDE_STEP    = "side_step"
    PIVOT        = "pivot"


# ── Parametric gait curves ────────────────────────────────────────────────────

def _gait_hip_flex(phase: float, amplitude: float) -> float:
    """Hip flexion/extension over a normalised gait cycle [0–1]."""
    # Positive = flexion (forward swing), negative = extension (push-off)
    return amplitude * np.sin(2 * np.pi * phase)


def _gait_knee_flex(phase: float, amplitude: float) -> float:
    """Knee flexion — always ≥ 0 (anatomical hinge), peaks at mid-swing."""
    raw = amplitude * np.sin(2 * np.pi * (phase - 0.15))
    return float(np.clip(raw, 0.0, amplitude))


def _gait_ankle_dflex(phase: float, amplitude: float) -> float:
    """Ankle dorsiflexion — negative = plantarflexion (toe-off push)."""
    return amplitude * np.sin(2 * np.pi * (phase + 0.25))


def _pelvis_tilt_roll(phase: float, amplitude: float) -> float:
    """Pelvis lateral tilt (roll) in sync with steps."""
    return amplitude * np.sin(2 * np.pi * phase)


def _pelvis_yaw(phase: float, amplitude: float) -> float:
    """Pelvis axial rotation opposing hip swing."""
    return amplitude * np.sin(2 * np.pi * phase)


# ── Segment angle model for each motion type ─────────────────────────────────

@dataclass
class GaitParams:
    cadence:          float   # steps/sec (each step = 0.5 gait cycle)
    hip_amplitude:    float   # degrees
    knee_amplitude:   float   # degrees
    ankle_amplitude:  float   # degrees
    pelvis_tilt:      float   # degrees lateral roll
    pelvis_yaw:       float   # degrees axial
    chest_yaw:        float   # degrees counter-rotation
    chest_pitch:      float   # forward lean degrees
    arm_swing:        float   # degrees (shoulder flex/ext)
    vertical_bob:     float   # metres root oscillation


_GAIT_TABLE: Dict[MotionType, GaitParams] = {
    MotionType.IDLE: GaitParams(
        cadence=0.0, hip_amplitude=2, knee_amplitude=5, ankle_amplitude=2,
        pelvis_tilt=1, pelvis_yaw=1, chest_yaw=1, chest_pitch=0,
        arm_swing=3, vertical_bob=0.005,
    ),
    MotionType.WALK: GaitParams(
        cadence=1.8, hip_amplitude=30, knee_amplitude=60, ankle_amplitude=20,
        pelvis_tilt=5, pelvis_yaw=6, chest_yaw=4, chest_pitch=3,
        arm_swing=25, vertical_bob=0.025,
    ),
    MotionType.RUN: GaitParams(
        cadence=3.0, hip_amplitude=45, knee_amplitude=90, ankle_amplitude=30,
        pelvis_tilt=8, pelvis_yaw=10, chest_yaw=6, chest_pitch=8,
        arm_swing=55, vertical_bob=0.08,
    ),
    MotionType.SQUAT: GaitParams(
        cadence=0.5, hip_amplitude=70, knee_amplitude=110, ankle_amplitude=20,
        pelvis_tilt=2, pelvis_yaw=0, chest_yaw=0, chest_pitch=15,
        arm_swing=10, vertical_bob=0.15,
    ),
    MotionType.LUNGE: GaitParams(
        cadence=0.4, hip_amplitude=80, knee_amplitude=90, ankle_amplitude=15,
        pelvis_tilt=3, pelvis_yaw=5, chest_yaw=3, chest_pitch=10,
        arm_swing=15, vertical_bob=0.10,
    ),
    MotionType.SIDE_STEP: GaitParams(
        cadence=1.5, hip_amplitude=20, knee_amplitude=35, ankle_amplitude=10,
        pelvis_tilt=8, pelvis_yaw=3, chest_yaw=2, chest_pitch=2,
        arm_swing=10, vertical_bob=0.02,
    ),
    MotionType.PIVOT: GaitParams(
        cadence=1.2, hip_amplitude=15, knee_amplitude=30, ankle_amplitude=10,
        pelvis_tilt=4, pelvis_yaw=20, chest_yaw=5, chest_pitch=5,
        arm_swing=15, vertical_bob=0.015,
    ),
}


# ── IMUSimulator ──────────────────────────────────────────────────────────────

class IMUSimulator:
    """
    Generates SuitFrame streams with realistic IMU data for any MotionType.

    Usage:
        sim = IMUSimulator(sample_rate=100, mode="professional")
        for frame in sim.stream(duration=5.0, motion=MotionType.WALK):
            engine.process(frame)
    """

    NOISE_GYRO  = 0.005   # rad/s RMS
    NOISE_ACCEL = 0.15    # m/s² RMS
    NOISE_MAG   = 0.02    # normalised units

    def __init__(
        self,
        sample_rate: float = 100.0,
        mode:        str   = "consumer",   # "consumer" or "professional"
        noise:       bool  = True,
        body_dir_deg: float = 0.0,         # initial facing direction (yaw)
    ):
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        self.sensors = (
            PROFESSIONAL_SENSORS if mode == "professional" else CONSUMER_SENSORS
        )
        self.noise = noise
        self._body_yaw = np.radians(body_dir_deg)
        self._prev_quats: Dict[str, np.ndarray] = {}
        self._phase = 0.0
        self._frame_id = 0
        self._t = 0.0

    # ------------------------------------------------------------------
    def stream(
        self,
        duration:   float,
        motion:     MotionType = MotionType.WALK,
        motion_seq: Optional[List[Tuple[float, MotionType]]] = None,
    ):
        """
        Generator yielding SuitFrame objects.
        motion_seq: [(t0, motion0), (t1, motion1), ...] for time-varying motion.
        """
        n_frames = int(duration * self.sample_rate)
        for _ in range(n_frames):
            if motion_seq:
                current_motion = motion
                for t_start, m in motion_seq:
                    if self._t >= t_start:
                        current_motion = m
            else:
                current_motion = motion

            frame = self._generate_frame(current_motion)
            self._t += self.dt
            yield frame

    def generate_frame(self, motion: MotionType = MotionType.WALK) -> SuitFrame:
        return self._generate_frame(motion)

    # ------------------------------------------------------------------
    def _generate_frame(self, motion: MotionType) -> SuitFrame:
        p = _GAIT_TABLE[motion]

        # Advance gait phase
        if p.cadence > 0:
            self._phase = (self._phase + p.cadence * self.dt) % 1.0
        else:
            self._phase = (self._phase + 0.003) % 1.0  # idle sway

        ph = self._phase
        frame = SuitFrame(
            timestamp=self._t,
            frame_id=self._frame_id,
        )
        self._frame_id += 1

        # ── World-frame orientations for each sensor ───────────────────
        quats = self._compute_segment_quats(ph, p)

        for sid in self.sensors:
            q = quats.get(sid)
            if q is None:
                q = quat.IDENTITY.copy()

            # Angular velocity from quaternion derivative
            q_prev = self._prev_quats.get(sid, q)
            gyro = self._quat_to_gyro(q_prev, q, self.dt)

            # Accelerometer: gravity rotated into sensor frame + linear component
            g_world = np.array([0.0, _G, 0.0])
            g_sensor = quat.rotate_vector(quat.inverse(q), g_world)
            lin_accel = self._estimate_linear_accel(sid, p, ph)
            lin_sensor = quat.rotate_vector(quat.inverse(q), lin_accel)
            accel = g_sensor + lin_sensor

            # Noise
            if self.noise:
                gyro  += np.random.normal(0, self.NOISE_GYRO, 3)
                accel += np.random.normal(0, self.NOISE_ACCEL, 3)

            self._prev_quats[sid] = q.copy()
            frame.add(IMUReading(
                sensor_id=sid,
                timestamp=self._t,
                gyro=gyro,
                accel=accel,
            ))

        return frame

    # ------------------------------------------------------------------
    def _compute_segment_quats(
        self, ph: float, p: GaitParams
    ) -> Dict[str, np.ndarray]:
        """Build world-frame quaternion for every sensor at gait phase ph."""
        BY = self._body_yaw
        d = _DEG

        # Left phase leads, right phase offset by 0.5
        ph_l = ph
        ph_r = (ph + 0.5) % 1.0

        # ── Pelvis ────────────────────────────────────────────────────
        pel_yaw   = BY + _pelvis_yaw(ph, p.pelvis_yaw) * d
        pel_pitch = np.radians(-p.chest_pitch * 0.3)  # slight forward lean
        pel_roll  = _pelvis_tilt_roll(ph, p.pelvis_tilt) * d
        q_pelvis  = quat.from_euler_yup(pel_yaw, pel_pitch, pel_roll)

        # ── Chest ─────────────────────────────────────────────────────
        ch_yaw   = BY - _pelvis_yaw(ph, p.chest_yaw) * d    # counter-rotation
        ch_pitch = np.radians(p.chest_pitch)
        ch_roll  = -pel_roll * 0.3
        q_chest  = quat.from_euler_yup(ch_yaw, ch_pitch, ch_roll)

        # ── Thighs (hip flex/ext) ──────────────────────────────────────
        hip_l = _gait_hip_flex(ph_l, p.hip_amplitude) * d
        hip_r = _gait_hip_flex(ph_r, p.hip_amplitude) * d
        q_lthigh = quat.normalize(quat.multiply(
            q_pelvis, quat.from_euler_yup(0, hip_l, 0)
        ))
        q_rthigh = quat.normalize(quat.multiply(
            q_pelvis, quat.from_euler_yup(0, hip_r, 0)
        ))

        # ── Shins (knee flex) ──────────────────────────────────────────
        kn_l = _gait_knee_flex(ph_l, p.knee_amplitude) * d
        kn_r = _gait_knee_flex(ph_r, p.knee_amplitude) * d
        q_lshin = quat.normalize(quat.multiply(
            q_lthigh, quat.from_euler_yup(0, -kn_l, 0)
        ))
        q_rshin = quat.normalize(quat.multiply(
            q_rthigh, quat.from_euler_yup(0, -kn_r, 0)
        ))

        # ── Feet (ankle dflex) ─────────────────────────────────────────
        ank_l = _gait_ankle_dflex(ph_l, p.ankle_amplitude) * d
        ank_r = _gait_ankle_dflex(ph_r, p.ankle_amplitude) * d
        q_lfoot = quat.normalize(quat.multiply(
            q_lshin, quat.from_euler_yup(0, ank_l, 0)
        ))
        q_rfoot = quat.normalize(quat.multiply(
            q_rshin, quat.from_euler_yup(0, ank_r, 0)
        ))

        # ── Head ──────────────────────────────────────────────────────
        q_head = quat.from_euler_yup(BY, 0, 0)  # facing forward

        # ── Shoulders / upper arms (arm swing) ────────────────────────
        arm_l = -_gait_hip_flex(ph_l, p.arm_swing) * d   # counter-phase to same leg
        arm_r = -_gait_hip_flex(ph_r, p.arm_swing) * d

        # Left shoulder: attached to chest, arm swings forward/back
        q_lshoulder = q_chest.copy()
        q_rshoulder = q_chest.copy()
        # Upper arm: relative to shoulder, rotates in sagittal plane
        q_l_upper_arm = quat.normalize(quat.multiply(
            q_lshoulder, quat.from_euler_yup(0, arm_l, 0)
        ))
        q_r_upper_arm = quat.normalize(quat.multiply(
            q_rshoulder, quat.from_euler_yup(0, arm_r, 0)
        ))
        # Forearms: slight elbow bend during running
        elbow_flex = np.radians(30 + (p.arm_swing / 55.0) * 60)
        q_l_forearm = quat.normalize(quat.multiply(
            q_l_upper_arm, quat.from_euler_yup(0, -elbow_flex, 0)
        ))
        q_r_forearm = quat.normalize(quat.multiply(
            q_r_upper_arm, quat.from_euler_yup(0, -elbow_flex, 0)
        ))

        return {
            SensorID.PELVIS:      q_pelvis,
            SensorID.CHEST:       q_chest,
            SensorID.L_THIGH:     q_lthigh,
            SensorID.R_THIGH:     q_rthigh,
            SensorID.L_SHIN:      q_lshin,
            SensorID.R_SHIN:      q_rshin,
            SensorID.L_FOOT:      q_lfoot,
            SensorID.R_FOOT:      q_rfoot,
            SensorID.HEAD:        q_head,
            SensorID.L_SHOULDER:  q_lshoulder,
            SensorID.R_SHOULDER:  q_rshoulder,
            SensorID.L_UPPER_ARM: q_l_upper_arm,
            SensorID.R_UPPER_ARM: q_r_upper_arm,
            SensorID.L_FOREARM:   q_l_forearm,
            SensorID.R_FOREARM:   q_r_forearm,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _quat_to_gyro(q1: np.ndarray, q2: np.ndarray, dt: float) -> np.ndarray:
        """Numerical differentiation of quaternion → body-frame angular velocity."""
        dq = quat.multiply(quat.inverse(q1), q2)
        # dq ≈ [1, ω*dt/2]  for small rotations
        angle = 2.0 * np.arccos(np.clip(abs(dq[0]), 0.0, 1.0))
        if abs(dq[0]) < 1.0 - 1e-8:
            axis = dq[1:] / np.linalg.norm(dq[1:] + 1e-12)
        else:
            axis = np.array([0.0, 1.0, 0.0])
        omega_world = axis * (angle / max(dt, 1e-9))
        # Rotate into sensor body frame
        return quat.rotate_vector(quat.inverse(q2), omega_world)

    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_linear_accel(sid: str, p: GaitParams, ph: float) -> np.ndarray:
        """Rough vertical bob acceleration component (world frame)."""
        if p.vertical_bob < 0.001:
            return np.zeros(3)
        # Second derivative of A*sin(2π*f*t) = -(2πf)² * A * sin(…)
        freq = p.cadence * 2          # up-down at double step frequency
        accel_y = -(2 * np.pi * freq)**2 * p.vertical_bob * np.sin(2 * np.pi * ph)
        return np.array([0.0, accel_y, 0.0])
