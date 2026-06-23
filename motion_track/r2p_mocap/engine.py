"""
R2P Inertial Motion Capture Engine

Orchestrates all subsystems:
  IMU frame → Madgwick fusion → Drift correction → Retargeting
  → Biomechanical constraints → FK → Foot contact → IK grounding
  → SkeletonPose output
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import logging
import time
import numpy as np

log = logging.getLogger("r2p.engine")

from .core.config   import R2PConfig, SensorID
from .core.skeleton import Bone, Skeleton, SkeletonPose
from .core          import quaternion as quat
from .core.imu_data import SuitFrame

from .fusion.madgwick            import MadgwickAHRS
from .calibration.calibrator     import Calibrator, CalibrationData
from .reconstruction.retargeter  import SensorToSkeletonRetargeter
from .reconstruction.root_solver import RootMotionSolver
from .biomechanics.constraints   import BiomechanicalConstraints
from .biomechanics.foot_contact  import FootContactDetector, FootState
from .drift.drift_corrector      import DriftCorrector
from .ik.ik_solver               import FootGroundingIK
from .output.skeleton_output     import SkeletonOutput, MotionCapFrame


@dataclass
class EngineStats:
    frames_processed: int   = 0
    avg_process_ms:   float = 0.0
    last_process_ms:  float = 0.0
    active_sensors:   int   = 0
    motion_state:     str   = "idle"


class R2PEngine:
    """
    Core mocap engine.  Feed SuitFrames, receive SkeletonPoses.

    Quick start:
        engine = R2PEngine(R2PConfig(mode=SuitMode.CONSUMER))
        engine.auto_calibrate()          # identity calibration for sim use
        for frame in simulator.stream(5.0, MotionType.WALK):
            pose = engine.process(frame)
    """

    def __init__(self, config: Optional[R2PConfig] = None):
        self.config = config or R2PConfig()

        # Sub-systems
        self._skeleton   = Skeleton(self.config.body_height)
        self._filters:   Dict[str, MadgwickAHRS]  = {}   # one per sensor
        self._calibrator = Calibrator(accumulation_frames=60)
        self._calibration: Optional[CalibrationData] = None
        self._retargeter: Optional[SensorToSkeletonRetargeter] = None
        self._root_solver = RootMotionSolver(
            body_height=self.config.body_height,
            dt=self.config.dt,
        )
        self._constraints   = BiomechanicalConstraints(strength=0.9)
        self._foot_detector = FootContactDetector()
        self._drift_corrector = DriftCorrector(
            correction_rate=self.config.drift_correction_rate,
        )
        self._ik = FootGroundingIK()

        # Smoothing buffers — per bone quaternion history
        self._smooth_q: Dict[Bone, np.ndarray] = {}

        # Output buffer
        self.output = SkeletonOutput(
            sample_rate=self.config.sample_rate,
            body_height=self.config.body_height,
        )

        # State
        self._stats = EngineStats()
        self._prev_orientations: Dict[str, np.ndarray] = {}
        self._frame_id = 0
        self._calibrated = False

        # Prefer imufusion if installed; fall back to Madgwick
        self._use_imufusion = False
        try:
            from .fusion.imufusion_ahrs import ImufusionAHRS
            if ImufusionAHRS.available():
                self._use_imufusion = True
                log.info("AHRS backend: imufusion (xioTechnologies) — gyro bias correction active")
        except Exception:
            pass
        if not self._use_imufusion:
            log.info("AHRS backend: Madgwick (built-in)")

        self._init_filters()

    # ── Initialisation ─────────────────────────────────────────────────────────

    def _make_filter(self):
        if self._use_imufusion:
            from .fusion.imufusion_ahrs import ImufusionAHRS
            return ImufusionAHRS(
                sample_rate=self.config.sample_rate,
                gain=max(0.1, self.config.madgwick_beta * 1.8),
            )
        return MadgwickAHRS(
            sample_rate=self.config.sample_rate,
            beta=self.config.madgwick_beta,
        )

    def _init_filters(self):
        for sid in self.config.active_sensors:
            self._filters[sid] = self._make_filter()

    def auto_calibrate(self, forward_yaw_deg: float = 0.0):
        """
        Set identity calibration — use when sensor axes are already aligned
        (simulator mode, or after hardware T-pose is handled externally).
        """
        self._calibration = Calibrator.from_identity(self.config.active_sensors)
        self._drift_corrector.set_reference_yaw(np.radians(forward_yaw_deg))
        self._retargeter = SensorToSkeletonRetargeter(self._calibration)
        self._calibrated = True

    def begin_calibration(self):
        """Start accumulating frames for T-pose calibration."""
        self._calibrator.reset()
        self._calibrated = False

    def feed_calibration_frame(self, orientations: Dict[str, np.ndarray]) -> bool:
        """
        Feed one frame of world-orientations during the calibration hold.
        Returns True when enough frames have been collected.
        """
        return self._calibrator.add_frame(orientations)

    def finish_calibration(self, timestamp: float = 0.0) -> CalibrationData:
        """Finalise calibration and activate it."""
        self._calibration = self._calibrator.compute("tpose", timestamp)
        self._retargeter  = SensorToSkeletonRetargeter(self._calibration)
        self._drift_corrector.set_reference_yaw(
            float(np.arctan2(
                self._calibration.forward[0],
                self._calibration.forward[2],
            ))
        )
        self._calibrated = True
        return self._calibration

    # ── Main processing loop ───────────────────────────────────────────────────

    def process(self, suit_frame: SuitFrame) -> SkeletonPose:
        """
        Process one SuitFrame and return a solved SkeletonPose.
        Also appends a MotionCapFrame to self.output.
        """
        t0 = time.perf_counter()

        if not self._calibrated:
            self.auto_calibrate()

        # ── 1. Sensor fusion (Madgwick per-sensor) ─────────────────────
        orientations: Dict[str, np.ndarray] = {}
        gyros:        Dict[str, np.ndarray] = {}
        accels:       Dict[str, np.ndarray] = {}

        for sid, reading in suit_frame.readings.items():
            if reading.quaternion is not None:
                # Hardware already fused the quaternion — skip Madgwick entirely
                q = quat.normalize(reading.quaternion)
            else:
                filt = self._filters.get(sid)
                if filt is None:
                    filt = self._make_filter()
                    self._filters[sid] = filt
                q = filt.update(reading.gyro, reading.accel, reading.mag)
            orientations[sid] = q
            gyros[sid]  = reading.gyro
            accels[sid] = reading.accel

        # ── 2. Drift correction ────────────────────────────────────────
        if self.config.enable_drift_correction:
            orientations = self._drift_corrector.correct(orientations, gyros, accels)

        # ── 3. Quaternion smoothing ────────────────────────────────────
        alpha = self.config.smoothing_alpha
        for sid, q in orientations.items():
            prev = self._prev_orientations.get(sid, q)
            orientations[sid] = quat.smooth_exp(prev, q, alpha)
        self._prev_orientations = {k: v.copy() for k, v in orientations.items()}

        # Expose fused orientations so the server can forward them to the browser
        self.last_orientations: Dict[str, np.ndarray] = {k: v.copy() for k, v in orientations.items()}

        # ── 4. Retarget → local bone rotations ────────────────────────
        local_rotations = self._retargeter.retarget(orientations)

        # ── 5. Biomechanical constraints ───────────────────────────────
        local_rotations = self._constraints.apply(local_rotations)

        # ── 6. Root motion solving ─────────────────────────────────────
        root_pos = self._root_solver.update(
            pelvis_orientation   = orientations.get(SensorID.PELVIS,  quat.IDENTITY),
            l_thigh_orientation  = orientations.get(SensorID.L_THIGH),
            r_thigh_orientation  = orientations.get(SensorID.R_THIGH),
            l_shin_orientation   = orientations.get(SensorID.L_SHIN),
            r_shin_orientation   = orientations.get(SensorID.R_SHIN),
        )

        # ── 7. Forward kinematics ──────────────────────────────────────
        pose = self._skeleton.forward_kinematics(local_rotations, root_pos)
        pose.timestamp = suit_frame.timestamp
        pose.frame_id  = self._frame_id
        self._frame_id += 1

        # ── 8. Foot contact detection ──────────────────────────────────
        foot_states: Dict[str, FootState] = {}
        if self.config.enable_foot_contact:
            foot_states = self._foot_detector.update(
                l_shin_q = orientations.get(SensorID.L_SHIN),
                r_shin_q = orientations.get(SensorID.R_SHIN),
                l_foot_q = orientations.get(SensorID.L_FOOT),
                r_foot_q = orientations.get(SensorID.R_FOOT),
            )

        # ── 9. Foot grounding IK ───────────────────────────────────────
        if self.config.enable_ik and foot_states:
            left_contact  = foot_states.get("left")  and foot_states["left"].contact
            right_contact = foot_states.get("right") and foot_states["right"].contact
            self._ik.apply(
                pose,
                ground_y=0.0,
                left_contact=bool(left_contact),
                right_contact=bool(right_contact),
            )

        # ── 10. Emit output frame ──────────────────────────────────────
        motion_label = self._classify_motion()
        mcf = MotionCapFrame.from_skeleton_pose(
            pose,
            root_vel=self._root_solver.state.velocity,
            foot_states=foot_states,
            motion_state=motion_label,
        )
        self.output.add_frame(mcf)

        # ── Stats ──────────────────────────────────────────────────────
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._stats.last_process_ms = elapsed_ms
        self._stats.avg_process_ms  = (
            self._stats.avg_process_ms * 0.95 + elapsed_ms * 0.05
        )
        self._stats.frames_processed += 1
        self._stats.active_sensors    = len(suit_frame.readings)
        self._stats.motion_state      = motion_label

        return pose

    def process_batch(self, frames: List[SuitFrame]) -> List[SkeletonPose]:
        return [self.process(f) for f in frames]

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _classify_motion(self) -> str:
        rs = self._root_solver.state
        if not rs.is_moving:
            return "idle"
        s = rs.speed_estimate
        if s < 0.5:
            return "walk_slow"
        if s < 2.0:
            return "walk"
        if s < 4.5:
            return "run"
        return "sprint"

    def reset(self):
        """Reset all stateful sub-systems (keeps calibration)."""
        for filt in self._filters.values():
            filt.reset()
        self._root_solver.reset()
        self._smooth_q.clear()
        self._prev_orientations.clear()
        self._frame_id = 0
        self._stats = EngineStats()

    @property
    def stats(self) -> EngineStats:
        return self._stats

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def skeleton(self) -> Skeleton:
        return self._skeleton
