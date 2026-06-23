"""
backend/root_integrator.py  (v7 — NBA2K-style smooth motion + calibration capture)
-----------------------------------------------------------------------------------
Converts raw pelvis IMU data into world-space (dx, dz, dy) displacement deltas
for SkeletonEngine.update_root_translation(dx, dz, dy), with smooth momentum-
carried locomotion and per-user motion calibration.

v7 CHANGES vs v6
----------------
1. VELOCITY-DAMPED LOCOMOTION (the smoothness fix). v6 spread each step's
   displacement over a Hann window — this lurches (steady-phase velocity ripple
   ~1.46 m/s in testing). v7 instead models a *velocity* that is critically
   eased toward a target each frame; position integrates that velocity. Result:
   steady-phase ripple ~0.01 m/s — a continuous glide between footfalls, like a
   game character. No overshoot (exponential approach, not spring).

2. SPEED GRADING (walk vs run). Target speed is derived from step cadence:
   speed = stride_length / cadence. Faster cadence -> higher target speed -> the
   avatar automatically walks, jogs, or runs. A gait label (idle/walk/jog/run)
   is exposed for the frontend.

3. PER-USER CALIBRATION CAPTURE. A guided routine (CalibrationCapture) records
   the user's own walk-bounce amplitude, jump launch magnitude, and squat
   descent depth, then auto-sets detection thresholds to that body.

4. MEASURED VERTICAL EVENTS (for biomechanics). Jump and squat emit event
   records with measured quantities via on_event, so biomechanics can score
   them. Visual dy arcs are still produced.

5. dy OUTPUT retained. update() returns (dx, dz, dy).

BACK-COMPAT
-----------
update(quaternion, accel_sensor) -> (dx, dz, dy). All v4/v5 tuning shims
preserved. translation_speed_ms now reflects the smoothed avatar speed.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Callable, Optional


def _rotate_vec_by_quat(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    def _mul(a, b):
        return np.array([
            a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
            a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
            a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
            a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
        ])
    qv     = np.array([0.0, v[0], v[1], v[2]])
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    return _mul(_mul(q, qv), q_conj)[1:]


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Return a unit quaternion; falls back to identity if degenerate."""
    n = float(np.linalg.norm(q))
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


class GravityEstimator:
    """Running-mean gravity vector in sensor frame. Axis-convention agnostic."""

    def __init__(self, n_init: int = 140, lr_still: float = 0.001) -> None:
        self._n_init   = n_init
        self._lr_still = lr_still
        self._gravity  = np.zeros(3)
        self._frames   = 0
        self._ready    = False
        self._warmup   = []

    def reset(self) -> None:
        self._gravity[:] = 0.0
        self._frames     = 0
        self._ready      = False
        self._warmup     = []

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def gravity(self) -> np.ndarray:
        return self._gravity.copy()

    def update(self, accel_sensor: np.ndarray, is_quiet: bool) -> None:
        if not self._ready:
            self._warmup.append(accel_sensor.copy())
            self._frames += 1
            if self._frames >= self._n_init:
                self._gravity = np.median(np.array(self._warmup), axis=0)
                self._warmup  = []
                self._ready   = True
        elif is_quiet:
            self._gravity = (1.0 - self._lr_still) * self._gravity \
                + self._lr_still * accel_sensor

    def linear(self, accel_sensor: np.ndarray) -> np.ndarray:
        return accel_sensor - self._gravity


class _EaseSmoother:
    """Spreads a discrete vertical displacement across n_frames via a Hann
    window. Used ONLY for jump/squat arcs — discrete events that should ease."""

    def __init__(self, n_frames: int = 30) -> None:
        self._n = n_frames
        w = np.hanning(n_frames)
        self._weights = w / w.sum()
        self._pending: deque = deque()

    def reset(self) -> None:
        self._pending.clear()

    def add(self, dy: float) -> None:
        self._pending.append([dy, 0])

    def tick(self) -> float:
        out = 0.0
        finished = []
        for entry in self._pending:
            dy, k = entry
            if k < self._n:
                out += dy * float(self._weights[k])
                entry[1] += 1
            else:
                finished.append(entry)
        for f in finished:
            self._pending.remove(f)
        return out

    @property
    def active(self) -> bool:
        return len(self._pending) > 0


class _VerticalClassifier:
    """Classifies jump and squat from gravity-axis linear accel (vert_up,
    positive=up). Emits measured event dicts via callbacks."""

    def __init__(self, fps: float,
                 launch_thresh:  float = 8.0,
                 ff_thresh:      float = 2.0,
                 min_ff:         int   = 5,
                 descend_thresh: float = 1.8,
                 min_descend:    int   = 5) -> None:
        self._fps            = fps
        self._dt             = 1.0 / fps
        self._launch_thresh  = launch_thresh
        self._ff_thresh      = ff_thresh
        self._min_ff         = min_ff
        self._descend_thresh = descend_thresh
        self._min_descend    = min_descend

        self._await_ff   = False
        self._await_cnt  = 0
        self._ff_run     = 0
        self._launch_v   = 0.0
        self._descend_run = 0
        self._in_squat    = False
        self._refractory  = 0          # jump refractory (post-jump debounce)
        self._squat_refractory = 0     # v7-fix: separate, so squat can't block a jump launch
        self._frame       = 0
        self._last_squat_frame = -999

    def reset(self) -> None:
        self._await_ff = False
        self._await_cnt = 0
        self._ff_run = 0
        self._launch_v = 0.0
        self._descend_run = 0
        self._in_squat = False
        self._refractory = 0
        self._squat_refractory = 0
        self._frame = 0
        self._last_squat_frame = -999

    def update(self, vert: float, on_jump: Callable, on_squat: Callable,
               walking: bool = False, on_cancel_squat: Callable = None) -> None:
        self._frame += 1
        if self._refractory > 0:
            self._refractory -= 1
        if self._squat_refractory > 0:
            self._squat_refractory -= 1

        # JUMP — gated only by the jump refractory, NOT by squat refractory, so
        # a jump's launch can fire immediately after the load-phase crouch.
        if not self._await_ff and vert > self._launch_thresh and self._refractory == 0:
            self._await_ff  = True
            self._await_cnt = 0
            self._ff_run    = 0
            self._launch_v  = vert
            # v7-fix: a jump's pre-launch crouch can look like a squat descent.
            # If a squat was emitted recently (within ~0.6s) and a launch spike
            # now appears, retract it — it was the load phase. Also clear the
            # squat refractory so the jump isn't suppressed.
            recent = (self._frame - self._last_squat_frame) < int(0.6 * self._fps)
            if (self._in_squat or recent) and on_cancel_squat is not None:
                on_cancel_squat()
            self._in_squat        = False
            self._descend_run     = 0
            self._squat_refractory = 0
        elif self._await_ff:
            self._await_cnt += 1
            if abs(vert) < self._ff_thresh:
                self._ff_run += 1
                if self._ff_run >= self._min_ff:
                    flight_time = self._ff_run * self._dt
                    est_height = float(np.clip(9.81 * flight_time**2 / 8.0, 0.03, 0.8))
                    on_jump({
                        "type":        "jump",
                        "flight_time": round(flight_time, 3),
                        "est_height":  round(est_height, 3),
                        "launch_acc":  round(self._launch_v, 2),
                    })
                    self._await_ff   = False
                    self._refractory = int(0.6 * self._fps)
            else:
                self._ff_run = 0
            if self._await_cnt > int(0.5 * self._fps):
                self._await_ff = False

        # SQUAT (suppressed while walking)
        if not self._await_ff and not walking:
            if vert < -self._descend_thresh:
                self._descend_run += 1
                if (self._descend_run >= self._min_descend
                        and not self._in_squat and self._squat_refractory == 0):
                    descent_time = self._descend_run * self._dt
                    depth = float(np.clip(0.5 * descent_time * abs(vert) * 0.1
                                          + 0.04 * abs(vert), 0.08, 0.5))
                    on_squat({
                        "type":         "squat",
                        "descent_time": round(descent_time, 3),
                        "depth":        round(depth, 3),
                    })
                    self._in_squat   = True
                    self._last_squat_frame = self._frame
                    self._squat_refractory = int(0.5 * self._fps)
            else:
                self._descend_run = 0
                if self._in_squat and vert > self._descend_thresh * 0.5:
                    self._in_squat = False


class GaitConfirmation:
    """
    Multi-IMU translation gate.

    Decides how confident we are that the user is *actually locomoting* (walking
    or running) versus just bobbing the pelvis in place. The discriminator is
    the LEFT/RIGHT THIGH swing relationship measured from the leg IMUs:

        - Real gait  -> thighs swing ANTI-PHASE (one forward while the other is
          back) -> negative L/R correlation + sufficient swing energy.
        - Bouncing in place -> thighs move IN-PHASE or barely at all
          -> non-negative correlation and/or low energy -> confidence ~0.
        - Standing -> negligible limb energy -> confidence 0.

    Returns a smoothed 0..1 confidence that the integrator multiplies into the
    per-step displacement (a SOFT gate, so transitions glide).

    Fallback: if no leg sensors have reported recently, confidence falls back to
    `no_limb_confidence` (default 1.0 = preserve pelvis-only behaviour) and the
    integrator flags the translation as `confirmed=False`.

    Feed it the thigh swing-axis angular rate (rad/s or deg/s — units only need
    to be consistent). The swing axis is typically the thigh IMU's gyro X.
    """

    def __init__(self, fps: float, win_s: float = 0.8,
                 energy_ref: float = 1.5,
                 min_energy: float = 0.3,
                 smooth: float = 0.15,
                 stale_s: float = 0.4,
                 no_limb_confidence: float = 1.0) -> None:
        self._fps     = fps
        self._n       = max(4, int(win_s * fps))
        self._lbuf    = deque(maxlen=self._n)
        self._rbuf    = deque(maxlen=self._n)
        self._energy_ref = energy_ref
        self._min_energy = min_energy
        self._smooth     = smooth
        self._stale_frames = int(stale_s * fps)
        self._no_limb_conf = no_limb_confidence
        self.confidence  = no_limb_confidence
        self._since_limb = 10_000          # frames since a leg sample arrived
        self.has_limbs   = False

    def reset(self) -> None:
        self._lbuf.clear()
        self._rbuf.clear()
        self.confidence  = self._no_limb_conf
        self._since_limb = 10_000
        self.has_limbs   = False

    def push_thigh(self, side: str, swing_rate: float) -> None:
        """Record a thigh swing-axis angular rate sample. side in {'l','r'}."""
        if side == "l":
            self._lbuf.append(float(swing_rate))
        elif side == "r":
            self._rbuf.append(float(swing_rate))
        self._since_limb = 0
        self.has_limbs   = True

    def tick(self) -> float:
        """
        Advance one frame; returns the current confidence 0..1.
        Call once per pelvis frame so staleness tracks real time.
        """
        self._since_limb += 1

        # No recent leg data -> fall back (pelvis-only mode).
        if self._since_limb > self._stale_frames:
            self.has_limbs = False
            target = self._no_limb_conf
            self.confidence += (target - self.confidence) * self._smooth
            return self.confidence

        self.has_limbs = True
        # Need both buffers reasonably full to judge phase.
        if len(self._lbuf) < self._n // 2 or len(self._rbuf) < self._n // 2:
            return self.confidence

        m = min(len(self._lbuf), len(self._rbuf))
        L = np.array(list(self._lbuf)[-m:])
        R = np.array(list(self._rbuf)[-m:])

        if np.std(L) < self._min_energy or np.std(R) < self._min_energy:
            target = 0.0                     # limbs essentially still
        else:
            corr = float(np.corrcoef(L, R)[0, 1])
            antiphase = float(np.clip(-corr, 0.0, 1.0))   # 1 if perfectly anti-phase
            energy = float(np.clip((np.std(L) + np.std(R)) / 2.0 / self._energy_ref, 0.0, 1.0))
            target = antiphase * energy

        self.confidence += (target - self.confidence) * self._smooth
        return self.confidence


class CalibrationCapture:
    """Records a user's motion signatures during a guided routine so detection
    thresholds adapt to that body."""

    PHASES = ("walk_forward", "walk_back", "walk_left", "walk_right",
              "jump", "squat")

    def __init__(self, fps: float) -> None:
        self._fps = fps
        self.reset()

    def reset(self) -> None:
        self._phase: Optional[str] = None
        self._bounce_peaks: list = []
        self._launch_peaks: list = []
        self._descend_peaks: list = []
        self._surge_by_phase: dict = {p: [] for p in self.PHASES}
        self._frame = 0

    def set_phase(self, phase: str) -> None:
        if phase not in self.PHASES:
            raise ValueError(f"unknown calibration phase: {phase}")
        self._phase = phase

    def feed(self, vert_up: float, horiz_surge_world: np.ndarray) -> None:
        if self._phase is None:
            return
        self._frame += 1
        hmag = float(np.linalg.norm(horiz_surge_world[[0, 2]]))
        if self._phase.startswith("walk"):
            self._bounce_peaks.append(abs(vert_up))
            if hmag > 0.3:
                self._surge_by_phase[self._phase].append(horiz_surge_world.copy())
        elif self._phase == "jump":
            if vert_up > 2.0:
                self._launch_peaks.append(vert_up)
        elif self._phase == "squat":
            if vert_up < -1.0:
                self._descend_peaks.append(abs(vert_up))

    def summarize(self) -> dict:
        def pct(arr, p, default):
            return float(np.percentile(arr, p)) if len(arr) >= 3 else default
        bounce_typical = pct(self._bounce_peaks, 60, 3.5)
        launch_typical = pct(self._launch_peaks, 70, 8.0)
        descend_typ    = pct(self._descend_peaks, 70, 1.8)
        step_thresh   = float(np.clip(0.55 * bounce_typical, 1.8, 8.0))
        launch_thresh = float(np.clip(0.6 * launch_typical,
                                      max(4.0, 1.5*bounce_typical), 30.0))
        descend_thresh = float(np.clip(0.6 * descend_typ, 1.2, 5.0))
        return {
            "step_accel_thresh":    round(step_thresh, 2),
            "jump_launch_thresh":   round(launch_thresh, 2),
            "squat_descend_thresh": round(descend_thresh, 2),
            "signatures": {
                "walk_bounce_p60":   round(bounce_typical, 2),
                "jump_launch_p70":   round(launch_typical, 2),
                "squat_descend_p70": round(descend_typ, 2),
                "samples": {
                    "bounce":  len(self._bounce_peaks),
                    "launch":  len(self._launch_peaks),
                    "descend": len(self._descend_peaks),
                },
            },
        }


class RootIntegrator:
    """v7: smooth momentum-carried locomotion + 3D jump/squat + calibration."""

    _GAIT_BANDS = [(0.2, "idle"), (1.4, "walk"), (2.5, "jog"), (99.0, "run")]

    def __init__(
        self,
        fps:                  float = 50.0,
        stride_length:        float = 0.65,
        step_accel_thresh:    float = 3.5,
        min_step_period_s:    float = 0.18,
        heading_smooth:       float = 0.5,
        accel_unit_scale:     float = 0.0,
        n_gravity_init:       int   = 140,
        forward_axis:         tuple = (0.0, 0.0, 1.0),
        adaptive_thresh:      bool  = True,
        auto_forward:         bool  = True,
        fwd_calib_steps:      int   = 5,
        omnidirectional:      bool  = True,
        move_tau:             float = 0.18,
        stop_decay:           float = 0.90,
        max_speed:            float = 4.0,
        enable_vertical:      bool  = True,
        jump_smooth_frames:   int   = 30,
        squat_smooth_frames:  int   = 40,
        on_event:             Optional[Callable[[dict], None]] = None,
        hp_alpha:        float | None = None,
        velocity_decay:  float | None = None,
        accel_threshold: float | None = None,
        scale:           float | None = None,
    ) -> None:
        self._dt  = 1.0 / fps
        self._fps = fps

        if scale           is not None: stride_length     = 0.65 * float(scale)
        if accel_threshold is not None: step_accel_thresh = max(1.8, float(accel_threshold) * 10.0)
        if velocity_decay  is not None: heading_smooth    = float(np.clip(velocity_decay, 0.05, 0.95))
        self._accel_smooth_alpha = float(hp_alpha) if hp_alpha is not None else 0.6

        self._stride_length     = stride_length
        self._step_accel_thresh = step_accel_thresh
        self._min_step_frames   = max(1, int(min_step_period_s * fps))
        self._heading_smooth    = heading_smooth
        self._forward_axis      = np.asarray(forward_axis, dtype=float)
        self._forward_default   = self._forward_axis.copy()
        self._adaptive          = adaptive_thresh
        self._omnidirectional   = bool(omnidirectional)

        self._move_tau   = move_tau
        self._stop_decay = stop_decay
        self._max_speed  = max_speed
        self._vel_xz     = np.zeros(2)
        self._target_xz  = np.zeros(2)
        self._last_cadence = min_step_period_s

        self._enable_vertical = bool(enable_vertical)
        self._jump_smoother   = _EaseSmoother(n_frames=max(1, jump_smooth_frames))
        self._squat_smoother  = _EaseSmoother(n_frames=max(1, squat_smooth_frames))
        self._vclass = _VerticalClassifier(fps=fps, launch_thresh=8.0)
        # Multi-IMU translation gate: confirms locomotion from leg-swing anti-phase.
        self._gait_confirm = GaitConfirmation(fps=fps, no_limb_confidence=1.0)
        self.translation_confirmed: bool = False
        self.translation_mode: str = "test"   # "test" | "user"
        self._on_event = on_event

        self._calib: Optional[CalibrationCapture] = None
        self._calibrating = False

        self._fwd_auto         = bool(auto_forward)
        self._fwd_detected     = False
        self._fwd_cov          = np.zeros((3, 3))
        self._fwd_samples      = 0
        self._fwd_surge_samples: list = []
        self._fwd_min_steps    = int(fwd_calib_steps)
        self._fwd_step_marks   = 0
        self._fwd_travel       = np.zeros(3)

        self._fwd_world_xz   = np.array([0.0, 1.0])
        self._right_world_xz = np.array([1.0, 0.0])

        self._accel_unit_scale = accel_unit_scale
        self._unit_auto        = (accel_unit_scale == 0.0)
        self._gravity = GravityEstimator(n_init=n_gravity_init)

        self._peak_ema     = 0.0
        self._vert_max_run = 0.0
        self._vert_smooth  = 0.0
        self._vert_detect  = 0.0   # v7-fix: lighter EMA for step-peak detection
        self._rising       = False
        self._frames_since = 999
        self._step_count   = 0

        self._heading_xz   = np.array([0.0, 1.0])
        self._heading_init = False

        # ── Directional readout state (v7.1) ──────────────────────────────────
        self.heading_deg:   float = 0.0      # compass heading, 0=forward, +CW
        self.turn_rate_dps: float = 0.0      # signed yaw rate; +CW/right, -CCW/left
        self.turn_dir:      str   = "none"   # "cw" | "ccw" | "none"
        self._prev_heading_deg: Optional[float] = None
        self._turn_deadband_dps: float = 8.0  # below this, treat as not turning
        self._forward_ref_q: Optional[np.ndarray] = None  # explicit forward pose

        self.translation_speed_ms: float = 0.0
        self.gravity_ready:        bool  = False
        self._last_step_dxdz: np.ndarray = np.zeros(2)
        self._jump_count: int  = 0
        self._squat_count: int = 0
        self.gait: str = "idle"
        self.action: str = "none"          # "jumping" | "squatting" | "none"
        self._action_hold: int = 0

    def reset(self) -> None:
        self._gravity.reset()
        self._jump_smoother.reset()
        self._squat_smoother.reset()
        self._vclass.reset()
        self._gait_confirm.reset()
        self.translation_confirmed = False
        self._vel_xz[:]    = 0.0
        self._target_xz[:] = 0.0
        self._vert_smooth  = 0.0
        self._vert_detect  = 0.0
        self._rising       = False
        self._frames_since = 999
        self._step_count   = 0
        self._heading_xz   = np.array([0.0, 1.0])
        self._heading_init = False
        self.heading_deg     = 0.0
        self.turn_rate_dps   = 0.0
        self.turn_dir        = "none"
        self._prev_heading_deg = None
        self._peak_ema     = 0.0
        self._vert_max_run = 0.0
        self._jump_count   = 0
        self._squat_count  = 0
        self.gait          = "idle"
        self.action        = "none"
        self._action_hold  = 0
        if self._fwd_auto:
            self._fwd_detected      = False
            self._fwd_cov[:]        = 0.0
            self._fwd_samples       = 0
            self._fwd_surge_samples = []
            self._fwd_step_marks    = 0
            self._fwd_travel[:]     = 0.0
            self._forward_axis      = self._forward_default.copy()
        self.translation_speed_ms = 0.0
        self.gravity_ready        = False
        self._last_step_dxdz[:]   = 0.0

    # calibration API
    def begin_calibration(self) -> None:
        self._calib = CalibrationCapture(self._fps)
        self._calibrating = True

    def set_calibration_phase(self, phase: str) -> None:
        if self._calib is not None:
            self._calib.set_phase(phase)

    def finish_calibration(self) -> dict:
        if self._calib is None:
            return {}
        summary = self._calib.summarize()
        self._step_accel_thresh      = summary["step_accel_thresh"]
        self._vclass._launch_thresh  = summary["jump_launch_thresh"]
        self._vclass._descend_thresh = summary["squat_descend_thresh"]
        self._calibrating = False
        self._calib = None
        return summary

    # main update
    def update_limb(self, sensor_id: str, gyro: np.ndarray) -> None:
        """
        Feed a NON-pelvis IMU sample to the multi-IMU translation gate.

        Call this from the BLE callback for thigh sensors so the integrator can
        confirm real locomotion (anti-phase leg swing) before translating. Only
        the thigh swing-axis rate is used; other limbs are ignored here but the
        method accepts them harmlessly so the caller can forward everything.

        sensor_id : e.g. "thigh_l", "thigh_r" (matches ble_receiver IDs).
        gyro      : angular velocity vector [gx, gy, gz]; gx is the swing axis.
        """
        g = np.asarray(gyro, dtype=float)
        swing = float(g[0]) if g.shape[0] >= 1 else 0.0
        if sensor_id == "thigh_l":
            self._gait_confirm.push_thigh("l", swing)
        elif sensor_id == "thigh_r":
            self._gait_confirm.push_thigh("r", swing)
        # other sensors: reserved for richer gait models when more IMUs arrive

    def update(self, quaternion: np.ndarray, accel_sensor: np.ndarray) -> tuple:
        accel_in = np.asarray(accel_sensor, dtype=float)
        scale    = self._accel_unit_scale if self._accel_unit_scale != 0.0 else 1.0
        accel_raw = accel_in * scale

        mag   = float(np.linalg.norm(accel_raw))
        g_mag = float(np.linalg.norm(self._gravity.gravity)) or mag
        is_quiet = abs(mag - g_mag) < (0.3 * max(scale, 0.03) if self._unit_auto else 0.3)
        self._gravity.update(accel_raw, is_quiet=is_quiet)
        self.gravity_ready = self._gravity.ready

        self._frames_since += 1

        # Advance the multi-IMU translation gate every frame (tracks staleness).
        gait_conf = self._gait_confirm.tick()
        self.translation_confirmed = self._gait_confirm.has_limbs

        if not self._gravity.ready:
            return self._integrate_velocity()

        if self._unit_auto:
            gm = float(np.linalg.norm(self._gravity.gravity))
            self._accel_unit_scale = 9.81 if gm < 3.0 else 1.0
            self._unit_auto = False
            if self._accel_unit_scale != 1.0:
                self._gravity._gravity *= self._accel_unit_scale
            accel_raw = accel_in * self._accel_unit_scale

        g       = self._gravity.gravity
        g_unit  = g / (np.linalg.norm(g) + 1e-9)
        linear  = self._gravity.linear(accel_raw)
        vert    = float(np.dot(linear, g_unit))
        a = self._accel_smooth_alpha
        self._vert_smooth = a * self._vert_smooth + (1.0 - a) * vert
        # v7-fix: a much lighter EMA preserves the bounce peak for step
        # detection (heavy smoothing was attenuating fast-cadence peaks below
        # threshold, so jog/run produced no steps).
        ad = 0.25
        self._vert_detect = ad * self._vert_detect + (1.0 - ad) * vert
        # Step bounce uses the heavy EMA for stable cadence; the vertical
        # event classifier (jump/squat) uses the LIGHT EMA so launch spikes and
        # free-fall transients are caught promptly instead of being blurred and
        # delayed (which broke the jump-load-phase squat cancellation).
        vert_up        = -self._vert_smooth   # legacy: used elsewhere if needed
        vert_up_detect = -self._vert_detect

        horiz_sensor = linear - vert * g_unit
        surge_world  = _rotate_vec_by_quat(quaternion, horiz_sensor)

        if self._calibrating and self._calib is not None:
            self._calib.feed(vert_up, surge_world)

        if self._fwd_auto and not self._fwd_detected:
            if float(np.linalg.norm(horiz_sensor)) > 0.4:
                self._fwd_cov += np.outer(horiz_sensor, horiz_sensor)
                self._fwd_samples += 1
                self._fwd_surge_samples.append(horiz_sensor.copy())

        self._update_world_basis(quaternion)

        if self._enable_vertical:
            # v7-fix: "walking" must span the user's actual stride period, not
            # the minimum. Based on min_step_frames the flag dropped between
            # slow steps, letting the walk's down-bounce fire false squats.
            cadence_frames = max(self._min_step_frames,
                                 int(self._last_cadence * self._fps))
            walking = self._frames_since < int(1.5 * cadence_frames)
            self._vclass.update(vert_up_detect, self._on_jump, self._on_squat,
                                walking=walking,
                                on_cancel_squat=self._on_cancel_squat)

        thresh = self._step_accel_thresh
        if self._adaptive and self._peak_ema > 0.5:
            thresh = max(1.8, 0.45 * self._peak_ema)

        # v7-fix: detect on the signed UPWARD bounce (vert_up), not abs(). Using
        # abs() fired twice per stride (up-bounce AND down-bounce), halving the
        # apparent cadence and causing a speed sawtooth. Heel-strike is a single
        # upward impulse, so a one-sided test gives one step per stride.
        bounce = max(0.0, -self._vert_detect)
        if bounce > thresh and not self._rising:
            self._rising       = True
            self._vert_max_run = bounce
            if self._frames_since >= self._min_step_frames:
                cadence = max(self._frames_since * self._dt, 1e-3)
                self._last_cadence = cadence
                self._frames_since = 0
                self._step_count  += 1
                step_dir = self._step_direction(quaternion, surge_world)
                speed = float(np.clip(self._stride_length / cadence, 0.0, self._max_speed))
                # SOFT MULTI-IMU GATE: scale translation by how confident we are
                # this is real locomotion (legs swinging anti-phase) vs the
                # pelvis just bobbing in place. With no leg sensors, gait_conf
                # falls back to 1.0 (pelvis-only behaviour) and
                # translation_confirmed stays False.
                speed *= gait_conf
                self._target_xz = step_dir * speed
                self._last_step_dxdz[:] = step_dir * self._stride_length * gait_conf
                if self._fwd_auto and not self._fwd_detected:
                    self._fwd_step_marks += 1
                    self._fwd_travel[0] += self._heading_xz[0]
                    self._fwd_travel[2] += self._heading_xz[1]
                    if (self._fwd_step_marks >= self._fwd_min_steps
                            and self._fwd_samples >= 10):
                        self._lock_forward_axis(quaternion)
        elif self._rising:
            self._vert_max_run = max(self._vert_max_run, bounce)
            if bounce < thresh * 0.5:
                self._peak_ema = (0.8 * self._peak_ema + 0.2 * self._vert_max_run
                                  if self._peak_ema > 0 else self._vert_max_run)
                self._rising = False

        # v7-fix: the "stopped walking" window must scale with the user's
        # ACTUAL cadence, not the minimum step period. A slow walk has a long
        # stride period; using min_step_frames here made the target decay
        # between every slow step, causing the sawtooth ripple. We allow up to
        # 1.6× the last observed cadence (in frames) before declaring a stop.
        cadence_frames = max(self._min_step_frames,
                             int(self._last_cadence * self._fps))
        if self._frames_since > 1.6 * cadence_frames:
            self._target_xz *= self._stop_decay

        return self._integrate_velocity()

    def _integrate_velocity(self) -> tuple:
        alpha = 1.0 - np.exp(-self._dt / max(self._move_tau, 1e-3))
        self._vel_xz += (self._target_xz - self._vel_xz) * alpha
        speed = float(np.linalg.norm(self._vel_xz))
        self.translation_speed_ms = speed
        self.gait = self._gait_label(speed)
        # Decay the live action state; clears to "none" once the vertical arc
        # has played out, so the 3D model knows when the jump/squat is over.
        if self._action_hold > 0:
            self._action_hold -= 1
            if self._action_hold == 0:
                self.action = "none"
        dx = float(self._vel_xz[0]) * self._dt
        dz = float(self._vel_xz[1]) * self._dt
        dy = self._jump_smoother.tick() + self._squat_smoother.tick()
        return dx, dz, dy

    def _gait_label(self, speed: float) -> str:
        for hi, label in self._GAIT_BANDS:
            if speed < hi:
                return label
        return "run"

    def _on_jump(self, ev: dict) -> None:
        self._jump_count += 1
        h = ev["est_height"]
        self._jump_smoother.add(+h)
        self._jump_smoother.add(-h)
        self.action = "jumping"          # live action state for the 3D model
        self._action_hold = int(0.8 * self._fps)
        if self._on_event:
            self._on_event(ev)

    def _on_squat(self, ev: dict) -> None:
        self._squat_count += 1
        d = ev["depth"]
        self._squat_smoother.add(-d)
        self._squat_smoother.add(+d)
        self._last_squat_depth = d
        self.action = "squatting"        # live action state for the 3D model
        self._action_hold = int(1.0 * self._fps)
        if self._on_event:
            self._on_event(ev)

    def _on_cancel_squat(self) -> None:
        # Retract a squat that turned out to be a jump's load phase. The arc is
        # still mostly pending in the smoother, so clearing it removes the dip;
        # the jump arc will be queued instead. Roll back the count + event.
        self._squat_smoother.reset()
        if self._squat_count > 0:
            self._squat_count -= 1
        if self._on_event:
            self._on_event({"type": "squat_cancelled"})

    def _update_world_basis(self, quaternion: np.ndarray) -> None:
        fwd_world = _rotate_vec_by_quat(quaternion, self._forward_axis)
        hxz = np.array([fwd_world[0], fwd_world[2]])
        n = np.linalg.norm(hxz)
        if n > 1e-3:
            hxz = hxz / n
            if not self._heading_init:
                self._heading_xz   = hxz
                self._heading_init = True
            else:
                s = self._heading_smooth
                self._heading_xz = s * self._heading_xz + (1.0 - s) * hxz
                hn = np.linalg.norm(self._heading_xz)
                if hn > 1e-6:
                    self._heading_xz = self._heading_xz / hn
        self._fwd_world_xz   = self._heading_xz.copy()
        self._right_world_xz = np.array([self._heading_xz[1], -self._heading_xz[0]])

        # ── Live directional readout (heading + signed turn rate) ─────────────
        # Compass heading: 0° = +Z (forward), +90° = +X (right), measured CW.
        # turn_rate_dps: signed yaw velocity. +ve = clockwise / turning right,
        # -ve = anticlockwise / turning left. Derived from successive headings,
        # wrap-safe via shortest signed delta.
        heading_deg = float(np.degrees(np.arctan2(
            self._heading_xz[0], self._heading_xz[1])))
        if self._prev_heading_deg is not None:
            d = heading_deg - self._prev_heading_deg
            d = (d + 180.0) % 360.0 - 180.0          # shortest signed delta
            inst_rate = d * self._fps                # deg/s
            # light EMA so the readout isn't jittery
            self.turn_rate_dps = 0.6 * self.turn_rate_dps + 0.4 * inst_rate
        self._prev_heading_deg = heading_deg
        self.heading_deg = heading_deg
        if   self.turn_rate_dps >  self._turn_deadband_dps: self.turn_dir = "cw"
        elif self.turn_rate_dps < -self._turn_deadband_dps: self.turn_dir = "ccw"
        else:                                               self.turn_dir = "none"

    def _step_direction(self, quaternion: np.ndarray, surge_world: np.ndarray) -> np.ndarray:
        if not self._omnidirectional:
            return self._heading_xz.copy()
        sxz = np.array([surge_world[0], surge_world[2]])
        smag = float(np.linalg.norm(sxz))
        if smag < 0.3:
            return self._heading_xz.copy()
        sxz = sxz / smag
        fdot = float(np.dot(sxz, self._fwd_world_xz))
        rdot = float(np.dot(sxz, self._right_world_xz))
        step_dir = fdot * self._fwd_world_xz + rdot * self._right_world_xz
        dn = float(np.linalg.norm(step_dir))
        if dn < 1e-6:
            return self._fwd_world_xz.copy()
        return step_dir / dn

    def _lock_forward_axis(self, quaternion: np.ndarray) -> None:
        g      = self._gravity.gravity
        g_unit = g / (np.linalg.norm(g) + 1e-9)
        P      = np.eye(3) - np.outer(g_unit, g_unit)
        cov_h  = P @ self._fwd_cov @ P
        try:
            w, v = np.linalg.eigh(cov_h)
        except np.linalg.LinAlgError:
            return
        axis = v[:, int(np.argmax(w))]
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        sign_resolved = False
        travel_world  = self._fwd_travel.copy()
        travel_mag    = float(np.linalg.norm(travel_world[[0, 2]]))
        if travel_mag > 0.5 * self._fwd_step_marks:
            q_conj        = np.array([quaternion[0], -quaternion[1],
                                      -quaternion[2], -quaternion[3]])
            travel_sensor = _rotate_vec_by_quat(q_conj, travel_world)
            if np.dot(axis, travel_sensor) < 0:
                axis = -axis
            sign_resolved = True
        if not sign_resolved and self._fwd_surge_samples:
            proj = np.array([float(np.dot(s, axis)) for s in self._fwd_surge_samples])
            m, sd = proj.mean(), proj.std()
            if sd > 1e-6:
                skew = float(np.mean(((proj - m) / sd) ** 3))
                if skew < 0:
                    axis = -axis
        self._forward_axis = axis
        self._fwd_detected = True

    def recalibrate_forward(self) -> None:
        self._fwd_auto          = True
        self._fwd_detected      = False
        self._fwd_cov[:]        = 0.0
        self._fwd_samples       = 0
        self._fwd_surge_samples = []
        self._fwd_step_marks    = 0
        self._fwd_travel[:]     = 0.0

    def flip_forward(self) -> None:
        self._forward_axis = -self._forward_axis

    def set_translation_mode(self, mode: str) -> None:
        """
        'test' — pelvis-only bounce translates (no limb confirmation needed).
                 Use while testing with only the pelvis connected.
        'user' — translation requires limb confirmation; with no leg sensors
                 the gate yields 0 and the avatar will not drift on bounce.
        """
        if mode == "test":
            self._gait_confirm._no_limb_conf = 1.0
        elif mode == "user":
            self._gait_confirm._no_limb_conf = 0.0
        else:
            raise ValueError("mode must be 'test' or 'user'")
        self.translation_mode = mode

    # ── Explicit directional initialisation (v7.1) ────────────────────────────
    #
    # Instead of inferring the forward axis from walking (auto-detection, which
    # can guess the sign wrong and need flip_forward), the user can EXPLICITLY
    # teach the body frame by holding two poses:
    #
    #   1. "Face forward and hold"  -> set_forward_reference(quaternion)
    #   2. "Turn 90° right and hold" -> set_right_reference(quaternion)  [optional]
    #
    # From these, forward / back / left / right and clockwise / anticlockwise
    # all derive deterministically. The sensor axis that maps to the body's
    # forward heading is captured directly, so no walking is required and the
    # sign is never ambiguous.

    def set_forward_reference(self, quaternion: np.ndarray) -> dict:
        """
        Capture the current held orientation as 'facing forward'.

        We want a constant SENSOR-frame vector that, when rotated by the live
        quaternion, points along the body's forward heading in the world's
        horizontal plane. The reference is: take world +Z (the canonical
        forward) and express it in the sensor frame at this instant, i.e.
        forward_axis_sensor = R(q)^-1 · world_forward.  Rotating that by any
        later q gives the world forward heading for the new orientation.
        """
        q = normalize_quaternion(np.asarray(quaternion, dtype=float))
        world_forward = np.array([0.0, 0.0, 1.0])           # canonical +Z
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
        fwd_sensor = _rotate_vec_by_quat(q_conj, world_forward)
        # Project out the gravity component so forward is purely horizontal.
        if self._gravity.ready:
            g = self._gravity.gravity
            g_unit = g / (np.linalg.norm(g) + 1e-9)
            fwd_sensor = fwd_sensor - np.dot(fwd_sensor, g_unit) * g_unit
        n = np.linalg.norm(fwd_sensor)
        if n < 1e-6:
            return {"status": "error", "message": "Degenerate forward pose."}
        self._forward_axis  = fwd_sensor / n
        self._forward_ref_q = q.copy()
        self._fwd_detected  = True          # bypass auto-detection
        self._fwd_auto      = False
        self._heading_init  = False         # re-seed heading on next frame
        self._prev_heading_deg = None
        return {
            "status":       "ok",
            "forward_axis": self._forward_axis.tolist(),
            "message":      "Forward reference captured.",
        }

    def set_right_reference(self, quaternion: np.ndarray) -> dict:
        """
        Capture a 'turned ~90° to the right' pose to confirm the yaw sign.

        With forward already set, turning right should increase the compass
        heading (clockwise / +). If the captured right-pose heading is LESS
        than the forward heading (i.e. the turn read as anticlockwise), the
        forward axis sign is inverted and we flip it so right is unambiguously
        positive. This removes the only remaining sign ambiguity.
        """
        if self._forward_ref_q is None:
            return {"status": "error", "message": "Set forward reference first."}

        def _heading(qq):
            fw = _rotate_vec_by_quat(qq, self._forward_axis)
            h = np.array([fw[0], fw[2]])
            nn = np.linalg.norm(h)
            if nn < 1e-9:
                return None
            h /= nn
            return float(np.degrees(np.arctan2(h[0], h[1])))

        q = normalize_quaternion(np.asarray(quaternion, dtype=float))
        h_fwd   = _heading(self._forward_ref_q)
        h_right = _heading(q)
        if h_fwd is None or h_right is None:
            return {"status": "error", "message": "Degenerate pose."}
        d = (h_right - h_fwd + 180.0) % 360.0 - 180.0   # signed delta
        flipped = False
        if d < 0:                       # right turn read as CCW → axis sign wrong
            self._forward_axis = -self._forward_axis
            flipped = True
        return {
            "status":        "ok",
            "turn_delta_deg": round(d, 1),
            "axis_flipped":   flipped,
            "forward_axis":   self._forward_axis.tolist(),
            "message":        ("Right turn confirmed; axis sign corrected."
                               if flipped else "Right turn confirmed."),
        }

    def get_direction_state(self) -> dict:
        """
        Live directional readout for the 3D model / HUD.

        heading_deg   : compass heading, 0°=forward, +90°=right, measured CW.
        turn_rate_dps : signed yaw rate (+CW/right, -CCW/left).
        turn_dir      : "cw" | "ccw" | "none".
        heading_xz    : unit forward vector in world XZ.
        right_xz      : unit right (strafe) vector in world XZ.
        move_dir      : coarse label of current locomotion direction relative
                        to facing — forward/backward/strafe_left/strafe_right —
                        derived from the last step direction vs the body frame.
        """
        move_dir = "idle"
        if self.translation_speed_ms > 0.05 and float(np.linalg.norm(self._last_step_dxdz)) > 1e-6:
            sd = self._last_step_dxdz / (np.linalg.norm(self._last_step_dxdz) + 1e-9)
            fdot = float(np.dot(sd, self._fwd_world_xz))
            rdot = float(np.dot(sd, self._right_world_xz))
            if abs(fdot) >= abs(rdot):
                move_dir = "forward" if fdot >= 0 else "backward"
            else:
                move_dir = "strafe_right" if rdot >= 0 else "strafe_left"
        return {
            "heading_deg":   round(self.heading_deg, 1),
            "turn_rate_dps": round(self.turn_rate_dps, 1),
            "turn_dir":      self.turn_dir,
            "heading_xz":    self._fwd_world_xz.tolist(),
            "right_xz":      self._right_world_xz.tolist(),
            "move_dir":      move_dir,
            "gait":          self.gait,
            "action":        self.action,
            "translation_confidence": round(self._gait_confirm.confidence, 3),
            "translation_confirmed":  self.translation_confirmed,
            "speed_ms":      round(self.translation_speed_ms, 3),
        }

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self._vel_xz[0], 0.0, self._vel_xz[1]])

    @property
    def _scale(self) -> float:
        return self._stride_length / 0.65
    @_scale.setter
    def _scale(self, v: float) -> None:
        self._stride_length = 0.65 * float(v)

    @property
    def _accel_threshold(self) -> float:
        return self._step_accel_thresh / 10.0
    @_accel_threshold.setter
    def _accel_threshold(self, v: float) -> None:
        self._step_accel_thresh = max(1.8, float(v) * 10.0)

    @property
    def _velocity_decay(self) -> float:
        return self._heading_smooth
    @_velocity_decay.setter
    def _velocity_decay(self, v: float) -> None:
        self._heading_smooth = float(np.clip(v, 0.05, 0.95))

    class _HPShim:
        def __init__(self, outer): self._outer = outer
        @property
        def _alpha(self): return self._outer._accel_smooth_alpha
        @_alpha.setter
        def _alpha(self, v): self._outer._accel_smooth_alpha = float(v)

    @property
    def _hp(self):
        return RootIntegrator._HPShim(self)

    def get_debug_info(self) -> dict:
        return {
            "model":                "velocity_damped_v7",
            "gait":                 self.gait,
            "gravity_ready":        self.gravity_ready,
            "gravity_estimate":     self._gravity.gravity.tolist(),
            "step_count":           self._step_count,
            "jump_count":           self._jump_count,
            "squat_count":          self._squat_count,
            "action":               self.action,
            "translation_confidence": round(self._gait_confirm.confidence, 3),
            "translation_confirmed":  self.translation_confirmed,
            "has_limb_sensors":       self._gait_confirm.has_limbs,
            "heading_xz":           self._heading_xz.tolist(),
            "forward_xz":           self._fwd_world_xz.tolist(),
            "right_xz":             self._right_world_xz.tolist(),
            "vel_xz":               self._vel_xz.tolist(),
            "target_xz":            self._target_xz.tolist(),
            "translation_speed_ms": round(self.translation_speed_ms, 4),
            "move_tau":             self._move_tau,
            "last_cadence":         round(self._last_cadence, 3),
            "omnidirectional":      self._omnidirectional,
            "vertical_enabled":     self._enable_vertical,
            "jump_active":          self._jump_smoother.active,
            "squat_active":         self._squat_smoother.active,
            "stride_length":        self._stride_length,
            "step_accel_thresh":    self._step_accel_thresh,
            "jump_launch_thresh":   self._vclass._launch_thresh,
            "squat_descend_thresh": self._vclass._descend_thresh,
            "forward_axis":         self._forward_axis.tolist(),
            "forward_detected":     self._fwd_detected,
            "calibrating":          self._calibrating,
        }