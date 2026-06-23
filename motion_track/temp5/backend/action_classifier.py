"""
backend/action_classifier.py
----------------------------
Multi-IMU biomechanical action-recognition pipeline.

Implements the design spec: classify lower-body locomotion / action from a
fused set of IMUs, with the pelvis as the global root reference.

States:  IDLE, WALK_FORWARD, RUN_FORWARD, JUMP, SQUAT are ACTIVE and validated
         (against simulation). STRAFE_LEFT/RIGHT, WALK_BACKWARD, and the hybrid
         states (CROUCH_WALK, RUN_STRAFE_*, WALK_DIAGONAL_*) are STUBBED by
         default: their scoring logic exists and runs for diagnostics, but they
         are excluded from the committed vote until `enable_experimental=True`.
         Enable them only after the leg IMUs are connected and the relevant
         thresholds are tuned on real data — otherwise they would emit
         unvalidated labels. Set the flag via the constructor or
         enable_experimental_states().

Output contract (see ActionState):
  action          str   — current label
  confidence      float — 0..100
  phase           str   — action sub-phase if applicable
  cadence_hz      float
  symmetry        float — 0..100 (L/R balance)
  stability       float — 0..100 (lower torso/pelvis sway = higher)
  direction_deg   float — 0=fwd, 180=back, -90=left, +90=right
  fatigue         float — 0..100 (movement-quality degradation)

═══════════════════════════════════════════════════════════════════════════
IMPORTANT — SIMULATION-TUNED THRESHOLDS
═══════════════════════════════════════════════════════════════════════════
This module's numeric thresholds (the THRESH dataclass below) were tuned
against SYNTHETIC signals only, because at authoring time only the pelvis IMU
was connected. Every threshold is a starting point, NOT a validated value.
Re-tune all of them against real multi-IMU recordings before relying on the
output for anything beyond the pelvis-only fallback. Each threshold is grouped
and commented so a tuning pass is mechanical.

SCOPE NOTES
-----------
* Fusion: this pipeline consumes the per-sensor quaternions already produced by
  firmware (Madgwick on-device). It does NOT re-run Madgwick/Kalman here.
* Cadence: uses peak-interval timing, not FFT. FFT would slot into
  _Periodicity if you later want frequency-domain confirmation.
* Edge inference: this runs in the Python backend, not on the ESP32.
* Foot-contact: estimated from foot-IMU vertical accel + low angular velocity
  ("quiet" foot ≈ stance). Real plantar pressure would be more reliable.

DATA FEED
---------
Call update_sensor(sensor_id, quaternion, accel, gyro, t) for every packet,
then call tick() once per pelvis frame to get an ActionState. Missing sensors
degrade gracefully — the classifier uses whatever subset is present and lowers
confidence accordingly.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# QUATERNION HELPERS (shared convention with skeleton.py / root_integrator.py)
# ─────────────────────────────────────────────────────────────────────────────

def _qmul(a, b):
    return np.array([
        a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
        a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
        a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
        a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0],
    ])

def _qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def _qrot(q, v):
    qv = np.array([0.0, v[0], v[1], v[2]])
    return _qmul(_qmul(q, qv), _qconj(q))[1:]

def _qnorm(q):
    n = float(np.linalg.norm(q))
    return q / n if n > 1e-9 else np.array([1.0, 0, 0, 0])


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS  (ALL SIMULATION-TUNED — RETUNE ON REAL DATA)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class THRESH:
    # ── cadence (Hz) ──────────────────────────────────────────────────────────
    walk_cadence_min: float = 0.7
    walk_cadence_max: float = 2.6
    run_cadence_min:  float = 2.4

    # ── angular velocity (rad/s on the swing axis, gyro X of thigh/shin) ──────
    shin_swing_walk:  float = 1.2     # min shin swing energy to be "moving"
    shin_swing_run:   float = 3.0     # bilateral swing peak → running
    thigh_swing_walk: float = 0.8

    # ── pelvis linear accel (m/s², gravity removed) ───────────────────────────
    pelvis_vacc_walk_max: float = 6.0    # walking keeps vertical bob modest
    pelvis_vacc_run_min:  float = 7.0    # running raises vertical accel
    jump_launch_vacc:     float = 8.0    # explosive upward spike
    lateral_acc_strafe:   float = 2.5    # lateral dominates → strafe

    # ── foot contact / flight ─────────────────────────────────────────────────
    foot_quiet_gyro:   float = 0.6    # below → foot likely in stance
    foot_impact_acc:   float = 12.0   # landing spike
    flight_time_min_s: float = 0.06   # both feet airborne → run/jump
    jump_flight_min_s: float = 0.12   # longer synchronized flight → jump

    # ── squat ─────────────────────────────────────────────────────────────────
    knee_flex_squat_deg: float = 35.0   # min bilateral knee flexion for squat
    squat_descend_vacc:  float = 1.8    # sustained downward pelvis accel

    # ── synchronization / symmetry ────────────────────────────────────────────
    antiphase_gait:    float = -0.3   # L/R corr below this ⇒ alternating gait
    sync_jump:         float =  0.5   # L/R corr above this ⇒ synchronized (jump)
    direction_fwd_cone: float = 45.0  # ± deg around forward = "forward"

    # ── persistence / hysteresis (frames @ fps) ───────────────────────────────
    persist_s:    float = 0.4    # an action must hold this long to commit
    hysteresis:   float = 12.0   # confidence margin to switch states


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT STRUCT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ActionState:
    action:        str   = "IDLE"
    confidence:    float = 0.0
    phase:         str   = ""
    cadence_hz:    float = 0.0
    symmetry:      float = 100.0
    stability:     float = 100.0
    direction_deg: float = 0.0
    fatigue:       float = 0.0
    # diagnostics (handy for the HUD / tuning)
    raw_scores:    dict  = field(default_factory=dict)
    active_sensors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# PER-SENSOR ROLLING STATE
# ─────────────────────────────────────────────────────────────────────────────

class _SensorTrack:
    """Rolling buffers + derived signals for one IMU."""
    def __init__(self, fps: float, win_s: float = 1.2):
        self.fps = fps
        self.n   = max(4, int(win_s * fps))
        self.quat:  np.ndarray = np.array([1.0, 0, 0, 0])
        self.accel: np.ndarray = np.zeros(3)
        self.gyro:  np.ndarray = np.zeros(3)
        self.last_t: float = 0.0
        self.swing = deque(maxlen=self.n)   # swing-axis angular rate (gyro X)
        self.vacc  = deque(maxlen=self.n)   # |accel| - g  (linear magnitude)
        self.fresh = False

    def push(self, quat, accel, gyro, t):
        self.quat  = _qnorm(np.asarray(quat, float))
        self.accel = np.asarray(accel, float)
        self.gyro  = np.asarray(gyro, float)
        self.last_t = t
        self.swing.append(float(self.gyro[0]) if self.gyro.shape[0] else 0.0)
        self.vacc.append(float(np.linalg.norm(self.accel)) - 9.81)
        self.fresh = True

    def stale(self, now, timeout=0.4):
        return (now - self.last_t) > timeout

    def swing_energy(self):
        return float(np.std(self.swing)) if len(self.swing) > 3 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# PERIODICITY / CADENCE  (peak-interval; FFT would slot in here)
# ─────────────────────────────────────────────────────────────────────────────

class _Periodicity:
    def __init__(self, fps):
        self.fps = fps
        self._last_peak_frame = None
        self._intervals = deque(maxlen=6)
        self._frame = 0
        self._prev = 0.0
        self._rising = False
        self._refractory = 0
        self._min_gap = max(2, int(0.20 * fps))   # ≤ ~2.5 Hz cap on peak rate

    def update(self, signal_val, thresh):
        """
        One peak per full cycle. We arm on a rising crossing above `thresh`,
        then register a single peak when the signal turns over (starts falling).
        A refractory window prevents the second half-cycle / noise from
        double-counting (which doubled the apparent cadence).
        """
        self._frame += 1
        if self._refractory > 0:
            self._refractory -= 1

        crossed_up = (signal_val > thresh) and (self._prev <= thresh)
        if crossed_up and self._refractory == 0:
            if self._last_peak_frame is not None:
                self._intervals.append(self._frame - self._last_peak_frame)
            self._last_peak_frame = self._frame
            self._refractory = self._min_gap
        self._prev = signal_val

        if len(self._intervals) >= 2:
            mean_int = float(np.mean(self._intervals))
            if mean_int > 1e-6:
                return self.fps / mean_int
        return 0.0

    def reset(self):
        self._last_peak_frame = None
        self._intervals.clear()
        self._prev = 0.0
        self._rising = False
        self._refractory = 0


# ─────────────────────────────────────────────────────────────────────────────
# ACTION CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

# Priority sensor groups (per spec section 14)
PRIMARY_SENSORS   = ["pelvis", "shin_l", "shin_r", "thigh_l", "thigh_r", "l_foot", "r_foot"]
SECONDARY_SENSORS = ["chest", "head"]
ARM_SENSORS       = ["l_upper_arm", "r_upper_arm", "l_forearm", "r_forearm",
                     "l_shoulder", "r_shoulder"]


class ActionClassifier:
    """
    Feed update_sensor() per packet; call tick() once per pelvis frame.

    mode:
      "user" — full biomechanical classification (needs legs for best results)
      "test" — pelvis-only: report WALK_FORWARD on pelvis bounce so the avatar
               moves during single-sensor testing (mirrors the integrator's
               test mode). No limb confirmation required.
    """

    def __init__(self, fps: float = 50.0, mode: str = "user",
                 thresh: Optional[THRESH] = None,
                 enable_experimental: bool = False):
        self.fps    = fps
        self.dt     = 1.0 / fps
        self.mode   = mode
        self.T      = thresh or THRESH()
        # When False (default), strafe / backward / hybrid states are STUBBED:
        # their scorers still exist and run for diagnostics, but they do NOT
        # compete in the committed vote and never become the output label.
        # Flip to True only after legs are connected and these are tuned on
        # real multi-IMU data. Validated-now states (walk/run/jump/squat/idle)
        # are unaffected by this flag.
        self.enable_experimental = bool(enable_experimental)
        self._EXPERIMENTAL_STATES = {
            "STRAFE_LEFT", "STRAFE_RIGHT", "WALK_BACKWARD",
            "WALK_DIAGONAL_LEFT", "WALK_DIAGONAL_RIGHT",
            "RUN_STRAFE_LEFT", "RUN_STRAFE_RIGHT", "CROUCH_WALK",
        }
        self._tracks: dict[str, _SensorTrack] = {}
        self._cad   = _Periodicity(fps)

        # committed-state machine
        self.state  = ActionState()
        self._candidate = "IDLE"
        self._cand_frames = 0
        self._persist_frames = max(1, int(self.T.persist_s * fps))

        # gravity for pelvis vertical decomposition
        self._g_pelvis = np.zeros(3)
        self._g_n = 0

        # fatigue tracking: degradation of symmetry/stability over time
        self._sym_hist = deque(maxlen=int(8 * fps))
        self._stab_hist = deque(maxlen=int(8 * fps))

        # flight-phase tracking
        self._airborne_frames = 0
        self._last_flight_s = 0.0

    # ── feed ──────────────────────────────────────────────────────────────────

    def update_sensor(self, sensor_id, quaternion, accel, gyro, t):
        if sensor_id not in self._tracks:
            self._tracks[sensor_id] = _SensorTrack(self.fps)
        self._tracks[sensor_id].push(quaternion, accel, gyro, t)
        if sensor_id == "pelvis":
            a = np.asarray(accel, float)
            self._g_n += 1
            lr = 0.001 if self._g_n > 100 else 1.0 / self._g_n
            self._g_pelvis = (1 - lr) * self._g_pelvis + lr * a

    def enable_experimental_states(self, enabled: bool = True) -> None:
        """
        Turn the stubbed states (strafe / backward / hybrids) on or off at
        runtime. Keep OFF until legs are connected and tuned on real data.
        """
        self.enable_experimental = bool(enabled)

    def set_mode(self, mode: str):
        if mode not in ("user", "test"):
            raise ValueError("mode must be 'user' or 'test'")
        self.mode = mode

    def reset(self):
        self._tracks.clear()
        self._cad.reset()
        self.state = ActionState()
        self._candidate = "IDLE"
        self._cand_frames = 0
        self._g_pelvis = np.zeros(3)
        self._g_n = 0
        self._sym_hist.clear()
        self._stab_hist.clear()
        self._airborne_frames = 0

    # ── helpers ─────────────────────────────────────────────────────────────

    def _active(self, now):
        return [sid for sid, tr in self._tracks.items() if not tr.stale(now)]

    def _have(self, *ids):
        return all(i in self._tracks and self._tracks[i].fresh for i in ids)

    def _pelvis_linear(self):
        """Pelvis linear accel with gravity removed, split into vert/lateral/fwd."""
        if "pelvis" not in self._tracks:
            return 0.0, 0.0, 0.0
        tr = self._tracks["pelvis"]
        g = self._g_pelvis
        gn = np.linalg.norm(g)
        if gn < 1e-6:
            return 0.0, 0.0, 0.0
        g_unit = g / gn
        lin = tr.accel - g
        vert = float(np.dot(lin, g_unit))                  # +down/-up depending on frame
        horiz = lin - vert * g_unit
        # express horizontal in world via pelvis quaternion → fwd / lateral
        horiz_world = _qrot(tr.quat, horiz)
        fwd = float(horiz_world[2]); lat = float(horiz_world[0])
        return vert, lat, fwd

    def _bilateral_corr(self, a_id, b_id):
        """L/R swing correlation: -1 anti-phase (gait), +1 in-phase (jump)."""
        if not self._have(a_id, b_id):
            return 0.0, 0.0
        A = np.array(self._tracks[a_id].swing)
        B = np.array(self._tracks[b_id].swing)
        m = min(len(A), len(B))
        if m < 6 or np.std(A[-m:]) < 1e-3 or np.std(B[-m:]) < 1e-3:
            return 0.0, 0.0
        corr = float(np.corrcoef(A[-m:], B[-m:])[0, 1])
        energy = float((np.std(A[-m:]) + np.std(B[-m:])) / 2.0)
        return corr, energy

    def _knee_flexion(self, side):
        """Approx knee flexion from thigh-vs-shin quaternion angular difference."""
        th, sh = f"thigh_{side}", f"shin_{side}"
        if not self._have(th, sh):
            return 0.0
        q_rel = _qmul(_qconj(self._tracks[th].quat), self._tracks[sh].quat)
        w = float(np.clip(abs(q_rel[0]), -1, 1))
        return float(np.degrees(2 * np.arccos(w)))

    def _foot_contact(self, side):
        """True if foot looks like it's in stance (quiet gyro)."""
        fid = f"{side}_foot"
        if fid not in self._tracks or not self._tracks[fid].fresh:
            return None  # unknown
        return float(np.linalg.norm(self._tracks[fid].gyro)) < self.T.foot_quiet_gyro

    # ── main tick ─────────────────────────────────────────────────────────────

    def tick(self, now: Optional[float] = None) -> ActionState:
        if now is None:
            now = self._tracks["pelvis"].last_t if "pelvis" in self._tracks else 0.0
        active = self._active(now)

        # ── TEST MODE: pelvis-only bounce → WALK_FORWARD ──────────────────────
        if self.mode == "test":
            return self._tick_test_mode(active)

        vert, lat, fwd = self._pelvis_linear()
        scores: dict[str, float] = {}

        # cadence from signed shin swing (one rising crossing per stride);
        # feeding abs() would double the count by catching both half-cycles.
        if self._have("shin_l"):
            cad_sig = self._tracks["shin_l"].swing[-1] if self._tracks["shin_l"].swing else 0.0
            cadence = self._cad.update(cad_sig, self.T.shin_swing_walk)
        else:
            cadence = self._cad.update(-vert, self.T.pelvis_vacc_walk_max * 0.4)

        # bilateral coordination
        corr_thigh, eng_thigh = self._bilateral_corr("thigh_l", "thigh_r")
        corr_shin,  eng_shin  = self._bilateral_corr("shin_l", "shin_r")

        # foot contacts / flight
        cl, cr = self._foot_contact("l"), self._foot_contact("r")
        both_air = (cl is False and cr is False)
        if both_air:
            self._airborne_frames += 1
        else:
            if self._airborne_frames > 0:
                self._last_flight_s = self._airborne_frames * self.dt
            self._airborne_frames = 0
        flight_s = self._airborne_frames * self.dt

        # knee flexion (squat)
        kf_l, kf_r = self._knee_flexion("l"), self._knee_flexion("r")
        knee_flex = (kf_l + kf_r) / 2

        # ── SCORE each candidate state (0..100) ───────────────────────────────
        # SQUAT: bilateral knee flexion, feet planted, pelvis lowering not translating
        feet_planted = (cl is not False and cr is not False)
        scores["SQUAT"] = self._score_squat(knee_flex, vert, feet_planted)

        # JUMP: synchronized bilateral takeoff + long flight + vertical dominance
        scores["JUMP"] = self._score_jump(corr_thigh, vert, flight_s, self._last_flight_s)

        # RUN: high cadence + strong bilateral anti-phase swing + flight phases
        scores["RUN_FORWARD"] = self._score_run(cadence, corr_shin, eng_shin, vert, flight_s)

        # WALK family: alternating swing, modest vert, persistent cadence.
        # Score "walking" as ONE state so direction ambiguity doesn't split the
        # vote (forward/back/strafe is resolved after commit, from direction_deg).
        walk_base = self._score_walk(cadence, corr_shin, eng_shin, vert)
        direction_deg = self._movement_direction(fwd, lat)
        scores["WALK"] = walk_base

        # STRAFE scored separately (lateral-dominant gait is mechanically distinct)
        scores["STRAFE_LEFT"]  = self._score_strafe(lat, fwd, eng_shin, side="left")
        scores["STRAFE_RIGHT"] = self._score_strafe(lat, fwd, eng_shin, side="right")

        # IDLE: everything quiet
        scores["IDLE"] = self._score_idle(active, vert, eng_shin, eng_thigh)

        # ── pick winner, apply persistence + hysteresis ───────────────────────
        # Pick the winner. When experimental states are disabled (default),
        # they are removed from the vote entirely — their scores remain in
        # raw_scores for diagnostics/tuning but cannot become the label.
        if self.enable_experimental:
            votable = scores
        else:
            votable = {k: v for k, v in scores.items()
                       if k not in self._EXPERIMENTAL_STATES}
        winner = max(votable, key=votable.get)
        win_score = votable[winner]

        committed = self._commit(winner, win_score, votable)
        self._committed_base = committed

        # symmetry / stability / fatigue
        symmetry  = self._symmetry(kf_l, kf_r, eng_shin)
        stability = self._stability(vert, lat)
        self._sym_hist.append(symmetry); self._stab_hist.append(stability)
        fatigue   = self._fatigue()

        # hybrid refinement (crouch-walk, run-strafe, diagonal)
        committed, phase = self._hybrid_and_phase(committed, knee_flex, direction_deg,
                                                  vert, flight_s)

        self.state = ActionState(
            action=committed,
            confidence=round(float(np.clip(win_score, 0, 100)), 1),
            phase=phase,
            cadence_hz=round(cadence, 2),
            symmetry=round(symmetry, 1),
            stability=round(stability, 1),
            direction_deg=round(direction_deg, 1),
            fatigue=round(fatigue, 1),
            raw_scores={k: round(v, 1) for k, v in scores.items()},
            active_sensors=active,
        )
        return self.state

    # ── test-mode tick ─────────────────────────────────────────────────────────

    def _tick_test_mode(self, active) -> ActionState:
        # Use recent vertical-accel ENERGY (std over the buffer), not the
        # instantaneous value — the instantaneous bob crosses zero twice per
        # cycle and would flicker to IDLE at the crossings.
        vacc_energy = 0.0
        if "pelvis" in self._tracks and len(self._tracks["pelvis"].vacc) > 4:
            vacc_energy = float(np.std(self._tracks["pelvis"].vacc))
        vert, lat, fwd = self._pelvis_linear()
        bouncing = vacc_energy > self.T.pelvis_vacc_walk_max * 0.25
        cadence = self._cad.update(-vert, 0.0)
        action = "WALK_FORWARD" if bouncing else "IDLE"
        conf = 60.0 if bouncing else 100.0   # test mode: low conf, flagged
        self.state = ActionState(
            action=action, confidence=conf, phase="test_mode",
            cadence_hz=round(cadence, 2), symmetry=100.0, stability=100.0,
            direction_deg=0.0, fatigue=0.0,
            raw_scores={"test_mode": 1.0, "vacc_energy": round(vacc_energy, 2)},
            active_sensors=active,
        )
        return self.state

    # ── per-state scorers (0..100) ──────────────────────────────────────────────

    def _score_walk(self, cadence, corr_shin, eng_shin, vert):
        if not (self.T.walk_cadence_min <= cadence <= self.T.walk_cadence_max):
            cad_s = 0.0
        else:
            cad_s = 1.0
        antiphase = np.clip(-corr_shin, 0, 1)          # alternating legs
        energy    = np.clip(eng_shin / self.T.shin_swing_run, 0, 1)
        vmod      = np.clip(1 - (abs(vert) - self.T.pelvis_vacc_walk_max)
                            / self.T.pelvis_vacc_walk_max, 0, 1)
        return 100.0 * cad_s * (0.5 * antiphase + 0.3 * energy + 0.2 * vmod)

    def _score_run(self, cadence, corr_shin, eng_shin, vert, flight_s):
        cad_s   = np.clip((cadence - self.T.run_cadence_min) / 1.0, 0, 1)
        swing_s = np.clip(eng_shin / self.T.shin_swing_run, 0, 1)
        vacc_s  = np.clip((abs(vert) - self.T.pelvis_vacc_run_min)
                          / self.T.pelvis_vacc_run_min, 0, 1)
        flight_s_score = np.clip(flight_s / self.T.flight_time_min_s, 0, 1)
        antiphase = np.clip(-corr_shin, 0, 1)
        return 100.0 * (0.3 * cad_s + 0.3 * swing_s + 0.2 * vacc_s
                        + 0.1 * flight_s_score + 0.1 * antiphase)

    def _score_jump(self, corr_thigh, vert, flight_s, last_flight_s):
        # vert sign: upward spike. Our `vert` is along gravity; treat magnitude.
        launch = np.clip((abs(vert) - self.T.jump_launch_vacc)
                         / self.T.jump_launch_vacc, 0, 1)
        sync   = np.clip(corr_thigh, 0, 1)              # synchronized (in-phase) legs
        flight = np.clip(max(flight_s, last_flight_s) / self.T.jump_flight_min_s, 0, 1)
        return 100.0 * (0.45 * launch + 0.25 * sync + 0.30 * flight)

    def _score_squat(self, knee_flex, vert, feet_planted):
        if knee_flex < self.T.knee_flex_squat_deg:
            return 0.0
        flex_s = np.clip(knee_flex / 90.0, 0, 1)
        planted_s = 1.0 if feet_planted else 0.3
        return 100.0 * (0.7 * flex_s + 0.3 * planted_s)

    def _score_strafe(self, lat, fwd, eng_shin, side):
        lat_dom = abs(lat) > abs(fwd) and abs(lat) > self.T.lateral_acc_strafe
        if not lat_dom or eng_shin < self.T.shin_swing_walk * 0.5:
            return 0.0
        correct_side = (lat < 0) if side == "left" else (lat > 0)
        if not correct_side:
            return 0.0
        return 100.0 * np.clip(abs(lat) / (self.T.lateral_acc_strafe * 2), 0, 1)

    def _score_idle(self, active, vert, eng_shin, eng_thigh):
        moving = abs(vert) > 1.0 or eng_shin > self.T.shin_swing_walk * 0.4 \
                 or eng_thigh > self.T.thigh_swing_walk * 0.4
        return 0.0 if moving else 100.0

    # ── direction / symmetry / stability / fatigue ──────────────────────────────

    def _movement_direction(self, fwd, lat):
        """
        0=fwd, ±90=strafe, 180=back. Uses a leaky accumulation of the horizontal
        movement vector so direction reflects sustained travel, not the
        per-frame accel sign (which oscillates within each stride).
        """
        if not hasattr(self, "_dir_accum"):
            self._dir_accum = np.zeros(2)
        self._dir_accum = 0.92 * self._dir_accum + 0.08 * np.array([lat, fwd])
        a = self._dir_accum
        if np.linalg.norm(a) < 1e-3:
            return self.state.direction_deg
        return float(np.degrees(np.arctan2(a[0], a[1])))

    def _symmetry(self, kf_l, kf_r, eng_shin):
        if kf_l + kf_r < 1e-3:
            return 100.0
        diff = abs(kf_l - kf_r) / (abs(kf_l + kf_r) / 2 + 1e-6)
        return float(np.clip(100 * (1 - diff), 0, 100))

    def _stability(self, vert, lat):
        sway = abs(vert) + abs(lat)
        return float(np.clip(100 - sway * 6, 0, 100))

    def _fatigue(self):
        """Degradation: symmetry/stability trending down over the window."""
        if len(self._sym_hist) < self.fps * 2:
            return 0.0
        def _trend(d):
            y = np.array(d); x = np.arange(len(y))
            if np.std(x) < 1e-6: return 0.0
            return float(np.polyfit(x, y, 1)[0])
        sym_tr = _trend(self._sym_hist)   # negative = worsening
        stab_tr = _trend(self._stab_hist)
        deg = max(0.0, -(sym_tr + stab_tr) * self.fps * 20)
        return float(np.clip(deg, 0, 100))

    # ── state commit (persistence + hysteresis) ─────────────────────────────────

    def _commit(self, winner, win_score, scores):
        """
        Commit a state only after the SAME winner persists for persist_frames,
        with hysteresis so we don't flip-flop between close scores.
        Operates on BASE labels (WALK/RUN_FORWARD/JUMP/SQUAT/STRAFE_*/IDLE);
        directional resolution of WALK happens after commit.
        """
        cur = getattr(self, "_committed_base", "IDLE")

        if win_score < self.T.hysteresis:
            winner = "IDLE"
            win_score = scores.get("IDLE", 0.0)

        # Track candidate persistence: increment while the winner is stable.
        if winner == self._candidate:
            self._cand_frames += 1
        else:
            self._candidate = winner
            self._cand_frames = 1

        # Already in this state → stay.
        if winner == cur:
            return cur

        # Decide whether the candidate is allowed to take over.
        cur_score = scores.get(cur, 0.0)
        leaving_idle = (cur == "IDLE")
        margin_ok = leaving_idle or (win_score >= cur_score + self.T.hysteresis)

        if margin_ok and self._cand_frames >= self._persist_frames:
            return winner
        return cur

    # ── hybrid states + phase ────────────────────────────────────────────────────

    def _hybrid_and_phase(self, action, knee_flex, direction_deg, vert, flight_s):
        phase = ""
        exp = self.enable_experimental
        # Resolve generic WALK into a directional label from smoothed direction.
        # With experimental states OFF, WALK always resolves to WALK_FORWARD
        # (the only validated walking label); directional variants are stubbed.
        if action == "WALK":
            if not exp:
                action = "WALK_FORWARD"
            else:
                ad = abs(direction_deg)
                if ad <= self.T.direction_fwd_cone:
                    action = "WALK_FORWARD"
                elif ad >= 180 - self.T.direction_fwd_cone:
                    action = "WALK_BACKWARD"
                elif 20 < ad < 70:
                    action = "WALK_DIAGONAL_LEFT" if direction_deg < 0 else "WALK_DIAGONAL_RIGHT"
                else:
                    action = "STRAFE_LEFT" if direction_deg < 0 else "STRAFE_RIGHT"
        # CROUCH_WALK / RUN_STRAFE promotions are experimental-only.
        if exp:
            if action in ("WALK_FORWARD", "WALK_BACKWARD") and knee_flex > self.T.knee_flex_squat_deg * 0.7:
                action = "CROUCH_WALK"
            if action == "RUN_FORWARD" and abs(abs(direction_deg) - 90) < 30:
                action = "RUN_STRAFE_LEFT" if direction_deg < 0 else "RUN_STRAFE_RIGHT"
        # phases
        if action == "SQUAT":
            phase = "descending" if vert < -0.5 else "ascending" if vert > 0.5 else "hold"
        elif action == "JUMP":
            phase = "flight" if flight_s > 0 else "launch_or_land"
        elif action.startswith(("WALK", "RUN", "CROUCH", "STRAFE")):
            phase = "cyclic"
        return action, phase