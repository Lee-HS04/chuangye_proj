"""
motion_detector.py  —  Pelvis-only locomotion + action detection
=================================================================

Detects the full set of actions the system needs to animate and log:

  idle              — standing still
  march_in_place    — stepping without travel
  walk              — walking forward
  run               — running
  walk_back         — walking backward
  strafe_left       — lateral step left
  strafe_right      — lateral step right
  squat             — sustained knee-bend (low pelvis + forward tilt)
  jump              — airborne phase (gravity-zero + impact transient)
  tilt_fwd          — forward lean without steps

Detection gates
---------------
Three independent signals must agree for locomotion:
  1. CADENCE    — vertical bounce rate (stepping)
  2. TRAVEL     — sustained horizontal accel in a consistent direction
  3. DIRECTION  — horizontal accel direction relative to body heading

Squat and jump are detected from pelvis height change and vertical
acceleration patterns independently of the locomotion detector.

Why not integrate to velocity?
-------------------------------
Double-integrating accelerometer drifts within 1 second.  We use
direction + magnitude persistence instead.

Outputs (MotionState)
---------------------
  action       : string action label
  move_dir     : [x_right, z_forward] unit vector in body frame
  cadence_hz   : step rate
  speed_grade  : 0=idle, 1=walk, 2=run
  h_accel      : filtered horizontal accel magnitude (m/s²)
  confidence   : 0–1
  in_place     : True when stepping but not travelling
  is_squat     : True during squat hold
  is_jump      : True during airborne/impact phase
  jump_phase   : 'ascent' | 'apex' | 'descent' | 'impact' | 'none'
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

GRAVITY = 9.81


# ── quat helpers ──────────────────────────────────────────────────────────────

def _qmul(q1, q2):
    w1,x1,y1,z1=q1; w2,x2,y2,z2=q2
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2,w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2])

def _qconj(q): return np.array([q[0],-q[1],-q[2],-q[3]])

def _qrot(q, v):
    qv=np.array([0.0,v[0],v[1],v[2]])
    return _qmul(_qmul(q,qv),_qconj(q))[1:]


# ── state ─────────────────────────────────────────────────────────────────────

@dataclass
class MotionState:
    action:      str   = 'idle'
    move_dir:    list  = field(default_factory=lambda: [0.0, 0.0])
    cadence_hz:  float = 0.0
    speed_grade: float = 0.0
    h_accel:     float = 0.0
    confidence:  float = 0.0
    in_place:    bool  = False
    is_squat:    bool  = False
    is_jump:     bool  = False
    jump_phase:  str   = 'none'   # ascent | apex | descent | impact | none

    def to_dict(self):
        d = asdict(self)
        d['move_dir'] = [round(float(x), 3) for x in self.move_dir]
        for k in ('cadence_hz','speed_grade','h_accel','confidence'):
            d[k] = round(float(d[k]), 3)
        return d


class MotionDetector:
    """
    Feed pelvis packets via update(quat, accel, gyro, ts).
    Returns MotionState every call.

    accel is expected in m/s² in the SENSOR frame as-is from ble_receiver.
    """

    def __init__(self, fps: float = 50.0):
        self.fps = fps
        self._win = max(8, int(fps * 0.8))

        # Locomotion windows
        self._vert:        deque = deque(maxlen=self._win)
        self._horiz:       deque = deque(maxlen=self._win)
        self._horiz_body:  deque = deque(maxlen=self._win)
        self._step_times:  deque = deque(maxlen=8)
        self._last_vert_sign = 0
        self._last_step_t    = 0.0

        # Squat detection windows
        self._vert_accel_sq: deque = deque(maxlen=max(4, int(fps * 0.4)))
        self._pitch_window:  deque = deque(maxlen=max(4, int(fps * 0.4)))
        self._squat_hold_t:  float = 0.0
        self._squat_active:  bool  = False

        # Jump detection windows
        self._jump_vert_win: deque = deque(maxlen=max(4, int(fps * 0.3)))
        self._in_flight:     bool  = False
        self._flight_start:  float = 0.0
        self._impact_pending:bool  = False
        self._impact_t:      float = 0.0
        self._jump_phase:    str   = 'none'

        # Tilt forward detection
        self._tilt_window: deque = deque(maxlen=max(4, int(fps * 0.5)))

        # Smoothed horizontal accel (low-pass)
        self._h_filt      = np.zeros(2)
        self._h_filt_body = np.zeros(2)

        # Tunables
        self.travel_thresh    = 1.2
        self.bounce_thresh    = 1.5
        self.run_cadence_hz   = 2.7
        self.walk_cadence_hz  = 1.0
        self.dir_stability    = 0.6
        self._lp_alpha        = 0.25

        # Squat params
        self.squat_vert_thresh   = -0.5    # m/s² downward linear accel (pelvis sinking)
        self.squat_pitch_thresh  = 8.0     # abs degrees of forward pitch (check: avg_pitch < -this)
        self.squat_hold_s        = 0.25    # seconds before squat is confirmed

        # Jump params
        self.jump_launch_thresh  = 3.5     # m/s² upward linear accel (push-off)
        self.jump_flight_thresh  = 0.3     # gravity fraction: near-zero = airborne
        self.jump_impact_thresh  = 4.0     # m/s² downward transient on landing

        self._state = MotionState()

    # ── tuning ────────────────────────────────────────────────────────────────

    def set_params(self, **kw):
        for k, v in kw.items():
            if hasattr(self, k) and v is not None:
                setattr(self, k, float(v))

    def reset(self):
        for q in (self._vert, self._horiz, self._horiz_body, self._step_times,
                  self._vert_accel_sq, self._pitch_window,
                  self._jump_vert_win, self._tilt_window):
            q.clear()
        self._h_filt[:] = 0; self._h_filt_body[:] = 0
        self._last_vert_sign = 0; self._last_step_t = 0.0
        self._squat_active = False; self._in_flight = False
        self._jump_phase = 'none'
        self._state = MotionState()

    def calibrate_from_window(self) -> float:
        if len(self._horiz_body) < 4:
            return self.travel_thresh
        mags = [np.linalg.norm(v) for v in self._horiz_body]
        floor = float(np.percentile(mags, 90))
        self.travel_thresh = max(0.6, floor * 1.6)
        return self.travel_thresh

    # ── main update ───────────────────────────────────────────────────────────

    def update(
        self,
        quat: np.ndarray,
        accel: np.ndarray,
        gyro: np.ndarray,
        ts: float,
    ) -> MotionState:
        q = quat.astype(float)
        a_sensor = accel.astype(float)

        # World-frame linear acceleration
        a_world = _qrot(q, a_sensor)
        a_lin   = a_world - np.array([0.0, GRAVITY, 0.0])

        vert         = float(a_lin[1])
        horiz_world  = np.array([a_lin[0], a_lin[2]])

        # Body-frame horizontal
        fwd_w = _qrot(q, np.array([0.0, 0.0, -1.0]))
        rgt_w = _qrot(q, np.array([1.0, 0.0, 0.0]))
        f2 = np.array([fwd_w[0], fwd_w[2]]); nf = np.linalg.norm(f2)
        r2 = np.array([rgt_w[0], rgt_w[2]]); nr = np.linalg.norm(r2)
        f2 = f2/nf if nf>1e-6 else np.array([0.0,-1.0])
        r2 = r2/nr if nr>1e-6 else np.array([1.0, 0.0])
        horiz_body = np.array([np.dot(horiz_world, r2), np.dot(horiz_world, f2)])

        # Low-pass filter
        self._h_filt      = (1-self._lp_alpha)*self._h_filt      + self._lp_alpha*horiz_world
        self._h_filt_body = (1-self._lp_alpha)*self._h_filt_body + self._lp_alpha*horiz_body

        self._vert.append(vert)
        self._horiz.append(self._h_filt.copy())
        self._horiz_body.append(self._h_filt_body.copy())

        # ── 1. CADENCE ────────────────────────────────────────────────────
        v_arr = np.array(self._vert)
        v_ac  = v_arr - v_arr.mean() if len(v_arr) else v_arr
        bounce_amp = float(np.percentile(np.abs(v_ac), 80)) if len(v_ac) else 0.0
        mean_vert = float(v_arr.mean()) if len(v_arr) else 0.0
        sign = 1 if (vert - mean_vert) > 0 else -1
        if (self._last_vert_sign <= 0 and sign > 0
                and bounce_amp > self.bounce_thresh
                and ts - self._last_step_t > 0.18):
            self._step_times.append(ts)
            self._last_step_t = ts
        self._last_vert_sign = sign
        cadence_hz = self._cadence(ts)

        # ── 2. TRAVEL ─────────────────────────────────────────────────────
        h_mag     = float(np.linalg.norm(self._h_filt_body))
        stability = self._dir_stability()
        stepping  = cadence_hz >= self.walk_cadence_hz and bounce_amp > self.bounce_thresh
        traveling = h_mag >= self.travel_thresh and stability >= self.dir_stability

        # ── 3. SQUAT DETECTION ────────────────────────────────────────────
        self._vert_accel_sq.append(vert)
        # Pitch from quaternion
        w, x, y, z = q
        pitch_deg = float(np.degrees(np.arcsin(np.clip(2*(w*x - y*z), -1, 1))))
        self._pitch_window.append(pitch_deg)

        avg_vert_sq = float(np.mean(self._vert_accel_sq)) if self._vert_accel_sq else 0.0
        avg_pitch   = float(np.mean(self._pitch_window))   if self._pitch_window   else 0.0

        # Squat: pelvis sinks (negative vert accel), body tilts forward, no cadence.
        # Forward tilt = negative pitch in this remap convention, so we check < threshold.
        squat_signal = (
            avg_vert_sq < self.squat_vert_thresh
            and avg_pitch < -self.squat_pitch_thresh
            and cadence_hz < self.walk_cadence_hz
        )
        if squat_signal:
            if not self._squat_active:
                self._squat_hold_t = ts
            elif ts - self._squat_hold_t >= self.squat_hold_s:
                self._squat_active = True
        else:
            self._squat_active = False

        # ── 4. JUMP DETECTION ─────────────────────────────────────────────
        a_mag = float(np.linalg.norm(a_sensor))
        a_g   = a_mag / GRAVITY

        self._jump_vert_win.append(vert)
        avg_vert_j = float(np.mean(self._jump_vert_win)) if self._jump_vert_win else 0.0

        # Launch: strong upward accel while not in flight
        if not self._in_flight and vert > self.jump_launch_thresh and cadence_hz < 1.0:
            self._in_flight  = True
            self._flight_start = ts
            self._jump_phase   = 'ascent'

        if self._in_flight:
            flight_t = ts - self._flight_start
            # Apex: gravity component near zero (weightless)
            if a_g < self.jump_flight_thresh:
                self._jump_phase = 'apex'
            # Descent: negative vert after apex
            elif self._jump_phase == 'apex' and vert < -1.0:
                self._jump_phase = 'descent'
            # Impact: large downward transient
            if vert < -self.jump_impact_thresh:
                self._jump_phase = 'impact'
                self._in_flight  = False
                self._impact_pending = True
                self._impact_t = ts

        # Clear impact after 0.3 s
        if self._impact_pending and ts - self._impact_t > 0.3:
            self._impact_pending = False
            self._jump_phase = 'none'

        is_jump = self._in_flight or self._impact_pending

        # ── 5. TILT FORWARD ───────────────────────────────────────────────
        self._tilt_window.append(pitch_deg)
        avg_tilt = float(np.mean(self._tilt_window)) if self._tilt_window else 0.0
        # Forward tilt produces NEGATIVE pitch in the remap convention used.
        # The sensor at rest (standing still) already reads around -12 to -14°
        # due to pelvis mounting angle, so the threshold must clear that floor.
        # -25.0° requires ~11° of additional forward lean beyond the resting pose.
        is_tilt_fwd = avg_tilt < -25.0 and cadence_hz < self.walk_cadence_hz and not self._squat_active

        # ── 6. ASSEMBLE STATE ─────────────────────────────────────────────
        st = MotionState()
        st.cadence_hz  = cadence_hz
        st.h_accel     = h_mag
        st.is_squat    = self._squat_active
        st.is_jump     = is_jump
        st.jump_phase  = self._jump_phase

        # Priority: jump > squat > locomotion
        if is_jump:
            st.action     = 'jump'
            st.confidence = 0.9
        elif self._squat_active:
            st.action     = 'squat'
            st.confidence = 0.85
        elif is_tilt_fwd and not stepping:
            st.action     = 'tilt_fwd'
            st.confidence = 0.7
        elif stepping and traveling:
            d  = self._h_filt_body.copy()
            dn = np.linalg.norm(d)
            d  = d/dn if dn > 1e-6 else np.array([0.0, 1.0])
            fwd_comp   =  d[1]
            right_comp =  d[0]
            st.move_dir    = [float(right_comp), float(-fwd_comp)]
            st.speed_grade = self._speed_grade(cadence_hz)
            st.confidence  = min(1.0, (h_mag/(self.travel_thresh*2)) * stability)

            if abs(fwd_comp) >= abs(right_comp):
                st.action = 'run' if cadence_hz >= self.run_cadence_hz else 'walk'
                if fwd_comp < 0:
                    st.action = 'walk_back'
            else:
                st.action = 'strafe_right' if right_comp > 0 else 'strafe_left'
        elif stepping:
            st.action    = 'march_in_place'
            st.in_place  = True
            st.speed_grade = self._speed_grade(cadence_hz)
        else:
            st.action = 'idle'

        self._state = st
        return st

    # ── helpers ───────────────────────────────────────────────────────────────

    def _cadence(self, now: float) -> float:
        if len(self._step_times) < 2: return 0.0
        dts = np.diff(np.array(self._step_times))
        if now - self._step_times[-1] > 0.9: return 0.0
        mean_dt = float(np.mean(dts[-5:])) if len(dts) else 0.0
        return 1.0/mean_dt if mean_dt > 1e-3 else 0.0

    def _speed_grade(self, cadence: float) -> float:
        if cadence < self.walk_cadence_hz: return 0.0
        if cadence >= self.run_cadence_hz: return 2.0
        return 1.0 + (cadence - self.walk_cadence_hz) / (self.run_cadence_hz - self.walk_cadence_hz)

    def _dir_stability(self) -> float:
        if len(self._horiz_body) < 4: return 0.0
        vecs = [v for v in list(self._horiz_body)[-6:] if np.linalg.norm(v) > 1e-3]
        if len(vecs) < 3: return 0.0
        units = [v/np.linalg.norm(v) for v in vecs]
        return float(np.linalg.norm(np.mean(units, axis=0)))

    def get_params(self) -> dict:
        return {
            'travel_thresh':   self.travel_thresh,
            'bounce_thresh':   self.bounce_thresh,
            'run_cadence_hz':  self.run_cadence_hz,
            'walk_cadence_hz': self.walk_cadence_hz,
            'dir_stability':   self.dir_stability,
        }