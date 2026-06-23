"""
backend/biomechanics.py
-----------------------
Computes movement quality metrics from a stream of joint positions.
All metrics are designed to work incrementally (call .update() per frame).

Metrics computed:
  Stability  — sway velocity, CV of sway, pelvis oscillation
  Symmetry   — L/R knee angle delta, ankle height asymmetry
  Smoothness — jerk magnitude, acceleration entropy
  Fatigue    — rolling trend of key metrics over rep window
  Summary    — snapshot for AI coaching report

v4 CHANGE (sway decoupling):
_compute_sway() now tracks pelvis position RELATIVE to the skeleton's
own hip midpoint, not in absolute world space. This means root translation
(the skeleton walking through the world) has zero effect on the sway signal.
Postural sway is purely the pelvis wobbling relative to the stance base —
which is exactly the clinically meaningful signal.

The absolute-world fallback (raw_accel proxy) is preserved for pre-calibration.

v3 change: update() accepts translation_speed_ms from RootIntegrator.
A locomotion penalty (0–30 pts, linear 0.3→3.0 m/s) is subtracted from
the stability score at the per-frame level so that both the live metrics
AND the session summary reflect translation-adjusted stability.
sway_stability_raw is preserved alongside sway_stability throughout.
get_summary() builds final_stability from the adjusted score history so
the AI coaching report and tier assignment are also translation-aware.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────
SWAY_WINDOW    = 120   # frames for sway CV window (~2.4 s at 50 Hz)
JERK_WINDOW    = 30    # frames for jerk computation
FATIGUE_WINDOW = 300   # frames for fatigue trend (~6 s at 50 Hz)
EMA_ALPHA      = 0.5   # smoothing factor for velocity EMA

# Locomotion penalty constants (shared with main.py block — keep in sync)
_PENALTY_MAX   = 30.0
_SPEED_ONSET   = 0.3
_SPEED_FULL    = 3.0


def _locomotion_penalty(translation_speed_ms: float) -> float:
    if translation_speed_ms <= _SPEED_ONSET:
        return 0.0
    frac = min(1.0, (translation_speed_ms - _SPEED_ONSET) / (_SPEED_FULL - _SPEED_ONSET))
    return frac * _PENALTY_MAX


class BiomechanicsEngine:
    def __init__(self, fps: float = 50, test_id: str = "squat"):
        self.fps     = fps
        self.dt      = 1.0 / fps
        self.test_id = test_id
        self.reset()

    def reset(self):
        self._start_time: float = time.time()

        self._pos: dict[str, deque] = {
            j: deque(maxlen=SWAY_WINDOW) for j in
            ["pelvis", "hip_l", "hip_r", "knee_l", "knee_r", "ankle_l", "ankle_r"]
        }
        self._vel_ema: dict[str, np.ndarray] = {}

        self._sway_vels: deque    = deque(maxlen=SWAY_WINDOW)
        self._sway_vel_ema: float = 0.0
        self._cv_slow_ema: float  = 0.0
        self._cv_vel_slow: float  = 0.0   # very-slow EMA of velocity for CV
        self._cv_history: deque   = deque()

        self._stability_adjusted_history: deque = deque(maxlen=SWAY_WINDOW)
        self._sway_fatigue_history: deque        = deque(maxlen=FATIGUE_WINDOW)

        self._accel_history:   deque = deque(maxlen=JERK_WINDOW)
        self._knee_l_history:  deque = deque(maxlen=FATIGUE_WINDOW)
        self._knee_r_history:  deque = deque(maxlen=FATIGUE_WINDOW)
        self._ankle_y_history: deque = deque(maxlen=FATIGUE_WINDOW)
        self._asym_history:    deque = deque(maxlen=FATIGUE_WINDOW)
        self._smooth_history:  deque = deque(maxlen=FATIGUE_WINDOW)

        # v4: track previous relative pelvis position for sway
        self._prev_pelvis_rel: Optional[np.ndarray] = None

        self._rep_count:     int  = 0
        self._rep_phase:     str  = "up"
        self._rep_snapshots: list = []
        self._frame:         int  = 0

    # ─────────────────────────────────────────────
    # MAIN UPDATE
    # ─────────────────────────────────────────────

    def update(
        self,
        joints:               dict,
        timestamp:            float,
        raw_accel:            Optional[np.ndarray] = None,
        translation_speed_ms: float                = 0.0,
    ) -> dict:
        self._frame += 1
        j = {k: np.array(v) for k, v in joints.items()}

        for name, pos in j.items():
            if name in self._pos:
                self._pos[name].append(pos.copy())

        # ── Stability ────────────────────────────────────────────────────────
        sway_vel, sway_cv = self._compute_sway(j, raw_accel)
        self._sway_fatigue_history.append(sway_vel)

        # Stability = 100 - cv.  CV is driven by a slow EMA of velocity so
        # small frame-to-frame changes don't cause large stability swings.
        stability_raw = round(max(0.0, 100.0 - sway_cv), 1)
        penalty       = _locomotion_penalty(translation_speed_ms)
        stability_adj = round(float(np.clip(stability_raw - penalty, 0.0, 100.0)), 1)
        self._stability_adjusted_history.append(stability_adj)

        # ── Symmetry ─────────────────────────────────────────────────────────
        sym = self._compute_symmetry(j)

        # ── Smoothness ───────────────────────────────────────────────────────
        smooth = self._compute_smoothness(j, raw_accel)

        # ── Fatigue ──────────────────────────────────────────────────────────
        self._asym_history.append(sym["knee_asymmetry_deg"])
        self._smooth_history.append(smooth["jerk_magnitude"])
        fatigue = self._compute_fatigue(sway_vel, stability_adj)

        # ── Rep detection ────────────────────────────────────────────────────
        knee_mean = (sym["knee_l_deg"] + sym["knee_r_deg"]) / 2
        self._detect_rep(knee_mean)

        metrics = {
            "sway_velocity":           round(sway_vel, 4),
            "sway_cv_pct":             round(sway_cv, 2),
            "sway_stability":          stability_adj,
            "sway_stability_raw":      stability_raw,
            "locomotion_penalty":      round(penalty, 2),
            "translation_speed_ms":    round(translation_speed_ms, 4),
            "sway_stability_adjusted": penalty > 0.0,

            "knee_l_deg":              round(sym["knee_l_deg"], 1),
            "knee_r_deg":              round(sym["knee_r_deg"], 1),
            "knee_asymmetry_deg":      round(sym["knee_asymmetry_deg"], 1),
            "ankle_height_asym_m":     round(sym["ankle_height_asym_m"], 4),

            "jerk_magnitude":          round(smooth["jerk_magnitude"], 4),
            "accel_entropy":           round(smooth["accel_entropy"], 3),

            "asym_trend":              round(fatigue["asym_trend"], 3),
            "jerk_trend":              round(fatigue["jerk_trend"], 3),
            "sway_trend":              round(fatigue["sway_trend"], 3),
            "fatigue_score":           round(fatigue["score"], 1),
            "stability_tier":          fatigue.get("stability_tier", ""),

            "rep_count":               self._rep_count,
            "rep_phase":               self._rep_phase,
            "frame":                   self._frame,
        }

        return metrics

    # ─────────────────────────────────────────────
    # STABILITY
    # ─────────────────────────────────────────────

    # Per-frame displacement below this is noise, not sway.
    _SWAY_DEADBAND_M   = 0.0003   # 0.3 mm

    # EMA decay per quiet frame — drives velocity to 0 at rest.
    _SWAY_EMA_DECAY    = 0.92

    # Noise floor for the accel fallback path.
    _ACCEL_NOISE_FLOOR = 0.5      # m/s²

    def _compute_sway(
        self,
        j:         dict,
        raw_accel: Optional[np.ndarray] = None,
    ) -> tuple:
        """
        Returns (sway_velocity m/s, sway_cv_pct %).

        sway_velocity   EMA-smoothed per-frame displacement × fps (m/s).
                        Decays to 0 at rest. Drives stability_score and
                        the locomotion penalty.

        sway_cv_pct     Display-only: CV of the EMA velocity buffer, capped
                        at 100%. Not used for the stability score because CV
                        of a sinusoidal signal is amplitude-invariant (~49%
                        regardless of whether you sway 1 cm or 10 cm). Kept
                        as a UI metric showing signal variability.

        stability_score = _vel_to_stability(sway_velocity), which is
        amplitude-sensitive: 10 cm sway correctly scores lower than 1 cm.

        Fallback priority:
          1. Relative pelvis-to-hip-midpoint XZ — translation-immune.
          2. Absolute pelvis XZ — legacy / pre-calibration.
          3. Gravity-deviation accel magnitude — noise-floored.
        """
        hip_l   = j.get("hip_l")
        hip_r   = j.get("hip_r")
        pelvis  = j.get("pelvis")
        ankle_l = j.get("ankle_l")
        ankle_r = j.get("ankle_r")

        # ── Primary: pelvis relative to ankle midpoint ────────────────────────
        # Hip midpoint is computed by rotating a fixed offset by q_pelvis, so
        # it orbits the pelvis origin whenever the IMU rotates — even with no
        # physical translation. This makes pure rotation spike sway velocity.
        #
        # Ankle midpoint is driven by thigh+shin sensors further down the chain,
        # so it is largely decoupled from pelvis rotation. Using it as the base
        # means sway = pelvis drifting relative to the feet, which is the
        # clinically correct definition of postural sway.
        #
        # Falls back to hip midpoint if ankles are not available.
        if pelvis is not None and ankle_l is not None and ankle_r is not None:
            base   = (ankle_l + ankle_r) * 0.5
            rel_xz = np.array([pelvis[0] - base[0], pelvis[2] - base[2]])

            if self._prev_pelvis_rel is not None:
                dist = float(np.linalg.norm(rel_xz - self._prev_pelvis_rel))
                if dist >= self._SWAY_DEADBAND_M:
                    self._sway_vel_ema = (
                        EMA_ALPHA * self._sway_vel_ema
                        + (1 - EMA_ALPHA) * dist * self.fps
                    )
                else:
                    self._sway_vel_ema *= self._SWAY_EMA_DECAY
                self._sway_vels.append(self._sway_vel_ema)

            self._prev_pelvis_rel = rel_xz
            return self._vel_and_cv()

        # ── Secondary: pelvis relative to hip midpoint (fallback) ─────────────
        if pelvis is not None and hip_l is not None and hip_r is not None:
            hip_mid = (hip_l + hip_r) * 0.5
            rel_xz  = np.array([pelvis[0] - hip_mid[0], pelvis[2] - hip_mid[2]])

            if self._prev_pelvis_rel is not None:
                dist = float(np.linalg.norm(rel_xz - self._prev_pelvis_rel))
                if dist >= self._SWAY_DEADBAND_M:
                    self._sway_vel_ema = (
                        EMA_ALPHA * self._sway_vel_ema
                        + (1 - EMA_ALPHA) * dist * self.fps
                    )
                else:
                    self._sway_vel_ema *= self._SWAY_EMA_DECAY
                self._sway_vels.append(self._sway_vel_ema)

            self._prev_pelvis_rel = rel_xz
            return self._vel_and_cv()

        # ── Tertiary: absolute pelvis XZ ──────────────────────────────────────
        history = self._pos["pelvis"]
        if len(history) >= 2:
            dist = float(np.linalg.norm(history[-1][[0,2]] - history[-2][[0,2]]))
            if dist >= self._SWAY_DEADBAND_M:
                self._sway_vel_ema = (
                    EMA_ALPHA * self._sway_vel_ema
                    + (1 - EMA_ALPHA) * dist * self.fps
                )
            else:
                self._sway_vel_ema *= self._SWAY_EMA_DECAY
            self._sway_vels.append(self._sway_vel_ema)
            return self._vel_and_cv()

        # ── Quaternary: accel proxy ───────────────────────────────────────────
        if raw_accel is not None and len(raw_accel) >= 3:
            linear_mag = abs(float(np.linalg.norm(raw_accel)) - 9.81)
            if linear_mag >= self._ACCEL_NOISE_FLOOR:
                self._sway_vel_ema = (
                    EMA_ALPHA * self._sway_vel_ema
                    + (1 - EMA_ALPHA) * linear_mag
                )
            else:
                self._sway_vel_ema *= self._SWAY_EMA_DECAY
            self._sway_vels.append(self._sway_vel_ema)
            return self._vel_and_cv()

        return 0.0, 0.0

    def _vel_and_cv(self) -> tuple:
        """
        Returns (ema_velocity m/s, cv_pct %).

        cv_pct is a slow EMA of (velocity / ceiling * 100), capped at 100.
        Using a slow EMA (alpha=0.05, ~20-frame time constant) means:
          - Small frame-to-frame velocity changes → tiny CV change
          - CV only moves meaningfully when sway is sustained
          - Still → CV decays smoothly back to 0 (no sudden jumps)

        ceiling = 0.30 m/s maps to 100% CV.  Adjust to taste:
          lower ceiling → more sensitive (CV reaches 100 at smaller sway)
          higher ceiling → less sensitive
        """
        vel = float(self._sway_vel_ema)

        # ── CV via very-slow EMA of velocity ─────────────────────────────────
        # Instead of std/mean (which reacts to every within-cycle oscillation),
        # run a second, much slower EMA on the velocity. This acts as a
        # low-pass filter with a ~4s time constant — it only responds to
        # sustained changes in sway level, ignoring momentary spikes.
        #
        # inst_cv maps the slow EMA velocity linearly to 0–100%:
        #   0 m/s   → 0%   (still)
        #   0.30 m/s → 100% (heavy sway)
        #
        # _ALPHA_SLOW = 0.01 → time constant ~100 frames (~2s at 50Hz)
        # A sudden vel change can move CV by at most 1% per frame.
        _CEILING    = 0.30
        _ALPHA_SLOW = 0.1 # lower = less sensitive

        self._cv_vel_slow = _ALPHA_SLOW * vel + (1.0 - _ALPHA_SLOW) * self._cv_vel_slow
        if vel < 0.001:
            self._cv_vel_slow *= 0.95   # decay to 0 when still

        inst_cv = min(100.0, self._cv_vel_slow / _CEILING * 100.0)
        self._cv_slow_ema = inst_cv   # already slow — no second EMA needed

        cv = round(float(self._cv_slow_ema), 2)
        self._cv_history.append(cv)
        return vel, cv


    # ─────────────────────────────────────────────
    # SYMMETRY
    # ─────────────────────────────────────────────

    def _compute_symmetry(self, j: dict) -> dict:
        knee_l = _joint_angle(j.get("hip_l"), j.get("knee_l"), j.get("ankle_l"))
        knee_r = _joint_angle(j.get("hip_r"), j.get("knee_r"), j.get("ankle_r"))
        self._knee_l_history.append(knee_l)
        self._knee_r_history.append(knee_r)
        ankle_asym = 0.0
        if j.get("ankle_l") is not None and j.get("ankle_r") is not None:
            ankle_asym = float(abs(j["ankle_l"][1] - j["ankle_r"][1]))
        return {
            "knee_l_deg":          knee_l,
            "knee_r_deg":          knee_r,
            "knee_asymmetry_deg":  abs(knee_l - knee_r),
            "ankle_height_asym_m": ankle_asym,
        }

    # ─────────────────────────────────────────────
    # SMOOTHNESS
    # ─────────────────────────────────────────────

    def _compute_smoothness(
        self,
        j:         dict,
        raw_accel: Optional[np.ndarray] = None,
    ) -> dict:
        history = self._pos["pelvis"]

        if len(history) >= 4:
            pos   = np.array(list(history)[-4:])
            vel   = np.diff(pos,   axis=0) * self.fps
            accel = np.diff(vel,   axis=0) * self.fps
            jerk  = np.diff(accel, axis=0) * self.fps
            jerk_mag  = float(np.linalg.norm(jerk[0]))
            accel_mag = np.linalg.norm(accel, axis=1)
            self._accel_history.extend(accel_mag.tolist())
            entropy = _signal_entropy(list(self._accel_history))
            return {"jerk_magnitude": jerk_mag, "accel_entropy": entropy}

        if raw_accel is not None and len(raw_accel) >= 3:
            mag = float(np.linalg.norm(raw_accel))
            self._accel_history.append(mag)
            jerk_mag = (
                abs(mag - list(self._accel_history)[-2]) * self.fps
                if len(self._accel_history) >= 2 else 0.0
            )
            entropy = _signal_entropy(list(self._accel_history))
            return {"jerk_magnitude": jerk_mag, "accel_entropy": entropy}

        return {"jerk_magnitude": 0.0, "accel_entropy": 0.0}

    # ─────────────────────────────────────────────
    # FATIGUE
    # ─────────────────────────────────────────────

    def _compute_fatigue(
        self,
        sway_vel:      float = 0.0,
        stability_adj: float = 100.0,
    ) -> dict:
        if self.test_id == "stability":
            tier          = _stability_tier(stability_adj)
            fatigue_score = max(0.0, 100.0 - stability_adj)
            sway_trend    = _linear_trend(list(self._sway_fatigue_history))
            return {
                "asym_trend":     0.0,
                "jerk_trend":     0.0,
                "sway_trend":     sway_trend,
                "score":          round(fatigue_score, 1),
                "stability_tier": tier,
            }

        asym_trend = _linear_trend(list(self._asym_history))
        jerk_trend = _linear_trend(list(self._smooth_history))
        asym_score = min(100.0, max(0.0, asym_trend * 100))
        jerk_score = min(100.0, max(0.0, jerk_trend * 50))
        return {
            "asym_trend":     asym_trend,
            "jerk_trend":     jerk_trend,
            "sway_trend":     0.0,
            "score":          (asym_score + jerk_score) / 2,
            "stability_tier": "",
        }

    # ─────────────────────────────────────────────
    # REP DETECTION
    # ─────────────────────────────────────────────

    def _detect_rep(self, knee_angle_mean: float):
        if self._rep_phase == "up" and knee_angle_mean > 60:
            self._rep_phase = "down"
        elif self._rep_phase == "down" and knee_angle_mean < 20:
            self._rep_phase = "up"
            self._rep_count += 1
            self._snap_rep_metrics()

    def _snap_rep_metrics(self):
        snap = {
            "rep":       self._rep_count,
            "sway_cv":   round(float(np.mean(self._sway_vels))     if self._sway_vels     else 0, 2),
            "asym_mean": round(float(np.mean(self._asym_history))   if self._asym_history   else 0, 2),
            "jerk_mean": round(float(np.mean(self._smooth_history)) if self._smooth_history else 0, 4),
        }
        self._rep_snapshots.append(snap)

    # ─────────────────────────────────────────────
    # SESSION SUMMARY
    # ─────────────────────────────────────────────

    def get_summary(self) -> dict:
        sway     = list(self._sway_vels)
        asym     = list(self._asym_history)
        jerk     = list(self._smooth_history)
        sway_fat = list(self._sway_fatigue_history)
        stab_adj = list(self._stability_adjusted_history)

        mean_sway          = float(np.mean(sway)) if sway else 0.0
        cv_hist            = list(self._cv_history)
        final_cv           = float(np.mean(cv_hist)) if cv_hist else 0.0
        stability_raw      = max(0.0, 100.0 - final_cv)
        # Derive adjusted score from the same CV history, minus mean penalty.
        # Using mean(stab_adj_history) over-weights the still warmup frames
        # where CV=0 inflates the mean. 100 - mean(cv) is the honest average.
        mean_penalty       = float(np.mean([_locomotion_penalty(v) for v in list(self._sway_fatigue_history)])) if self._sway_fatigue_history else 0.0
        stability_adjusted = float(np.clip(stability_raw - mean_penalty, 0.0, 100.0))

        final_tier = _stability_tier(stability_adjusted) if self.test_id == "stability" else None
        if self.test_id == "stability":
            fatigue_detected = stability_adjusted <= 49.0
        else:
            fatigue_detected = (
                _linear_trend(asym) > 0.01
                or _linear_trend(jerk) > 0.001
            )

        return {
            "total_reps":   self._rep_count,
            "total_frames": self._frame,
            "duration_s":   round(time.time() - self._start_time, 1),

            "sway": {
                "mean_velocity_ms":    round(mean_sway, 4),
                "cv_pct":              round(final_cv, 2),
                "stability_score":     round(stability_adjusted, 1),
                "stability_score_raw": round(stability_raw, 1),
            },

            "symmetry": {
                "mean_knee_asym_deg": round(float(np.mean(asym)) if asym else 0, 2),
                "max_knee_asym_deg":  round(float(np.max(asym))  if asym else 0, 2),
                "grade":              _grade_asym(float(np.mean(asym)) if asym else 0),
            },

            "smoothness": {
                "mean_jerk":  round(float(np.mean(jerk)) if jerk else 0, 5),
                "jerk_trend": round(_linear_trend(jerk), 5),
                "grade":      _grade_jerk(float(np.mean(jerk)) if jerk else 0),
            },

            "fatigue": {
                "asym_trend_per_frame":  round(_linear_trend(asym), 5),
                "jerk_trend_per_frame":  round(_linear_trend(jerk), 5),
                "sway_trend_per_frame":  round(_linear_trend(sway_fat), 5),
                "detected":              fatigue_detected,
                "stability_tier":        final_tier,
            },

            "per_rep": self._rep_snapshots,
        }


# ─────────────────────────────────────────────
# PURE HELPERS
# ─────────────────────────────────────────────

def _stability_tier(score: float) -> str:
    if score >= 95: return "Peak"
    if score >= 85: return "Fresh"
    if score >= 75: return "Ready"
    if score >= 65: return "Loaded"
    if score >= 50: return "Strained"
    if score >= 35: return "Fatigued"
    if score >= 20: return "Overreached"
    return "Critical"


def _vel_to_stability(sway_vel_ms: float) -> float:
    """
    Map absolute sway velocity (m/s) to a 0–100 stability score.

    Uses a soft exponential decay so the score:
      - is exactly 100 at 0 m/s  (perfect stillness)
      - is ~90 at 0.01 m/s       (clinical "excellent" postural sway)
      - is ~75 at 0.03 m/s       (normal quiet standing)
      - is ~50 at 0.07 m/s       (noticeable sway)
      - is ~20 at 0.15 m/s       (significant instability)
      - approaches 0 above 0.3 m/s

    This replaces CV (std/mean×100) which explodes near zero and is
    undefined when the EMA is small, causing 0↔100 oscillations on the
    still→moving transition.

    Tune the `k` constant to adjust sensitivity:
      higher k → score drops faster with small movements
      lower  k → score stays high even with moderate sway
    """
    k = 8.0    # decay rate — tuned so:
               #   0.015 m/s (minimal sway)    ≈ 89  → "Fresh"
               #   0.047 m/s (normal standing) ≈ 69  → "Loaded"
               #   0.094 m/s (moderate sway)   ≈ 47  → "Strained"
               #   0.157 m/s (significant)     ≈ 29  → "Fatigued"
    score = 100.0 * float(np.exp(-k * max(0.0, sway_vel_ms)))
    return round(float(np.clip(score, 0.0, 100.0)), 1)


def _joint_angle(
    a: Optional[np.ndarray],
    b: Optional[np.ndarray],
    c: Optional[np.ndarray],
) -> float:
    if a is None or b is None or c is None:
        return 0.0
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_a = np.dot(v1, v2) / (n1 * n2)
    return 180.0 - float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def _signal_entropy(values: list, bins: int = 16) -> float:
    if len(values) < 4:
        return 0.0
    v = np.array(values)
    hist, _ = np.histogram(v, bins=bins, density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log(hist + 1e-12)))


def _linear_trend(values: list) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom  = float(np.sum((x - x_mean) ** 2))
    if denom < 1e-12:
        return 0.0
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def _grade_asym(deg: float) -> str:
    if deg < 3:  return "excellent"
    if deg < 7:  return "good"
    if deg < 12: return "moderate"
    return "poor"


def _grade_jerk(mean_jerk: float) -> str:
    if mean_jerk < 0.5: return "smooth"
    if mean_jerk < 2.0: return "moderate"
    return "rough"