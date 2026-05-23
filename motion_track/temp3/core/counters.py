# core/counters.py

import numpy as np
from collections import deque


# ============================================================
# REP COUNTER
# ============================================================
class RepCounter:
    def __init__(self, exercise, feature, min_angle, max_angle):
        self.exercise = exercise
        self.feature = feature
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.reps = 0
        self.state = "up"

    def update(self, features):
        val = features.get(self.feature)

        if val is None:
            return self.reps

        if val <= self.min_angle and self.state == "up":
            self.state = "down"

        elif val >= self.max_angle and self.state == "down":
            self.state = "up"
            self.reps += 1

        return self.reps
    
    def get_fppa(self):
        return self.reps


# ============================================================
# SWAY TRACKER
# ============================================================
class SwayTracker:
    def __init__(self, fps=60, threshold=0.0):
        self.positions = deque(maxlen=2)

        self.cv_history = []

        # ✅ FULL HISTORY (no maxlen)
        self.vel_history = []

        self.fps = fps
        self.threshold = threshold

        self.sway_velocity = 0.0
        self.cv = 0.0
        self.one_minus_cv = 0.0
        self.final_cv = None
        self.final_one_minus_cv = None

    def update(self, mid_hip, mid_shoulder):
        if mid_hip is None or mid_shoulder is None:
            return

        hip = np.array(mid_hip, dtype=float)
        shoulder = np.array(mid_shoulder, dtype=float)

        if hip.shape[0] < 3 or shoulder.shape[0] < 3:
            return

        # weights
        w_hip = 0.7
        w_shoulder = 0.3

        v = w_hip * hip + w_shoulder * shoulder

        self.positions.append(v)

        if len(self.positions) < 2:
            return

        p1, p2 = self.positions[-2], self.positions[-1]

        diff = p2 - p1
        dist = np.linalg.norm(diff)

        if dist < self.threshold:
            dist = 0.0

        velocity = dist * self.fps


        # ----------------------------
        # SMOOTH VELOCITY (IMPORTANT)
        # ----------------------------
        alpha_v = 0.9

        if not hasattr(self, "vel_ema"):
            self.vel_ema = velocity
        else:
            self.vel_ema = alpha_v * self.vel_ema + (1 - alpha_v) * velocity

        # store smoothed velocity instead of raw
        self.vel_history.append(self.vel_ema)

        # ----------------------------
        # mean sway velocity (full video)
        # ----------------------------
        self.sway_velocity = float(np.mean(self.vel_history))



        # ----------------------------
        # CV over ENTIRE VIDEO
        # ----------------------------
        if len(self.vel_history) > 5:
            v = np.array(self.vel_history)

            mean_v = np.mean(v)
            std_v = np.std(v)

            if mean_v > 1e-6:
                self.cv = (std_v / mean_v) * 100
                self.one_minus_cv = 100 - self.cv
                self.cv_history.append(self.cv)

            else:
                self.cv = 0.0
        else:
            self.cv = 0.0

    def get_sway_velocity(self):
        return self.sway_velocity

    def get_cv(self):
        return self.cv
    
    def get_one_minus_cv(self):
        return self.one_minus_cv
    
    def finalize(self):
        self.final_cv = self.cv
        self.final_one_minus_cv = self.one_minus_cv

    def reset(self):
        self.positions.clear()
        self.vel_history.clear()
        self.sway_velocity = 0.0
        self.cv = 0.0

        # IMPORTANT: reset EMA state
        if hasattr(self, "vel_ema"):
            del self.vel_ema

# ============================================================
# SLS TRACKER
# ============================================================
class SLSDetector:
    def __init__(self, alpha_theta=0.9):
        self.alpha_theta = alpha_theta

        self.theta_ema = None
        self.theta_history = []

        self.knee_y_history = []

        self.peak_theta = 0.0

    def update(self, hip, knee, ankle):
        if hip is None or knee is None or ankle is None:
            return

        h = np.array(hip, dtype=float)
        k = np.array(knee, dtype=float)
        a = np.array(ankle, dtype=float)

        if h.shape[0] < 3:
            return

        # NEW: Handle 3D global data (meters) where Y increases upwards
        # Heuristic: pixels are hundreds, 3D meters are small (< 5.0)
        dist_h_k = np.linalg.norm(h - k)
        if dist_h_k < 5.0:
            h[1] = -h[1]
            k[1] = -k[1]
            a[1] = -a[1]

        # ----------------------------
        # vectors
        # ----------------------------
        vec1 = h - k
        vec2 = a - k

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            return

        cos_theta = np.dot(vec1, vec2) / (norm1 * norm2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        theta = 180.0 - np.degrees(np.arccos(cos_theta))

        # ----------------------------
        # EMA smoothing
        # ----------------------------
        if self.theta_ema is None:
            self.theta_ema = theta
        else:
            self.theta_ema = (
                self.alpha_theta * self.theta_ema +
                (1 - self.alpha_theta) * theta
            )

        self.theta_history.append(self.theta_ema)

        # ----------------------------
        # track knee depth
        # ----------------------------
        knee_y = k[1]
        self.knee_y_history.append(knee_y)

        # ----------------------------
        # detect deepest squat frame
        # ----------------------------
        idx = np.argmax(self.knee_y_history)

        self.peak_theta = self.theta_history[idx]

    def get_fppa(self):
        return self.peak_theta

    def reset(self):
        self.theta_ema = None
        self.theta_history.clear()
        self.knee_y_history.clear()
        self.peak_theta = 0.0


# ============================================================
# CMJ TRACKER
# ============================================================


class CMJTracker:
    def __init__(self, fps=60, alpha_v=0.8):
        self.fps = fps
        self.alpha_v = alpha_v
        
        # State tracking
        self.phase = "idle"
        self.frame_count = 0
        
        # Physical Baselines
        self.baseline_hip_y = None
        self.baseline_ankle_y = None
        self.torso_size = None
        self.baseline_samples = []

        # Previous positions for velocity
        self.prev_hip_y = None
        self.hip_vy_ema = 0.0

        # History for refinement
        self.y_hip_history = []
        self.t_history = []

        # Time markers
        self.t_start_dip = None    # Start of downward movement
        self.t_takeoff = None      # Feet leave ground
        self.t_landing = None      # Feet touch ground

    def update(self, hip_y, ankle_y, torso_size):
        """
        hip_y: Average of left and right hip Y
        ankle_y: Average of left and right ankle Y
        torso_size: Shoulder-to-Hip distance for normalizing thresholds
        """
        if hip_y is None or ankle_y is None: return
        
        # NEW: Handle 3D global data (meters) where Y increases upwards
        # Check against torso_size (pixels are 100+, meters are < 1.0)
        if torso_size is not None and torso_size < 5.0:
            hip_y = -hip_y
            ankle_y = -ankle_y

        t = self.frame_count / self.fps
        self.frame_count += 1
        self.torso_size = torso_size

        # 1. Capture Initial Standing Baseline (First 10 frames)
        if len(self.baseline_samples) < 10:
            self.baseline_samples.append((hip_y, ankle_y))
            if len(self.baseline_samples) == 10:
                self.baseline_hip_y = np.mean([s[0] for s in self.baseline_samples])
                self.baseline_ankle_y = np.mean([s[1] for s in self.baseline_samples])
            return

        # 2. Calculate Hip Velocity (Down is positive Y in image coords)
        if self.prev_hip_y is not None:
            raw_vy = (hip_y - self.prev_hip_y) * self.fps
            self.hip_vy_ema = (self.alpha_v * self.hip_vy_ema) + (1 - self.alpha_v) * raw_vy
        self.prev_hip_y = hip_y

        self.y_hip_history.append(hip_y)
        self.t_history.append(t)

        # 3. PHASE DETECTION
        
        # IDLE -> CONTRACTION (Hips move down > 5% of torso size per second)
        if self.phase == "idle" and self.hip_vy_ema > (self.torso_size * 0.5):
            self.phase = "contraction"
            self.t_start_dip = t

        # CONTRACTION -> FLIGHT (Ankles leave baseline height)
        # We use a 5% torso buffer to avoid noise (was 10%)
        elif self.phase == "contraction" and ankle_y < (self.baseline_ankle_y - self.torso_size * 0.05):
            self.phase = "flight"
            self.t_takeoff = t

        # FLIGHT -> LANDED (Ankles return to baseline)
        elif self.phase == "flight" and ankle_y >= (self.baseline_ankle_y - self.torso_size * 0.05):
            self.phase = "landed"
            self.t_landing = t

    def get_jump_results(self):
        """
        Returns scientifically robust metrics:
        Height (m), RSImod, and Time to Takeoff
        """
        if not self.t_start_dip or not self.t_takeoff:
            return None

        # Handle partial results if landing hasn't been reached yet
        if not self.t_landing:
            t_contraction = self.t_takeoff - self.t_start_dip
            return {
                "height_m": 0.0,
                "rsi_mod": 0.0,
                "t_flight": 0.0,
                "t_takeoff_phase": t_contraction,
                "status": "in_flight"
            }

        # 1. Air Time Calculation
        # We use the refined flight time if possible
        t_flight = self.get_refined_flight_time()
        
        # 2. Jump Height (Bosco Formula: h = (g * t^2) / 8)
        # Source: Bosco et al. (1983)
        g = 9.81
        height = (g * (t_flight**2)) / 8
        
        # 3. PHYSICAL SANITY CHECK
        # Max standing jump world record is ~1.2m. 
        # Anything above 1.5m is definitely a measurement error (e.g. FPS mismatch).
        if height > 1.5:
            # Fallback to a simpler distance-based estimate or cap it
            # We cap it at 1.5m and log the issue if we had a logger
            height = min(height, 1.5)
        
        # 4. Time to Takeoff (Contraction Phase)
        t_contraction = self.t_takeoff - self.t_start_dip
        
        # 4. RSImod (Reactive Strength Index Modified)
        # Source: Ebben & Petushek (2010)
        rsi_mod = height / t_contraction if t_contraction > 0 else 0
        
        return {
            "height_m": height,
            "rsi_mod": rsi_mod,
            "t_flight": t_flight,
            "t_takeoff_phase": t_contraction
        }

    def get_refined_flight_time(self):
        """
        Fits a parabola to the HIP Y-coordinates during flight.
        The hip is a better proxy for Center of Mass than the ankle.
        """
        t = np.array(self.t_history)
        y = np.array(self.y_hip_history)
        mask = (t >= self.t_takeoff) & (t <= self.t_landing)
        
        t_f, y_f = t[mask], y[mask]
        if len(t_f) < 5: return self.t_landing - self.t_takeoff

        # Fit parabola: y = at^2 + bt + c
        # Use a try-except block for polyfit in case of degenerate data
        try:
            a, b, c = np.polyfit(t_f, y_f, 2)
            
            # Physics check: 'a' must be negative for a downward-opening parabola (Y is down)
            # Actually, in image coords, Y increases downwards, so a jump (up then down) 
            # is a concave parabola (a > 0).
            # If torso_size < 5, we flipped signs, making jump 'concave' (a > 0) in the math.
            
            # Solve for roots where hip returns to its takeoff height
            y_takeoff = y_f[0]
            roots = np.roots([a, b, c - y_takeoff])
            
            if len(roots) == 2:
                t_roots = np.real(roots[np.isreal(roots)])
                if len(t_roots) == 2:
                    t_roots = np.sort(t_roots)
                    refined_t = float(t_roots[1] - t_roots[0])
                    # Sanity check: Refined time should be similar to raw time
                    raw_t = self.t_landing - self.t_takeoff
                    if 0.5 * raw_t < refined_t < 1.5 * raw_t:
                        return refined_t
        except:
            pass
        
        return self.t_landing - self.t_takeoff

    def get_condition_score(self, current_height, baseline_height):
        """
        Maps performance drop to a 1-10 scale based on 
        the Coefficient of Variation logic in Claudino et al. (2017).
        """
        ratio = current_height / baseline_height
        
        if ratio >= 0.98: return 10  # Optimal
        if ratio >= 0.95: return 8   # Mild Fatigue (within typical CV)
        if ratio >= 0.90: return 6   # Significant Fatigue (Moderate Effect Size)
        if ratio >= 0.85: return 4   # High Suppression (Large Effect Size)
        return 2                     # Severe Overload

    def reset(self):
        self.__init__(self.fps, self.alpha_v)

    def get_rsi(self):
        """Compatibility method for the real-time UI."""
        results = self.get_jump_results()
        if results and "rsi_mod" in results:
            return results["rsi_mod"]
        return 0.0


# ============================================================
# FPPA CALCULATION
# ============================================================
def calculate_fppa(joints):
    """
    Frontal Plane Projection Angle
    Uses left leg:
        Hip = 11
        Knee = 13
        Ankle = 15
    """

    try:
        hip = joints[11]
        knee = joints[13]
        ankle = joints[15]

        a = np.array(hip[:2]) - np.array(knee[:2])
        b = np.array(ankle[:2]) - np.array(knee[:2])

        dot = np.dot(a, b)

        angle = np.arccos(
            np.clip(
                dot / (np.linalg.norm(a) * np.linalg.norm(b)),
                -1.0,
                1.0
            )
        )

        return 180 - np.degrees(angle)

    except:
        return None


# ============================================================
# JUMP HEIGHT CALCULATION
# ============================================================
def calculate_jump_height(joints, baseline_feet_y):
    """
    Estimate jump height using ankle vertical displacement.
    """

    try:
        left_ankle = joints[15]
        right_ankle = joints[16]

        current_feet_y = (left_ankle[1] + right_ankle[1]) / 2

        jump_height = baseline_feet_y - current_feet_y

        return max(0, jump_height)

    except:
        return None

# ============================================================
# FEATURE EXTRACTION FROM TRACKER OUTPUT
# ============================================================
def extract_features(tracker_output, baseline_feet_y=None, prev_mid_hip=None, prev_mid_shoulder=None):
    """
    Accepts tracker dictionary:

    {
        "joints_3d_global": ...,
        "joints_3d_incam": ...,
        "K_fullimg": ...
    }

    Returns calculated biomechanical features.
    """

    features = {}

    if tracker_output is None:
        return features

    joints = tracker_output.get("joints_3d_global")

    if joints is None:
        return features

    joints = np.array(joints)

    # try:

    #     left_hip = joints[11]
    #     right_hip = joints[12]

    #     alpha = 1

    #     raw_mid_hip = np.array([
    #         (left_hip[0] + right_hip[0]) / 2,
    #         (left_hip[1] + right_hip[1]) / 2,
    #         (left_hip[2] + right_hip[2]) / 2,
    #     ], dtype=float)

    #     # ---------------- SMOOTHING ----------------
    #     if prev_mid_hip is None:
    #         mid_hip = raw_mid_hip
    #     else:
    #         mid_hip = alpha * raw_mid_hip + (1 - alpha) * prev_mid_hip

    #     features["mid_hip"] = tuple(mid_hip)

    #     # ---------------- FPPA ----------------
    #     features["sls_fppa"] = calculate_fppa(joints)

    #     # ---------------- JUMP HEIGHT ----------------
    #     if baseline_feet_y is not None:
    #         features["jump_feet"] = calculate_jump_height(
    #             joints,
    #             baseline_feet_y
    #         )
    #     else:
    #         features["jump_feet"] = None

    # except:
    #     features["mid_hip"] = None
    #     features["sls_fppa"] = None
    #     features["jump_feet"] = None

    # return features

    try:
        left_hip = joints[11]
        right_hip = joints[12]

        left_shoulder = joints[5]
        right_shoulder = joints[6]

        raw_mid_hip = np.array([
            (left_hip[0] + right_hip[0]) / 2,
            (left_hip[1] + right_hip[1]) / 2,
            (left_hip[2] + right_hip[2]) / 2,
        ], dtype=float)

        raw_mid_shoulder = np.array([
            (left_shoulder[0] + right_shoulder[0]) / 2,
            (left_shoulder[1] + right_shoulder[1]) / 2,
            (left_shoulder[2] + right_shoulder[2]) / 2,
        ], dtype=float)

        alpha = 1

        # ---- smoothing ----
        if prev_mid_hip is None:
            mid_hip = raw_mid_hip
        else:
            mid_hip = alpha * raw_mid_hip + (1 - alpha) * prev_mid_hip

        if prev_mid_shoulder is None:
            mid_shoulder = raw_mid_shoulder
        else:
            mid_shoulder = alpha * raw_mid_shoulder + (1 - alpha) * prev_mid_shoulder

        features["mid_hip"] = tuple(mid_hip)
        features["mid_shoulder"] = tuple(mid_shoulder)

        # ---- other features ----
        features["sls_fppa"] = calculate_fppa(joints)

        if baseline_feet_y is not None:
            features["jump_feet"] = calculate_jump_height(
                joints,
                baseline_feet_y
            )
        else:
            features["jump_feet"] = None

    except:
        features["mid_hip"] = None
        features["mid_shoulder"] = None
        features["sls_fppa"] = None
        features["jump_feet"] = None

    return features


# ============================================================
# R2P SCORER
# ============================================================
class R2PScorer:
    def compute(self, cv=None, fppa=None, delta_rsi=None):
        scores = []

        # BALANCE SCORE
        if cv is not None:
            if cv <= 10:
                scores.append(0)

            elif cv <= 20:
                scores.append((cv - 10) / 10)

            else:
                scores.append(1)

        # SLS SCORE
        if fppa is not None:
            if fppa <= 7:
                scores.append(0)

            elif fppa <= 10:
                scores.append((fppa - 7) / 3)

            else:
                scores.append(1)

        # CMJ SCORE
        if delta_rsi is not None:
            if delta_rsi < 5:
                scores.append(0)

            elif delta_rsi <= 8:
                scores.append((delta_rsi - 5) / 3)

            else:
                scores.append(1)

        if not scores:
            return None, "Detecting"

        total = sum(scores) / len(scores)

        if total <= 0.33:
            return total, "GREEN"

        elif total <= 0.66:
            return total, "YELLOW"

        else:
            return total, "RED"