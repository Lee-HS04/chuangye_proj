# # core/counters.py
# import numpy as np
# from collections import deque

# class RepCounter:
#     def __init__(self, exercise, feature, min_angle, max_angle):
#         self.exercise = exercise
#         self.feature = feature
#         self.min_angle = min_angle
#         self.max_angle = max_angle
#         self.reps = 0
#         self.state = "up"

#     def update(self, features):
#         val = features.get(self.feature)
#         if val is None:
#             return self.reps
#         if val <= self.min_angle and self.state=="up":
#             self.state="down"
#         elif val >= self.max_angle and self.state=="down":
#             self.state="up"
#             self.reps += 1
#         return self.reps


# class SwayTracker:
#     def __init__(self, window_size=50, fps=30):
#         self.positions = deque(maxlen=window_size)
#         self.sway_velocity = 0
#         self.cv = 0
#         self.fps = fps   # important

#     def update(self, mid_hip):
#         if mid_hip is None:
#             return

#         mid_hip = tuple(float(v) for v in mid_hip)
#         self.positions.append(mid_hip)

#         if len(self.positions) > 1:
#             pos_list = list(self.positions)
#             # velocity in pixels/sec
#             displacements = [
#                 np.linalg.norm(np.array(pos_list[i+1]) - np.array(pos_list[i])) * self.fps
#                 for i in range(len(pos_list) - 1)
#             ]
#             self.sway_velocity = np.mean(displacements)
#             mean_disp = np.mean(displacements)
#             std_disp = np.std(displacements)
#             self.cv = (std_disp / mean_disp * 100) if mean_disp > 0 else 0
#         else:
#             self.sway_velocity = 0
#             self.cv = 0

#     # ─── Add these getters ───
#     def get_sway_velocity(self):
#         return self.sway_velocity

#     def get_cv(self):
#         return self.cv

# def calculate_fppa(keypoints):
#     try:
#         hip = keypoints[11]
#         knee = keypoints[13]
#         ankle = keypoints[15]

#         a = np.array([hip[0]-knee[0], hip[1]-knee[1]])
#         b = np.array([ankle[0]-knee[0], ankle[1]-knee[1]])

#         dot = np.dot(a, b)
#         angle = np.arccos(np.clip(dot / (np.linalg.norm(a)*np.linalg.norm(b)), -1.0, 1.0))

#         return 180 - np.degrees(angle)
#     except:
#         return None
    
# class CMJTracker:
#     def __init__(self, fps=30):
#         self.fps = fps
#         self.jump_start = None
#         self.delta_RSI = None
#         self.baseline = 0.7  # adjustable

#     def update(self, jump_height, frame_index):
#         if jump_height is None:
#             return None

#         # Detect takeoff
#         if jump_height > 20 and self.jump_start is None:
#             self.jump_start = frame_index

#         # Detect landing
#         elif jump_height < 5 and self.jump_start is not None:
#             end = frame_index

#             T_flight = (end - self.jump_start) / self.fps
#             T_contraction = 0.4  # can improve later

#             RSI = T_flight / T_contraction

#             self.delta_RSI = max(0, (self.baseline - RSI) / self.baseline * 100)

#             self.jump_start = None

#         return self.delta_RSI


# def extract_features(keypoints_3d):
#     """
#     Extract numerical features from 3D keypoints for scoring.
#     Returns a dict:
#       sway_cv, jump_feet, fppa, etc.
#     """
#     features = {}

#     # Example: mid_hip for sway
#     if keypoints_3d is not None and len(keypoints_3d) > 11:
#         left_hip = keypoints_3d[11]
#         right_hip = keypoints_3d[12]
#         if left_hip is not None and right_hip is not None:
#             features["mid_hip"] = (
#                 (left_hip[0] + right_hip[0]) / 2,
#                 (left_hip[1] + right_hip[1]) / 2,
#                 (left_hip[2] + right_hip[2]) / 2,
#             )

#     # Placeholder for jump height (CMJ)
#     features["jump_feet"] = None  # replace with actual calc from keypoints

#     # Placeholder for FPPA (SLS)
#     features["sls_fppa"] = None  # replace with actual calc from keypoints

#     return features

# class R2PScorer:
#     def compute(self, cv=None, fppa=None, delta_rsi=None):
#         scores = []

#         # Balance
#         if cv is not None:
#             if cv <= 10:
#                 scores.append(0)
#             elif cv <= 20:
#                 scores.append((cv - 10) / 10)
#             else:
#                 scores.append(1)

#         # SLS
#         if fppa is not None:
#             if fppa <= 7:
#                 scores.append(0)
#             elif fppa <= 10:
#                 scores.append((fppa - 7) / 3)
#             else:
#                 scores.append(1)

#         # CMJ
#         if delta_rsi is not None:
#             if delta_rsi < 5:
#                 scores.append(0)
#             elif delta_rsi <= 8:
#                 scores.append((delta_rsi - 5) / 3)
#             else:
#                 scores.append(1)

#         if not scores:
#             return None, "Detecting"

#         total = sum(scores) / len(scores)

#         if total <= 0.33:
#             return total, "GREEN"
#         elif total <= 0.66:
#             return total, "YELLOW"
#         else:
#             return total, "RED"


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
# CMJ TRACKER
# ============================================================

class CMJTracker:
    def __init__(self, fps=60, alpha_v=0.8, threshold=0.02):
        self.fps = fps
        self.alpha_v = alpha_v
        self.threshold = threshold

        self.prev_y = None
        self.vy_ema = None

        self.y_history = []
        self.t_history = []

        self.phase = "idle"

        self.start_time = None
        self.takeoff_time = None
        self.landing_time = None

        self.frame_count = 0

    def update(self, ankle_y):
        if ankle_y is None:
            return

        t = self.frame_count / self.fps
        self.frame_count += 1

        self.y_history.append(ankle_y)
        self.t_history.append(t)

        if self.prev_y is None:
            self.prev_y = ankle_y
            return

        # ----------------------------
        # vertical velocity
        # ----------------------------
        vy = (ankle_y - self.prev_y) * self.fps
        self.prev_y = ankle_y

        # ----------------------------
        # EMA smoothing
        # ----------------------------
        if self.vy_ema is None:
            self.vy_ema = vy
        else:
            self.vy_ema = (
                self.alpha_v * self.vy_ema +
                (1 - self.alpha_v) * vy
            )

        # ----------------------------
        # PHASE DETECTION
        # ----------------------------

        # start contraction (downward movement)
        if self.phase == "idle" and self.vy_ema > self.threshold:
            self.phase = "contraction"
            self.start_time = t

        # takeoff (velocity flips strongly upward)
        elif self.phase == "contraction" and self.vy_ema < -self.threshold:
            self.phase = "flight"
            self.takeoff_time = t

        # landing (velocity returns positive)
        elif self.phase == "flight" and self.vy_ema > self.threshold:
            self.phase = "landed"
            self.landing_time = t

    # ----------------------------
    # RAW RSI
    # ----------------------------
    def get_rsi(self):
        if self.start_time is None or self.takeoff_time is None or self.landing_time is None:
            return 0.0

        T_contraction = self.takeoff_time - self.start_time
        T_flight = self.landing_time - self.takeoff_time

        if T_contraction <= 0:
            return 0.0

        return T_flight / T_contraction

    # def get_flight_time(self):
    #     return self.T_flight
    
    # def get_contraction_time(self):
    #     return self.T_contraction


    # ----------------------------
    # PARABOLA FIT (REFINED)
    # ----------------------------
    def get_refined_flight_time(self):
        if self.takeoff_time is None or self.landing_time is None:
            return 0.0

        # select flight phase data
        t = np.array(self.t_history)
        y = np.array(self.y_history)

        mask = (t >= self.takeoff_time) & (t <= self.landing_time)

        t_f = t[mask]
        y_f = y[mask]

        if len(t_f) < 5:
            return self.landing_time - self.takeoff_time

        # fit parabola y = at^2 + bt + c
        coeffs = np.polyfit(t_f, y_f, 2)
        a, b, c = coeffs

        # solve y = ground level
        y_ground = y_f[0]

        roots = np.roots([a, b, c - y_ground])

        if len(roots) != 2:
            return self.landing_time - self.takeoff_time

        t1, t2 = np.sort(roots)

        return float(t2 - t1)

    def reset(self):
        self.prev_y = None
        self.vy_ema = None

        self.y_history.clear()
        self.t_history.clear()

        self.phase = "idle"

        self.start_time = None
        self.takeoff_time = None
        self.landing_time = None

        self.frame_count = 0



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