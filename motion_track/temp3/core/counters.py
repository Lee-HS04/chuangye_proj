# core/counters.py
import numpy as np
from collections import deque

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
        if val <= self.min_angle and self.state=="up":
            self.state="down"
        elif val >= self.max_angle and self.state=="down":
            self.state="up"
            self.reps += 1
        return self.reps


class SwayTracker:
    def __init__(self, window_size=50, fps=30):
        self.positions = deque(maxlen=window_size)
        self.sway_velocity = 0
        self.cv = 0
        self.fps = fps   # important

    def update(self, mid_hip):
        if mid_hip is None:
            return

        mid_hip = tuple(float(v) for v in mid_hip)
        self.positions.append(mid_hip)

        if len(self.positions) > 1:
            pos_list = list(self.positions)
            # velocity in pixels/sec
            displacements = [
                np.linalg.norm(np.array(pos_list[i+1]) - np.array(pos_list[i])) * self.fps
                for i in range(len(pos_list) - 1)
            ]
            self.sway_velocity = np.mean(displacements)
            mean_disp = np.mean(displacements)
            std_disp = np.std(displacements)
            self.cv = (std_disp / mean_disp * 100) if mean_disp > 0 else 0
        else:
            self.sway_velocity = 0
            self.cv = 0

    # ─── Add these getters ───
    def get_sway_velocity(self):
        return self.sway_velocity

    def get_cv(self):
        return self.cv

def calculate_fppa(keypoints):
    try:
        hip = keypoints[11]
        knee = keypoints[13]
        ankle = keypoints[15]

        a = np.array([hip[0]-knee[0], hip[1]-knee[1]])
        b = np.array([ankle[0]-knee[0], ankle[1]-knee[1]])

        dot = np.dot(a, b)
        angle = np.arccos(np.clip(dot / (np.linalg.norm(a)*np.linalg.norm(b)), -1.0, 1.0))

        return 180 - np.degrees(angle)
    except:
        return None
    
class CMJTracker:
    def __init__(self, fps=30):
        self.fps = fps
        self.jump_start = None
        self.delta_RSI = None
        self.baseline = 0.7  # adjustable

    def update(self, jump_height, frame_index):
        if jump_height is None:
            return None

        # Detect takeoff
        if jump_height > 20 and self.jump_start is None:
            self.jump_start = frame_index

        # Detect landing
        elif jump_height < 5 and self.jump_start is not None:
            end = frame_index

            T_flight = (end - self.jump_start) / self.fps
            T_contraction = 0.4  # can improve later

            RSI = T_flight / T_contraction

            self.delta_RSI = max(0, (self.baseline - RSI) / self.baseline * 100)

            self.jump_start = None

        return self.delta_RSI


def extract_features(keypoints_3d):
    """
    Extract numerical features from 3D keypoints for scoring.
    Returns a dict:
      sway_cv, jump_feet, fppa, etc.
    """
    features = {}

    # Example: mid_hip for sway
    if keypoints_3d is not None and len(keypoints_3d) > 11:
        left_hip = keypoints_3d[11]
        right_hip = keypoints_3d[12]
        if left_hip is not None and right_hip is not None:
            features["mid_hip"] = (
                (left_hip[0] + right_hip[0]) / 2,
                (left_hip[1] + right_hip[1]) / 2,
                (left_hip[2] + right_hip[2]) / 2,
            )

    # Placeholder for jump height (CMJ)
    features["jump_feet"] = None  # replace with actual calc from keypoints

    # Placeholder for FPPA (SLS)
    features["sls_fppa"] = None  # replace with actual calc from keypoints

    return features

class R2PScorer:
    def compute(self, cv=None, fppa=None, delta_rsi=None):
        scores = []

        # Balance
        if cv is not None:
            if cv <= 10:
                scores.append(0)
            elif cv <= 20:
                scores.append((cv - 10) / 10)
            else:
                scores.append(1)

        # SLS
        if fppa is not None:
            if fppa <= 7:
                scores.append(0)
            elif fppa <= 10:
                scores.append((fppa - 7) / 3)
            else:
                scores.append(1)

        # CMJ
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

