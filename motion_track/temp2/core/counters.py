# core/counters.py
import numpy as np

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
    def __init__(self):
        self.history = []

    def update(self, mid_hip):
        if mid_hip is not None:
            self.history.append(mid_hip)
            if len(self.history) > 30:
                self.history.pop(0)

    def get_sway_velocity(self):
        if len(self.history)<2:
            return None
        diffs = [np.linalg.norm(np.array(self.history[i+1])-np.array(self.history[i])) for i in range(len(self.history)-1)]
        return float(np.mean(diffs))