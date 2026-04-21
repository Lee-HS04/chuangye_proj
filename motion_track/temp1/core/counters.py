import math
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
        angle = features.get(self.feature)
        if angle is None:
            return self.reps
        if self.state == "up" and angle < self.min_angle:
            self.state = "down"
        elif self.state == "down" and angle > self.max_angle:
            self.state = "up"
            self.reps += 1
        return self.reps

class SwayTracker:
    def __init__(self, window=5):
        self.window = window
        self.positions = deque(maxlen=window)

    def update(self, pos):
        if pos is not None:
            self.positions.append(pos)

    def get_sway_velocity(self):
        if len(self.positions) < 2:
            return None
        dx = self.positions[-1][0] - self.positions[0][0]
        dy = self.positions[-1][1] - self.positions[0][1]
        dt = len(self.positions)
        return math.sqrt(dx**2 + dy**2)/dt