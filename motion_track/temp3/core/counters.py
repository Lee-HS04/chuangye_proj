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

# class SwayTracker:
#     def __init__(self):
#         self.history = []

#     def update(self, mid_hip):
#         if mid_hip is not None:
#             self.history.append(mid_hip)
#             if len(self.history) > 30:
#                 self.history.pop(0)

#     def get_sway_velocity(self):
#         if len(self.history)<2:
#             return None
#         diffs = [np.linalg.norm(np.array(self.history[i+1])-np.array(self.history[i])) for i in range(len(self.history)-1)]
#         return float(np.mean(diffs))

class SwayTracker:
    def __init__(self, window_size=50): #=50
        """
        window_size: number of frames to consider for CV
        """
        self.positions = deque(maxlen=window_size)
        self.sway_velocity = 0
        self.cv = 0

    # def update(self, mid_hip):
    #     """
    #     mid_hip: tuple (x, y)
    #     """
    #     if mid_hip is None:
    #         return

    #     # Add new position
    #     self.positions.append(mid_hip)

    #     # Compute frame-to-frame displacements
    #     if len(self.positions) > 1:
    #         displacements = [
    #             np.linalg.norm(np.array(p2) - np.array(p1))
    #             for p1, p2 in zip(self.positions[:-1], self.positions[1:])
    #         ]
    #         self.sway_velocity = np.mean(displacements)
    #         mean_disp = np.mean(displacements)
    #         std_disp = np.std(displacements)
    #         # CV calculation
    #         self.cv = (std_disp / mean_disp * 100) if mean_disp > 0 else 0
    #         # Clamp CV to prevent extreme spikes
    #         self.cv = min(self.cv, 50)
    #     else:
    #         self.sway_velocity = 0
    #         self.cv = 0

    # def get_sway_velocity(self):
    #     return self.sway_velocity

    # def get_cv(self):
    #     return self.cv
    

#--------------------------------------------------------------------------------------


# class SwayTracker:
#     def __init__(self, max_len=50):
#         self.positions = deque(maxlen=max_len)
#         self.sway_velocity = 0.0
#         self.cv = 0.0

    def update(self, mid_hip):
        if mid_hip is None:
            return

        mid_hip = tuple(float(v) for v in mid_hip)
        self.positions.append(mid_hip)

        if len(self.positions) > 1:
            pos_list = list(self.positions)
            displacements = [
                np.linalg.norm(np.array(pos_list[i+1]) - np.array(pos_list[i]))
                for i in range(len(pos_list) - 1)
            ]
            self.sway_velocity = np.mean(displacements)
            mean_disp = np.mean(displacements)
            std_disp = np.std(displacements)
            self.cv = (std_disp / mean_disp * 1) if mean_disp > 0 else 0 #std_disp / mean_disp * 100
            #self.cv = min(self.cv, 50)
        else:
            self.sway_velocity = 0
            self.cv = 0

    # Add this getter method
    def get_sway_velocity(self):
        return self.sway_velocity

    def get_cv(self):
        return self.cv