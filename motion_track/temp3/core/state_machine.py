import os
import math
import numpy as np

class StateMachineFSM:
    def __init__(self, exercise_name="CMJ"):
        self.exercise_name = exercise_name
        self.current_state = 1
        self.state_history = []
        self.good_frames = 0
        
    def reset(self):
        self.current_state = 1
        self.state_history = []
        self.good_frames = 0

    def compute_angle(self, p1, p2, p3):
        """ Calculate angle between 3 points (p2 is the vertex) """
        if None in (p1, p2, p3):
            return None
        
        # vectors
        v1 = (p1[0]-p2[0], p1[1]-p2[1])
        v2 = (p3[0]-p2[0], p3[1]-p2[1])
        
        # calculate dot product and magnitudes
        dot_prod = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 * mag2 == 0:
            return None
            
        cos_angle = max(min(dot_prod / (mag1 * mag2), 1.0), -1.0)
        angle = math.degrees(math.acos(cos_angle))
        return angle
    
    def get_pixel_distance(self, p1, p2):
        if p1 is None or p2 is None:
            return None
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def is_near(self, p1, p2, reference_dist, threshold=0.5):
        """
        p1, p2: The two points to compare.
        reference_dist: A known length (e.g., shoulder-to-hip distance).
        threshold: What % of the reference distance counts as 'near'.
                   0.5 means 'if distance is less than 20% of torso length'.
        """
        if p1 is None or p2 is None or reference_dist is None:
            return False
        pixel_dist = self.get_pixel_distance(p1, p2)
    
        # Is the distance small compared to the body size. For now we consider the a distance of 1/2 the torso between hip and wrist joint as near enough
        return pixel_dist < (reference_dist * threshold)

    def process_frame(self, keypoints_2d):
        """
        Process a single frame's 2D keypoints through the State Machine.
        Returns: tuple (status, message, current_state)
        """
        # We need specific keypoints based on COCO17 model
        # right hip: 12, right knee: 14, right ankle: 16
        # left hip: 11, left knee: 13, left ankle: 15
        # shoulder: 5 (left) / 6 (right)
        
        if len(keypoints_2d) < 17:
            return "BAD", "Waiting for body detection", self.current_state
            
        r_hip = keypoints_2d[12]
        r_knee = keypoints_2d[14]
        r_ankle = keypoints_2d[16]
        
        l_hip = keypoints_2d[11]
        l_knee = keypoints_2d[13]
        l_ankle = keypoints_2d[15]
        
        r_knee_angle = self.compute_angle(r_hip, r_knee, r_ankle)
        l_knee_angle = self.compute_angle(l_hip, l_knee, l_ankle)
        
        r_shoulder = keypoints_2d[6]
        r_elbow = keypoints_2d[8]
        r_wrist = keypoints_2d[10]
        
        l_shoulder = keypoints_2d[5]
        l_elbow = keypoints_2d[7]
        l_wrist = keypoints_2d[9]
        
        r_elbow_angle = self.compute_angle(r_shoulder, r_elbow, r_wrist)
        l_elbow_angle = self.compute_angle(l_shoulder, l_elbow, l_wrist)
        
        torso_size = min(self.get_pixel_distance(r_hip, r_shoulder), self.get_pixel_distance(l_hip,l_shoulder))
        
        if r_knee_angle is None or l_knee_angle is None:
            return "BAD", "Legs not visible", self.current_state
            
        # Use average knee angle or lowest knee angle
        knee_angle = min(r_knee_angle, l_knee_angle)
        
        #calculate distance between ankles to check if one foot is in the air
        ankle_distance = abs(l_ankle[1]-r_ankle[1])
        
        if self.exercise_name == "CMJ":
            # State 1: Standing (Knee > 160)
            # State 2: Squat (Knee < 120)
            # State 3: Jump/Extend (Knee > 160)
            
            if self.current_state == 1:
                if knee_angle > 150:
                    self.good_frames += 1
                    if self.good_frames > 5: # Stable standing for a bit
                        return "GOOD", "Ready. Now Squat.", self.current_state
                if knee_angle < 130 and self.good_frames > 5:
                    self.current_state = 2
                    self.state_history.append(1)
                    return "GOOD", "Squatting... Go deeper.", self.current_state
                return "BAD", "Stand straight to begin.", self.current_state
                    
            elif self.current_state == 2:
                # Need to hit a not too deep squat before jumping
                if knee_angle > 100 and knee_angle < 150:
                    return "GOOD", "Good depth. JUMP!", self.current_state
                # Check for jump
                if knee_angle > 140:
                    if len(self.state_history) > 0 and self.state_history[-1] == 1:
                        self.current_state = 3
                        self.state_history.append(2)
                        return "GOOD", "Jumping!", self.current_state
                    else:
                        # Bad form jump
                        self.reset()
                        return "BAD", "Invalid squat depth. Try again.", self.current_state
                return "GOOD", "Go deeper.", self.current_state
                
            elif self.current_state == 3:
                if knee_angle > 150:
                    self.state_history.append(3)
                    return "REP_COMPLETE", "Perfect Rep!", self.current_state
                return "GOOD", "Finish jump.", self.current_state

        elif self.exercise_name == "Balance":
            # State 1: Standing straight
            # State 2: Lift one leg (Balance) -> Hold for 5 seconds
            
            if None in [torso_size, r_elbow_angle, l_elbow_angle]:
                return "BAD", "Body not fully visible", self.current_state
            
            if self.current_state == 1:
                if ankle_distance>torso_size*0.6 and r_elbow_angle<120 and r_elbow_angle>30 and l_elbow_angle<120 and l_elbow_angle>30 and self.is_near(r_wrist,r_hip, torso_size):
                    self.current_state = 2
                    self.good_frames = 0
                    return "GOOD", "Balancing. Hold it!", self.current_state
                elif knee_angle > 160 and r_elbow_angle<120 and r_elbow_angle>30 and l_elbow_angle<120 and l_elbow_angle>30 and self.is_near(r_wrist,r_hip, torso_size) :
                    self.good_frames += 1
                    if self.good_frames > 5:
                        return "GOOD", "Ready. Lift one leg.", self.current_state
                return "BAD", "Stand straight and hands on hips to begin.", self.current_state
                
            elif self.current_state == 2:
                akimbo = False
                if r_elbow_angle<120 and r_elbow_angle>30 and l_elbow_angle<120 and l_elbow_angle>30 and self.is_near(r_wrist,r_hip, torso_size):
                    akimbo = True
                if akimbo and ankle_distance>torso_size*0.6:
                    self.good_frames += 1
                    if self.good_frames >= 50: # roughly 5 seconds at 10fps
                        return "REP_COMPLETE", "Balance Complete!", self.current_state
                    return "GOOD", f"Holding... {self.good_frames/10:.1f}s", self.current_state
                else:
                    self.reset()
                    return "BAD", "Leg dropped! Reset FSM.", self.current_state

        return "BAD", "Invalid Exercise", self.current_state
