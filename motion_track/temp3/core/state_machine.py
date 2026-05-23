import os
import math
import numpy as np

class StateMachineFSM:
    def __init__(self, exercise_name="CMJ"):
        self.exercise_name = exercise_name
        self.current_state = 1
        self.state_history = []
        self.good_frames = 0
        
        # Baselines for landing detection
        self.baseline_ankle_y = None
        self.baseline_samples = []
        self.is_3d_engine = False
        
    def reset(self):
        self.current_state = 1
        self.state_history = []
        self.good_frames = 0
        self.baseline_samples = []
        self.baseline_ankle_y = None

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
        
        # Determine torso size safely (handle missing keypoints)
        r_torso = self.get_pixel_distance(r_hip, r_shoulder)
        l_torso = self.get_pixel_distance(l_hip, l_shoulder)
        
        valid_dists = [d for d in [r_torso, l_torso] if d is not None]
        if not valid_dists:
            return "BAD", "Body not fully visible", self.current_state
            
        torso_size = min(valid_dists)
        
        # Handle 3D engine vs 2D pixels (Sign flip for Y)
        curr_l_ankle_y = l_ankle[1]
        curr_r_ankle_y = r_ankle[1]
        if torso_size is not None and torso_size < 5.0:
            curr_l_ankle_y = -curr_l_ankle_y
            curr_r_ankle_y = -curr_r_ankle_y
        avg_ankle_y = (curr_l_ankle_y + curr_r_ankle_y) / 2.0

        if r_knee_angle is None or l_knee_angle is None:
            return "BAD", "Legs not visible", self.current_state
            
        # Use average knee angle or lowest knee angle
        knee_angle = min(r_knee_angle, l_knee_angle)
        
        #calculate distance between ankles to check if one foot is in the air
        ankle_distance = abs(l_ankle[1]-r_ankle[1])
        
        max_angle = 150
        min_angle = 30
        
        if self.exercise_name == "CMJ":
            # State 1: Standing / Sampling Baseline
            # State 2: Squatting (Knee < 130)
            # State 3: Flight (Knee > 150 and ankles above ground)
            # State 4: Landed / Recovering (Ankles back to ground)
            # State 5: Complete (Stands back up)
            
            if self.current_state == 1:
                # Collect first 10 frames of standing as baseline
                if len(self.baseline_samples) < 10:
                    self.baseline_samples.append(avg_ankle_y)
                    if len(self.baseline_samples) == 10:
                        self.baseline_ankle_y = np.mean(self.baseline_samples)
                    return "GOOD", "Sampling baseline...", self.current_state

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
                
                # Check for jump (Straightening knees AND ankles leaving ground)
                # We use a 5% torso buffer to ensure actual takeoff
                is_off_ground = avg_ankle_y < (self.baseline_ankle_y - torso_size * 0.05)
                
                if knee_angle > 145 and is_off_ground:
                    if len(self.state_history) > 0 and self.state_history[-1] == 1:
                        self.current_state = 3
                        self.state_history.append(2)
                        return "GOOD", "In Flight!", self.current_state
                    else:
                        # Bad form jump
                        self.reset()
                        return "BAD", "Invalid squat depth. Try again.", self.current_state
                return "GOOD", "Go deeper.", self.current_state
                
            elif self.current_state == 3:
                # In State 3 (Flight), wait until ankles return to baseline
                is_on_ground = False
                if self.baseline_ankle_y is not None:
                    # Feet are back near ground (not more than 3% torso size above)
                    if avg_ankle_y >= (self.baseline_ankle_y - torso_size * 0.03):
                        is_on_ground = True
                
                if is_on_ground:
                    # Feet touched ground, enter recovery phase
                    self.current_state = 4
                    self.state_history.append(3)
                    return "GOOD", "Landed! Now stand up.", 4
                
                return "GOOD", "In Air... Landing.", self.current_state

            elif self.current_state == 4:
                # Wait for user to stand back up for a clean recording finish
                if knee_angle > 150:
                    self.current_state = 5
                    self.state_history.append(4)
                    return "REP_COMPLETE", "Perfect Rep!", 5
                return "GOOD", "Stand up to complete.", 4
            
            elif self.current_state == 5:
                # Terminal state
                return "REP_COMPLETE", "Perfect Rep!", 5

        
        elif self.exercise_name == "Balance":
            # State 1: Standing straight
            # State 2: Lift one leg (Balance) -> Hold for 5 seconds
            
            if None in [torso_size, r_elbow_angle, l_elbow_angle]:
                return "BAD", "Body not fully visible", self.current_state
            
            if self.current_state == 1:
                if ankle_distance>torso_size*0.4 and r_elbow_angle<max_angle and r_elbow_angle>min_angle and l_elbow_angle<max_angle and l_elbow_angle>min_angle and self.is_near(r_wrist,r_hip, torso_size):
                    self.current_state = 2
                    self.good_frames = 0
                    return "GOOD", "Balancing. Hold it!", self.current_state
                elif r_elbow_angle<max_angle and r_elbow_angle>min_angle and l_elbow_angle<max_angle and l_elbow_angle>min_angle and self.is_near(r_wrist,r_hip, torso_size) :
                    self.good_frames += 1
                    if self.good_frames > 5:
                        return "GOOD", "Ready. Lift one leg.", self.current_state
                return "BAD", "Stand straight and hands on hips to begin.", self.current_state
                
            elif self.current_state == 2:
                akimbo = False
                if r_elbow_angle<max_angle and r_elbow_angle>min_angle and l_elbow_angle<max_angle and l_elbow_angle>min_angle and self.is_near(r_wrist,r_hip, torso_size):
                    akimbo = True
                if akimbo and ankle_distance>torso_size*0.4:
                    self.good_frames += 1
                    if self.good_frames >= 30: # roughly 5 seconds at 10fps
                        return "REP_COMPLETE", "Balance Complete!", self.current_state
                    return "GOOD", f"Holding... {self.good_frames/10:.1f}s", self.current_state
                else:
                    self.reset()
                    return "BAD", "Leg dropped! Reset FSM.", self.current_state

        return "BAD", "Invalid Exercise", self.current_state

