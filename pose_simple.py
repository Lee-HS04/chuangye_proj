#!/usr/bin/env python3
import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

LANDMARKS = {
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28,
    'left_foot': 31, 'right_foot': 32,
    'left_shoulder': 11, 'right_shoulder': 12
}

def angle(v1, v2):
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos = np.clip(dot / (norm + 1e-10), -1.0, 1.0)
    return np.degrees(np.arccos(cos))

class Smoother:
    def __init__(self, window=3):
        self.window = window
        self.history = []
    
    def smooth(self, landmarks):
        self.history.append(landmarks)
        if len(self.history) > self.window:
            self.history.pop(0)
        if not self.history:
            return landmarks
        avg = []
        for i in range(len(landmarks)):
            xs = [p[i].x for p in self.history]
            ys = [p[i].y for p in self.history]
            zs = [p[i].z for p in self.history]
            avg.append(type(landmarks[0])(
                x=np.mean(xs), y=np.mean(ys), z=np.mean(zs),
                visibility=landmarks[i].visibility))
        return avg

def calculate_angles(landmarks, img_shape):
    h, w = img_shape[:2]
    def coord(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h, lm.z * w])
    
    l_hip = coord(LANDMARKS['left_hip'])
    l_knee = coord(LANDMARKS['left_knee'])
    l_ankle = coord(LANDMARKS['left_ankle'])
    l_foot = coord(LANDMARKS['left_foot'])
    r_hip = coord(LANDMARKS['right_hip'])
    r_knee = coord(LANDMARKS['right_knee'])
    r_ankle = coord(LANDMARKS['right_ankle'])
    r_foot = coord(LANDMARKS['right_foot'])
    l_shoulder = coord(LANDMARKS['left_shoulder'])
    r_shoulder = coord(LANDMARKS['right_shoulder'])
    
    l_thigh = l_knee - l_hip
    l_shin = l_ankle - l_knee
    r_thigh = r_knee - r_hip
    r_shin = r_ankle - r_knee
    l_torso = l_shoulder - l_hip
    r_torso = r_shoulder - r_hip
    l_foot_vec = l_foot - l_ankle
    r_foot_vec = r_foot - r_ankle
    
    return {
        'left_hip': angle(l_thigh, l_torso),
        'right_hip': angle(r_thigh, r_torso),
        'left_knee': angle(l_thigh, l_shin),
        'right_knee': angle(r_thigh, r_shin),
        'left_ankle': angle(l_shin, l_foot_vec),
        'right_ankle': angle(r_shin, r_foot_vec)
    }

def main():
    cap = cv2.VideoCapture(0)
    pose = mp_pose.Pose()
    smoother = Smoother()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = pose.process(rgb)
        
        if results.pose_landmarks:
            landmarks = smoother.smooth(results.pose_landmarks.landmark)
            angles = calculate_angles(landmarks, frame.shape)
            rgb.flags.writeable = True
            output = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(output, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            y = 30
            for joint, ang in angles.items():
                cv2.putText(output, f'{joint}: {ang:.1f}°', (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                y += 25
            cv2.imshow('Pose Angles', output)
        else:
            cv2.imshow('Pose Angles', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
