# body_tracking.py
import cv2
from ultralytics import YOLO
import math

# Model
model = YOLO("yolov8n-pose.pt")  # lightweight, CPU-friendly

# -----------------------
# Helper functions
# -----------------------
def joint_angle(a, b, c):
    angle = math.degrees(
        math.atan2(c[1]-b[1], c[0]-b[0]) -
        math.atan2(a[1]-b[1], a[0]-b[0])
    )
    return abs(angle)

def angle_from_vertical(p1, p2):
    dx = p2[0] - p1[0]
    dy = p1[1] - p2[1]
    return math.degrees(math.atan2(dx, dy))

# -----------------------
# Main function
# -----------------------
def get_keypoints(frame):
    results = model(frame)
    annotated = results[0].plot()

    keypoints = None
    if results[0].keypoints is not None:
        kp = results[0].keypoints.xy
        if len(kp) > 0:
            keypoints = [tuple(map(int, p)) for p in kp[0]]
    return keypoints, annotated