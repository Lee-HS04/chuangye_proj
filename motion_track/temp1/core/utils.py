import math
import cv2

def calculate_angle(a, b, c):
    try:
        ab = (a[0]-b[0], a[1]-b[1])
        cb = (c[0]-b[0], c[1]-b[1])
        dot = ab[0]*cb[0] + ab[1]*cb[1]
        mag_ab = math.sqrt(ab[0]**2 + ab[1]**2)
        mag_cb = math.sqrt(cb[0]**2 + cb[1]**2)
        if mag_ab*mag_cb == 0:
            return None
        return math.degrees(math.acos(dot / (mag_ab*mag_cb)))
    except Exception:
        return None

def resize_if_needed(frame, max_width=800):
    if frame.shape[1] > max_width:
        scale = max_width / frame.shape[1]
        frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
    return frame