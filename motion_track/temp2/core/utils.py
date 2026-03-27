# core/utils.py
import numpy as np

def calculate_angle(a, b, c):
    if a is None or b is None or c is None:
        return None
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    return float(np.degrees(np.arccos(np.clip(cosine_angle,-1.0,1.0))))