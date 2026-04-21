# core/utils.py
import numpy as np

def calculate_angle(a, b, c):
    if a is None or b is None or c is None:
        return None
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    return float(np.degrees(np.arccos(np.clip(cosine_angle,-1.0,1.0))))

def calculate_frontal_projection_angle(hip, knee, ankle):
    """
    Computes the 2D 'Frontal Plane Projection Angle' (Valgus/Varus) entirely from 3D points 
    regardless of camera angle by projecting the leg structure onto the person's own structural plane.
    """
    if hip is None or knee is None or ankle is None:
        return None
        
    hip = np.array(hip)
    knee = np.array(knee)
    ankle = np.array(ankle)
    
    # 1. We assume the Y-axis (index 1) is "UP/DOWN". 
    # Let's project the 3D leg purely onto the X-Y plane of the body
    # Because GVHMR local coordinates align roughly with camera, the 'Z' axis is depth.
    # By simply ignoring the 'Z' depth axis, we simulate a perfect frontal camera.
    vec_femur = hip[:2] - knee[:2]
    vec_tibia = ankle[:2] - knee[:2]
    
    # Standard 2D angle calculation for the frontal plane
    cosine_angle = np.dot(vec_femur, vec_tibia) / (np.linalg.norm(vec_femur)*np.linalg.norm(vec_tibia)+1e-6)
    angle_2d = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    
    return float(angle_2d)