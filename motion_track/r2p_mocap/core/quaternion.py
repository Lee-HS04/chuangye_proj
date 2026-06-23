"""
Quaternion math for the R2P mocap engine.
Convention: [w, x, y, z]  (scalar-first)
Coordinate system: Y-up, X-right, Z-forward (right-handed)
"""

import numpy as np
from typing import Union

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < 1e-10:
        return IDENTITY.copy()
    return q / n


def multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def inverse(q: np.ndarray) -> np.ndarray:
    return conjugate(normalize(q))


def rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (active rotation)."""
    q = normalize(q)
    qv = np.array([0.0, v[0], v[1], v[2]])
    res = multiply(multiply(q, qv), conjugate(q))
    return res[1:]


def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between q1 and q2."""
    q1 = normalize(q1)
    q2 = normalize(q2)
    dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        return normalize(q1 + t * (q2 - q1))
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    s1 = np.cos(theta) - dot * np.sin(theta) / np.sin(theta_0)
    s2 = np.sin(theta) / np.sin(theta_0)
    return normalize(s1 * q1 + s2 * q2)


def to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize(q)
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def from_rotation_matrix(R: np.ndarray) -> np.ndarray:
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return normalize(np.array([0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s]))
    if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return normalize(np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s]))
    if R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return normalize(np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s]))
    s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
    return normalize(np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s]))


def from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis_n = axis / (np.linalg.norm(axis) + 1e-10)
    h = angle * 0.5
    return normalize(np.array([np.cos(h), *(axis_n * np.sin(h))]))


def to_euler_yup(q: np.ndarray) -> np.ndarray:
    """YXZ Euler decomposition (yaw=Y, pitch=X, roll=Z) for Y-up systems."""
    w, x, y, z = normalize(q)
    yaw   = np.arctan2(2*(w*y - x*z), 1 - 2*(y*y + x*x))
    pitch = np.arcsin(np.clip(2*(w*x + y*z), -1.0, 1.0))
    roll  = np.arctan2(2*(w*z - y*x), 1 - 2*(x*x + z*z))
    return np.array([yaw, pitch, roll])


def from_euler_yup(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Build quaternion from YXZ Euler angles (Y-up)."""
    cy, sy = np.cos(yaw*0.5),   np.sin(yaw*0.5)
    cp, sp = np.cos(pitch*0.5), np.sin(pitch*0.5)
    cr, sr = np.cos(roll*0.5),  np.sin(roll*0.5)
    return normalize(np.array([
        cy*cp*cr + sy*sp*sr,
        cy*sp*cr + sy*cp*sr,
        sy*cp*cr - cy*sp*sr,
        cy*cp*sr - sy*sp*cr,
    ]))


def relative_rotation(q_parent: np.ndarray, q_child: np.ndarray) -> np.ndarray:
    """Local rotation of child relative to parent: inv(parent) * child."""
    return normalize(multiply(inverse(q_parent), q_child))


def angle_between(q1: np.ndarray, q2: np.ndarray) -> float:
    """Angular distance in radians between two orientations."""
    dot = abs(np.clip(np.dot(normalize(q1), normalize(q2)), -1.0, 1.0))
    return 2.0 * np.arccos(dot)


def smooth_exp(q_prev: np.ndarray, q_new: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential smoothing for quaternion streams (alpha=1 → no smoothing)."""
    return slerp(q_prev, q_new, alpha)
