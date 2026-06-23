"""
Madgwick AHRS filter — fuses gyroscope + accelerometer (+ optional magnetometer).

Reference: S. O. H. Madgwick, "An efficient orientation filter for inertial and
inertial/magnetic sensor arrays", University of Bristol, 2010.

Coordinate system matches the R2P engine: Y-up.
At rest (sensor level, Y pointing up) the accelerometer reads ~[0, +g, 0].
Earth-frame reference for gravity: g_ref = [0, 1, 0] (normalised).
"""

import numpy as np
from ..core import quaternion as quat


class MadgwickAHRS:
    """
    Per-sensor orientation estimator.
    Maintain one instance per IMU; call update() every sample.
    """

    def __init__(self, sample_rate: float = 100.0, beta: float = 0.033):
        self.sample_rate = sample_rate
        self.beta = beta
        self.q = quat.IDENTITY.copy()

    def reset(self, q: np.ndarray = None):
        self.q = quat.normalize(q) if q is not None else quat.IDENTITY.copy()

    # ------------------------------------------------------------------
    def update(
        self,
        gyro:  np.ndarray,   # rad/s  body-frame
        accel: np.ndarray,   # m/s²   body-frame  (gravity included)
        mag:   np.ndarray = None,
    ) -> np.ndarray:
        dt = 1.0 / self.sample_rate

        # Gyro-only propagation when accelerometer has no signal
        acc_norm = np.linalg.norm(accel)
        if acc_norm < 0.1:
            self.q = self._integrate_gyro(self.q, gyro, dt)
            return self.q.copy()

        ax, ay, az = accel / acc_norm

        if mag is not None:
            mag_norm = np.linalg.norm(mag)
            if mag_norm > 1e-6:
                mx, my, mz = mag / mag_norm
                self.q = self._marg_step(self.q, gyro, ax, ay, az, mx, my, mz, dt)
                return self.q.copy()

        self.q = self._imu_step(self.q, gyro, ax, ay, az, dt)
        return self.q.copy()

    # ------------------------------------------------------------------
    def _integrate_gyro(self, q, gyro, dt):
        gx, gy, gz = gyro
        q_dot = 0.5 * quat.multiply(q, np.array([0.0, gx, gy, gz]))
        return quat.normalize(q + q_dot * dt)

    # ------------------------------------------------------------------
    def _imu_step(self, q, gyro, ax, ay, az, dt):
        """IMU update — Y-up gravity reference [0, 1, 0]."""
        q0, q1, q2, q3 = q
        gx, gy, gz = gyro

        # Predicted direction of gravity in sensor frame = R^T * [0,1,0]
        f1 = 2*(q0*q3 + q1*q2)          - ax
        f2 = 2*(0.5 - q1**2 - q3**2)    - ay
        f3 = 2*(q2*q3 - q0*q1)          - az

        # Jacobian rows
        J = np.array([
            [ 2*q3,  2*q2,  2*q1,  2*q0],
            [ 0,    -4*q1,  0,    -4*q3],
            [-2*q1, -2*q0,  2*q3,  2*q2],
        ])

        step = J.T @ np.array([f1, f2, f3])
        sn = np.linalg.norm(step)
        if sn > 1e-10:
            step /= sn

        q_dot = 0.5 * quat.multiply(q, np.array([0.0, gx, gy, gz]))
        q_dot -= self.beta * step
        return quat.normalize(q + q_dot * dt)

    # ------------------------------------------------------------------
    def _marg_step(self, q, gyro, ax, ay, az, mx, my, mz, dt):
        """MARG update — accelerometer + magnetometer + gyroscope."""
        q0, q1, q2, q3 = q
        gx, gy, gz = gyro

        # Rotate measured mag into earth frame to find reference flux
        h = quat.rotate_vector(q, np.array([mx, my, mz]))
        bx = float(np.sqrt(h[0]**2 + h[2]**2))  # horizontal magnitude (XZ plane for Y-up)
        by = float(h[1])                           # vertical component

        # Gravity objective (Y-up)
        f1 = 2*(q0*q3 + q1*q2)       - ax
        f2 = 2*(0.5 - q1**2 - q3**2) - ay
        f3 = 2*(q2*q3 - q0*q1)       - az

        # Magnetic objective (reference flux [bx, by, 0] projected into sensor frame)
        f4 = 2*bx*(0.5-q2**2-q3**2) + 2*by*(q1*q3-q0*q2) - mx
        f5 = 2*bx*(q1*q2-q0*q3)     + 2*by*(0.5-q1**2-q3**2) - my
        f6 = 2*bx*(q0*q2+q1*q3)     + 2*by*(q2*q3-q0*q1) - mz

        J = np.array([
            [ 2*q3,              2*q2,               2*q1,              2*q0            ],
            [ 0,                -4*q1,                0,                -4*q3            ],
            [-2*q1,             -2*q0,                2*q3,              2*q2            ],
            [-2*by*q2,           2*by*q3,            -4*bx*q2-2*by*q0,  -4*bx*q3+2*by*q1],
            [-2*bx*q3+2*by*q1,   2*bx*q2+2*by*q0,    2*bx*q1+2*by*q3,  -2*bx*q0+2*by*q2],
            [ 2*bx*q2,           2*bx*q3-4*by*q1,    2*bx*q0-4*by*q2,   2*bx*q1        ],
        ])

        step = J.T @ np.array([f1, f2, f3, f4, f5, f6])
        sn = np.linalg.norm(step)
        if sn > 1e-10:
            step /= sn

        q_dot = 0.5 * quat.multiply(q, np.array([0.0, gx, gy, gz]))
        q_dot -= self.beta * step
        return quat.normalize(q + q_dot * dt)
