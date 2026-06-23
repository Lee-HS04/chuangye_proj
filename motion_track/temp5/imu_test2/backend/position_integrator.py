"""
backend/position_integrator.py
--------------------------------
Dead-reckoning position integrator for a single IMU sensor.

Algorithm:
  1. Rotate measured accelerometer (specific force) into world frame
     using the live quaternion.
  2. Subtract gravity to isolate linear acceleration.
  3. Integrate → velocity, integrate again → position.
  4. Apply Zero Velocity Update (ZUPT): when the sensor detects it is
     nearly stationary (low linear accel AND low gyro), velocity is
     zeroed to stop drift accumulating.
  5. A gentle velocity decay (high-pass) suppresses slow drift between
     ZUPT events.

Coordinate convention (matches the R2P firmware / simulator):
  World frame is Z-up (ENU-style):
    +X  right
    +Y  forward
    +Z  up
  Gravity vector in world frame: [0, 0, -9.81] m/s²
  Accelerometer reads +9.81 on Z when flat and stationary.

Dead-reckoning drift is real — you will see position wander over time
on a real sensor. ZUPT corrects it whenever you hold the sensor still.
"""

from __future__ import annotations
import numpy as np
import numpy.typing as npt


class PositionIntegrator:
    # ── Gravity ──────────────────────────────────────────────────────────
    GRAVITY_WORLD = np.array([0.0, 0.0, -9.81])   # Z-up world frame

    # ── ZUPT thresholds ──────────────────────────────────────────────────
    # Linear accel magnitude (m/s²) below which the sensor is "still"
    ACCEL_STILL_THRESH = 0.12
    # Gyro magnitude (°/s) below which the sensor is "still"
    GYRO_STILL_THRESH  = 4.0
    # How many consecutive still samples needed before zeroing velocity
    ZUPT_WINDOW = 6

    # ── Drift suppression ────────────────────────────────────────────────
    # Velocity multiplied by this each step (≈ time constant of ~0.5 s at 50 Hz)
    VELOCITY_DECAY = 0.975
    ACCEL_DEADBAND = 0.04
    VERTICAL_STILL_DECAY = 0.92

    def __init__(self):
        self.velocity   = np.zeros(3)
        self.position   = np.zeros(3)
        self._prev_ts   = None
        self._still_ctr = 0

    # ── Public API ───────────────────────────────────────────────────────

    def update(self,
               q:     npt.NDArray[np.float64],   # [w, x, y, z] unit quaternion
               accel: npt.NDArray[np.float64],   # m/s²   measured in SENSOR frame
               gyro:  npt.NDArray[np.float64],   # °/s    in sensor frame
               ts:    float                       # UNIX timestamp
               ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Returns (position [m], velocity [m/s]) both in world frame.
        """
        if self._prev_ts is None:
            self._prev_ts = ts
            return self.position.copy(), self.velocity.copy()

        dt = ts - self._prev_ts
        self._prev_ts = ts

        if dt <= 0 or dt > 0.5:           # skip bad / stale packets
            return self.position.copy(), self.velocity.copy()

        R = self._quat_to_R(q)

        # Linear acceleration in world frame (gravity removed)
        a_world = R @ accel + self.GRAVITY_WORLD
        a_world[np.abs(a_world) < self.ACCEL_DEADBAND] = 0.0

        # ── ZUPT detection ────────────────────────────────────────────
        a_mag = float(np.linalg.norm(a_world))
        g_mag = float(np.linalg.norm(gyro))

        if a_mag < self.ACCEL_STILL_THRESH and g_mag < self.GYRO_STILL_THRESH:
            self._still_ctr += 1
        else:
            self._still_ctr = 0

        if self._still_ctr >= self.ZUPT_WINDOW:
            # Sensor is stationary — zero velocity to prevent drift
            self.velocity[:] = 0.0
            self.position[2] *= self.VERTICAL_STILL_DECAY
        else:
            self.velocity  += a_world * dt
            self.velocity  *= self.VELOCITY_DECAY   # drift suppression

        self.position += self.velocity * dt
        return self.position.copy(), self.velocity.copy()

    def reset(self):
        """Reset position and velocity to origin."""
        self.velocity[:]  = 0.0
        self.position[:]  = 0.0
        self._prev_ts     = None
        self._still_ctr   = 0

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _quat_to_R(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """[w,x,y,z] unit quaternion → 3×3 rotation matrix (body→world)."""
        w, x, y, z = q
        return np.array([
            [1-2*(y*y+z*z),  2*(x*y-w*z),   2*(x*z+w*y)  ],
            [2*(x*y+w*z),    1-2*(x*x+z*z), 2*(y*z-w*x)  ],
            [2*(x*z-w*y),    2*(y*z+w*x),   1-2*(x*x+y*y)],
        ])
