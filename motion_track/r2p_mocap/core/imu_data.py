from dataclasses import dataclass, field
from typing import Optional, Dict
import numpy as np


@dataclass
class IMUReading:
    """Raw IMU reading from one sensor at one timestamp."""
    sensor_id: str
    timestamp: float
    gyro:  np.ndarray              # rad/s  [x, y, z] in sensor body frame
    accel: np.ndarray              # m/s²   [x, y, z] — gravity included
    mag:        Optional[np.ndarray] = None  # uT [x, y, z], None when unavailable
    quaternion: Optional[np.ndarray] = None  # [w,x,y,z] pre-fused by hardware (bypasses Madgwick)

    def __post_init__(self):
        self.gyro  = np.asarray(self.gyro,  dtype=float)
        self.accel = np.asarray(self.accel, dtype=float)
        if self.mag is not None:
            self.mag = np.asarray(self.mag, dtype=float)
        if self.quaternion is not None:
            self.quaternion = np.asarray(self.quaternion, dtype=float)


@dataclass
class SuitFrame:
    """All sensor readings captured at one point in time."""
    timestamp: float
    frame_id:  int
    readings:  Dict[str, IMUReading] = field(default_factory=dict)

    def add(self, reading: IMUReading) -> None:
        self.readings[reading.sensor_id] = reading

    def has_sensor(self, sensor_id: str) -> bool:
        return sensor_id in self.readings
