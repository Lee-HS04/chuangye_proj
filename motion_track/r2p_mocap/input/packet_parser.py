"""
Parser for the R2P ESP32 packet format.

Packet format (sent over BLE NOTIFY and Serial):
    LOCR_THIGH AX0.12 AY-0.05 AZ9.81 GX0.01 GY0.02 GZ-0.03 QW0.70 QX0.01 QY0.02 QZ0.68

Fields:
    LOC<ID>     sensor location (uppercase, e.g. R_THIGH, PELVIS, CHEST)
    AX/AY/AZ    accelerometer m/s^2 (gravity included)
    GX/GY/GZ    gyroscope rad/s
    QW/QX/QY/QZ quaternion from hardware fusion [w, x, y, z]

Output: IMUReading with hardware quaternion populated.
"""

import time
import numpy as np
from ..core.imu_data import IMUReading


class PacketParseError(Exception):
    pass


_FIELD_LEN = {
    "AX": 2, "AY": 2, "AZ": 2,
    "GX": 2, "GY": 2, "GZ": 2,
    "QW": 2, "QX": 2, "QY": 2, "QZ": 2,
}


def parse_packet(raw: str, timestamp: float = None) -> IMUReading:
    """
    Parse one R2P packet string into an IMUReading.
    Raises PacketParseError on malformed input.
    """
    raw = raw.strip()
    if not raw:
        raise PacketParseError("empty packet")

    parts = raw.split()
    if not parts or not parts[0].startswith("LOC"):
        raise PacketParseError(f"missing LOC prefix: {raw!r}")

    sensor_id = parts[0][3:].lower()   # "R_THIGH" -> "r_thigh"
    if not sensor_id:
        raise PacketParseError(f"empty sensor id in: {raw!r}")

    values: dict = {}
    for token in parts[1:]:
        if len(token) < 3:
            continue
        key = token[:2].upper()
        try:
            values[key] = float(token[2:])
        except ValueError:
            pass  # skip malformed token

    def _get(k: str, default: float = 0.0) -> float:
        return values.get(k, default)

    accel = np.array([_get("AX"), _get("AY"), _get("AZ")], dtype=float)
    gyro  = np.array([_get("GX"), _get("GY"), _get("GZ")], dtype=float)

    # Quaternion is present when hardware fusion is running
    has_quat = all(k in values for k in ("QW", "QX", "QY", "QZ"))
    quat_arr = None
    if has_quat:
        q = np.array([_get("QW"), _get("QX"), _get("QY"), _get("QZ")], dtype=float)
        n = np.linalg.norm(q)
        quat_arr = q / n if n > 1e-6 else np.array([1.0, 0.0, 0.0, 0.0])

    return IMUReading(
        sensor_id=sensor_id,
        timestamp=timestamp if timestamp is not None else time.time(),
        gyro=gyro,
        accel=accel,
        quaternion=quat_arr,
    )
