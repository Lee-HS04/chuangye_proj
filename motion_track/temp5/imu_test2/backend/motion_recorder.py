"""
backend/motion_recorder.py
--------------------------
JSON session recorder for live IMU motion frames.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


class MotionRecorder:
    """Collect live motion frames and save them as a JSON session."""

    def __init__(self, output_dir: Path):
        self._output_dir = output_dir
        self._frames: list[dict] = []
        self._active = False
        self._started_at: Optional[float] = None
        self._sensor_id: Optional[str] = None

    @property
    def is_recording(self) -> bool:
        return self._active

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def sensor_id(self) -> Optional[str]:
        return self._sensor_id

    def start(self, sensor_id: Optional[str] = None) -> dict:
        self._frames = []
        self._active = True
        self._started_at = None
        self._sensor_id = sensor_id
        return self.status()

    def add_frame(self, frame: dict) -> None:
        if not self.is_recording:
            return

        if self._started_at is None:
            self._started_at = float(frame["ts"])

        if self._sensor_id and frame.get("sensor_id") != self._sensor_id:
            return

        ts = float(frame["ts"])
        self._frames.append({
            "time": ts - (self._started_at or ts),
            "timestamp": ts,
            "sensor_id": frame.get("sensor_id"),
            "device": frame.get("device"),
            "acc": frame.get("accel", [0.0, 0.0, 0.0]),
            "gyro": frame.get("gyro", [0.0, 0.0, 0.0]),
            "quat": frame.get("q", [1.0, 0.0, 0.0, 0.0]),
            "position": frame.get("pos", [0.0, 0.0, 0.0]),
            "velocity": frame.get("vel", [0.0, 0.0, 0.0]),
        })

    def stop(self) -> dict:
        if not self.is_recording:
            return self.status(recording=False)

        self._output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sensor = self._sensor_id or "all"
        path = self._output_dir / f"recording_{sensor}_{stamp}.json"
        duration = self._frames[-1]["time"] if self._frames else 0.0

        payload = {
            "schema": "imu_test2.motion_recording.v1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "sensor_id": self._sensor_id,
            "frame_count": len(self._frames),
            "duration": duration,
            "frames": self._frames,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        status = self.status(recording=False)
        status.update({
            "file": str(path),
            "frame_count": len(self._frames),
            "duration": duration,
        })

        self._frames = []
        self._active = False
        self._started_at = None
        self._sensor_id = None
        return status

    def status(self, recording: Optional[bool] = None) -> dict:
        return {
            "type": "recording",
            "recording": self.is_recording if recording is None else recording,
            "sensor_id": self._sensor_id,
            "frame_count": len(self._frames),
        }
