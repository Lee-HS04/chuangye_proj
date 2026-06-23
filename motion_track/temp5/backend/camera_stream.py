"""
backend/camera_stream.py
------------------------
Unified camera manager that handles:
  - Local webcam (OpenCV VideoCapture)
  - Phone / remote camera via MJPEG HTTP stream
    (phone apps like DroidCam, IP Webcam, or a browser MediaRecorder
     POST that streams to /camera/stream)

Frames are decoded and pushed to a shared async queue.
PoseDetector processes each frame; results broadcast via WebSocket.
"""

from __future__ import annotations

import asyncio
import base64
import io
import time
from typing import AsyncGenerator, Optional

import cv2
import numpy as np


class CameraManager:
    """
    Manages one active camera source at a time.
    Call start_webcam() or start_mjpeg_stream() to activate.
    Call stop() to release.
    Each decoded frame is placed into self.frame_queue for consumers.
    """

    def __init__(self, max_queue: int = 4):
        self.frame_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=max_queue)
        self._running      = False
        self._source_type  = None   # "webcam" | "mjpeg" | "ws_browser"
        self._cap: Optional[cv2.VideoCapture] = None
        self._target_fps   = 30

    # ── Webcam ────────────────────────────────────────────────────────────────

    async def start_webcam(self, device_index: int = 0, fps: int = 30):
        await self.stop()
        self._target_fps = fps
        self._source_type = "webcam"
        self._running = True
        asyncio.create_task(self._webcam_loop(device_index))

    async def _webcam_loop(self, device_index: int):
        cap = cv2.VideoCapture(device_index)
        cap.set(cv2.CAP_PROP_FPS, self._target_fps)
        self._cap = cap
        interval = 1.0 / self._target_fps

        while self._running:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.1)
                continue
            await self._push_frame(frame)
            await asyncio.sleep(interval)

        cap.release()
        self._cap = None

    # ── MJPEG HTTP stream (phone / IP camera) ─────────────────────────────────

    async def push_mjpeg_chunk(self, jpeg_bytes: bytes):
        """
        Called by the FastAPI route that receives MJPEG chunks from a phone.
        Each call is one JPEG frame (the HTTP client extracts boundaries).
        """
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            await self._push_frame(frame)

    # ── Browser WebSocket camera (MediaRecorder chunks → JPEG) ───────────────

    async def push_browser_frame(self, jpeg_b64: str):
        """
        Called when the browser sends a base64 JPEG frame via WebSocket.
        The frontend captures frames from getUserMedia and sends them here.
        """
        try:
            raw   = base64.b64decode(jpeg_b64)
            arr   = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                await self._push_frame(frame)
        except Exception as e:
            print(f"Frame decode error: {e}")

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _push_frame(self, frame: np.ndarray):
        """Non-blocking push; drops oldest frame if queue is full."""
        try:
            self.frame_queue.put_nowait(frame)
        except asyncio.QueueFull:
            try:
                self.frame_queue.get_nowait()   # discard oldest
            except asyncio.QueueEmpty:
                pass
            await self.frame_queue.put(frame)

    async def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        # Drain queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    # ── Frame → JPEG bytes for sending over WebSocket ─────────────────────────

    @staticmethod
    def frame_to_jpeg_b64(frame: np.ndarray, quality: int = 70) -> str:
        """Encode an OpenCV BGR frame to a base64 JPEG string."""
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf.tobytes()).decode("ascii")

    @staticmethod
    def resize_for_stream(frame: np.ndarray, max_width: int = 640) -> np.ndarray:
        h, w = frame.shape[:2]
        if w <= max_width:
            return frame
        scale = max_width / w
        return cv2.resize(frame, (max_width, int(h * scale)))