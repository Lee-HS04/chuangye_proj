"""
backend/ws_server.py
--------------------
WebSocket server: bridges BLE packets → JSON → browser.

Each outbound message:
  {
    "sensor_id": "pelvis",
    "device":    "R2P-PELVIS-AA01",
    "q":    [w, x, y, z],
    "pos":  [x, y, z],      ← metres, world frame (Z-up)
    "vel":  [vx, vy, vz],   ← m/s
    "accel":[ax, ay, az],
    "gyro": [gx, gy, gz],
    "ts":   1234567890.123
  }

Inbound commands (JSON from browser):
  { "cmd": "reset_pos", "sensor_id": "pelvis" }   ← reset integrator
  { "cmd": "reset_all" }                           ← reset every integrator
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict, Optional, Set

from backend.ble_receiver import BLEReceiver, SensorPacket
from backend.motion_recorder import MotionRecorder
from backend.position_integrator import PositionIntegrator

try:
    from websockets.asyncio.server import serve
except ImportError:
    from websockets.legacy.server import serve

PORT = 8765
HOST = "localhost"


class WSServer:

    def __init__(self, simulate: bool = False):
        self._simulate     = simulate
        self._clients:     Set                          = set()
        self._receiver:    Optional[BLEReceiver]        = None
        self._integrators: Dict[str, PositionIntegrator] = {}
        self._last_msg:    Dict[str, dict]              = {}
        root = Path(__file__).resolve().parents[1]
        self._recorder = MotionRecorder(root / "recordings")

    # ── Entry point ──────────────────────────────────────────────────────

    async def run(self):
        self._receiver = BLEReceiver(on_packet=self._on_packet)
        ble_task = asyncio.create_task(
            self._receiver._simulate() if self._simulate else self._receiver.run()
        )
        print(f"WebSocket server listening on ws://{HOST}:{PORT}")
        async with serve(self._handler, HOST, PORT):
            await ble_task

    # ── BLE callback ─────────────────────────────────────────────────────

    def _on_packet(self, packet: SensorPacket):
        sid = packet.sensor_id

        if sid not in self._integrators:
            self._integrators[sid] = PositionIntegrator()

        pos, vel = self._integrators[sid].update(
            packet.quaternion, packet.accel, packet.gyro, packet.timestamp
        )

        msg = {
            "sensor_id": sid,
            "device":    packet.device_name,
            "q":     packet.quaternion.tolist(),
            "pos":   pos.tolist(),
            "vel":   vel.tolist(),
            "accel": packet.accel.tolist(),
            "gyro":  packet.gyro.tolist(),
            "ts":    packet.timestamp,
        }
        self._last_msg[sid] = msg
        self._recorder.add_frame(msg)

        if self._clients:
            payload = json.dumps(msg)
            self._send_all(payload)

    # ── WebSocket handler ────────────────────────────────────────────────

    async def _handler(self, ws):
        self._clients.add(ws)
        print(f"Browser connected  (clients: {len(self._clients)})")

        # Send last known state immediately on connect
        for msg in self._last_msg.values():
            try:
                await ws.send(json.dumps(msg))
            except Exception:
                pass

        try:
            async for raw in ws:
                self._handle_command(raw)
        finally:
            self._clients.discard(ws)
            print(f"Browser disconnected (clients: {len(self._clients)})")

    def _handle_command(self, raw: str):
        try:
            cmd = json.loads(raw)
        except Exception:
            return
        if cmd.get("cmd") == "reset_pos":
            sid = cmd.get("sensor_id")
            if sid and sid in self._integrators:
                self._integrators[sid].reset()
                print(f"Position reset: {sid}")
        elif cmd.get("cmd") == "reset_all":
            for intg in self._integrators.values():
                intg.reset()
            print("All positions reset")
        elif cmd.get("cmd") == "record_start":
            sid = cmd.get("sensor_id")
            status = self._recorder.start(sid)
            label = sid or "all sensors"
            print(f"Recording started: {label}")
            self._send_all(json.dumps(status))
        elif cmd.get("cmd") == "record_stop":
            status = self._recorder.stop()
            file_path = status.get("file")
            if file_path:
                print(f"Recording saved: {file_path}")
            self._send_all(json.dumps(status))

    def _send_all(self, payload: str):
        dead = set()
        for ws in self._clients:
            try:
                asyncio.ensure_future(ws.send(payload))
            except Exception:
                dead.add(ws)
        self._clients -= dead
