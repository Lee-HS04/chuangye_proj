"""
backend/ble_receiver.py
-----------------------
Connects to ESP32-C3 mini IMU nodes running the R2P NimBLE firmware.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

try:
    from bleak import BleakClient, BleakScanner
    BLEAK_AVAILABLE = True
except ImportError:
    BLEAK_AVAILABLE = False
    print("WARNING: bleak not installed. BLE will be simulated.")


SERVICE_UUID  = "12345678-1234-1234-1234-1234567890ab"
CHAR_UUID     = "abcd1234-5678-1234-5678-abcdef123456"
DEVICE_PREFIX = "R2P-"


LOC_TO_SENSOR_ID: dict[str, str] = {
    "PELVIS":       "pelvis",
    "CHEST":        "chest",
    "L_THIGH":      "thigh_l",
    "R_THIGH":      "thigh_r",
    "L_SHIN":       "shin_l",
    "R_SHIN":       "shin_r",
    "L_SHOULDER":   "l_shoulder",
    "R_SHOULDER":   "r_shoulder",
    "L_UPPER_ARM":  "l_upper_arm",
    "R_UPPER_ARM":  "r_upper_arm",
    "L_FOREARM":    "l_forearm",
    "R_FOREARM":    "r_forearm",
    "HEAD":         "head",
    "L_FOOT":       "l_foot",
    "R_FOOT":       "r_foot",
}

SENSOR_POSITIONS = list(LOC_TO_SENSOR_ID.values())


@dataclass
class SensorPacket:
    sensor_id:   str
    loc_raw:     str
    quaternion:  np.ndarray
    accel:       np.ndarray
    gyro:        np.ndarray
    timestamp:   float
    mac:         str
    device_name: str


class BLEReceiver:

    SENSOR_IDS = SENSOR_POSITIONS

    def __init__(self, on_packet: Callable[[SensorPacket], None],
                 on_new_device: Optional[Callable[[str], None]] = None):
        self._on_packet     = on_packet
        self._on_new_device = on_new_device
        self._clients:      dict[str, BleakClient] = {}
        self._device_names: dict[str, str] = {}
        self._running = False

    async def run(self):
        if not BLEAK_AVAILABLE:
            print("bleak not available — running IMU simulator.")
            await self._simulate()
            return

        self._running = True
        print(f"Scanning for BLE devices with prefix '{DEVICE_PREFIX}'…")

        while self._running:
            try:
                devices = await BleakScanner.discover(timeout=4.0)
                r2p_devices = [
                    d for d in devices
                    if d.name and d.name.startswith(DEVICE_PREFIX)
                ]
                for device in r2p_devices:
                    mac = device.address.upper()
                    if mac not in self._clients:
                        print(f"Found: {device.name} ({mac})")
                        asyncio.create_task(self._connect(device))
            except Exception as exc:
                print(f"BLE scan error: {exc}")

            await asyncio.sleep(3.0)

    async def _connect(self, device):
        mac  = device.address.upper()
        name = device.name or "Unknown"

        def on_disconnect(client):
            print(f"Disconnected: {name} ({mac})")
            self._clients.pop(mac, None)
            self._device_names.pop(mac, None)

        try:
            client = BleakClient(device.address, disconnected_callback=on_disconnect)
            await client.connect(timeout=10.0)
            self._clients[mac] = client
            self._device_names[mac] = name
            print(f"Connected: {name} ({mac})")

            await client.start_notify(
                CHAR_UUID,
                lambda _, data, _mac=mac, _name=name:
                    self._on_notification(data, _mac, _name)
            )
        except Exception as exc:
            print(f"Failed to connect {name} ({mac}): {exc}")
            self._clients.pop(mac, None)

    def _on_notification(self, data: bytearray, mac: str, device_name: str):
        packet = self._parse(data, mac, device_name)
        if packet is not None:
            self._on_packet(packet)

    def _parse(self, data: bytearray, mac: str, device_name: str) -> Optional[SensorPacket]:
        try:
            text  = data.decode("utf-8").strip()
            parts = text.split()

            if len(parts) != 11:
                return None

            if not parts[0].startswith("LOC"):
                return None

            loc_raw   = parts[0][3:]
            sensor_id = LOC_TO_SENSOR_ID.get(loc_raw)

            if sensor_id is None:
                return None

            values: dict[str, float] = {}
            for p in parts[1:]:
                values[p[:2]] = float(p[2:])

            required = {"AX", "AY", "AZ", "GX", "GY", "GZ", "QW", "QX", "QY", "QZ"}
            if required - values.keys():
                return None

            accel = np.array([values["AX"], values["AY"], values["AZ"]])
            gyro  = np.array([values["GX"], values["GY"], values["GZ"]])
            q     = np.array([values["QW"], values["QX"], values["QY"], values["QZ"]])

            norm = np.linalg.norm(q)
            if norm < 1e-6:
                return None
            q /= norm

            return SensorPacket(
                sensor_id=sensor_id,
                loc_raw=loc_raw,
                quaternion=q,
                accel=accel,
                gyro=gyro,
                timestamp=time.time(),
                mac=mac,
                device_name=device_name,
            )

        except Exception:
            return None

    def get_connected_devices(self) -> list[dict]:
        return [
            {"mac": mac, "name": self._device_names.get(mac, "Unknown")}
            for mac in self._clients
        ]

    async def scan(self, timeout: float = 5.0) -> list[dict]:
        if not BLEAK_AVAILABLE:
            return []
        try:
            devices = await BleakScanner.discover(timeout=timeout)
            return [
                {"name": d.name, "mac": d.address}
                for d in devices
                if d.name and d.name.startswith(DEVICE_PREFIX)
            ]
        except Exception as exc:
            print(f"BLE scan error: {exc}")
            return []

    async def stop(self):
        self._running = False
        for client in list(self._clients.values()):
            try:
                await client.disconnect()
            except Exception:
                pass
        self._clients.clear()
        self._device_names.clear()

    async def _simulate(self):
        self._running = True
        t, dt = 0.0, 1 / 50

        sim_sensors = [
            ("AA:BB:CC:DD:EE:01", "R2P-PELVIS-AA01",  "PELVIS"),
            ("AA:BB:CC:DD:EE:02", "R2P-L_THIGH-AA02", "L_THIGH"),
            ("AA:BB:CC:DD:EE:03", "R2P-R_THIGH-AA03", "R_THIGH"),
        ]

        print("Simulating IMU @50Hz")

        while self._running:
            t += dt
            for i, (mac, name, loc) in enumerate(sim_sensors):
                flex = (np.pi / 6) * (1 - np.cos(2 * np.pi * 0.4 * t + i * 0.4))
                half = flex / 2

                q = np.array([np.cos(half), np.sin(half), 0, 0])
                q += np.random.normal(0, 0.002, 4)
                q /= np.linalg.norm(q)

                accel = np.array([0, 0, 9.81])
                gyro  = np.array([np.degrees(flex * 0.1), 0, 0])

                text = (
                    f"LOC{loc} "
                    f"AX{accel[0]:.2f} AY{accel[1]:.2f} AZ{accel[2]:.2f} "
                    f"GX{gyro[0]:.2f} GY{gyro[1]:.2f} GZ{gyro[2]:.2f} "
                    f"QW{q[0]:.2f} QX{q[1]:.2f} QY{q[2]:.2f} QZ{q[3]:.2f}"
                )

                packet = self._parse(bytearray(text.encode()), mac, name)
                if packet:
                    self._on_packet(packet)

            await asyncio.sleep(dt)
