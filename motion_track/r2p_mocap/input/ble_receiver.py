"""
BLE receiver for R2P IMU suit nodes.

Scans for devices advertising names that start with "R2P-", connects to all
of them concurrently, and subscribes to NOTIFY on the R2P characteristic.
Parsed IMUReadings are pushed into an asyncio.Queue for the engine loop.

Usage:
    queue = asyncio.Queue()
    receiver = BLEReceiver(queue)
    await receiver.run()   # runs until cancelled
"""

import asyncio
import logging
import time
from typing import Optional

from .packet_parser import parse_packet, PacketParseError

log = logging.getLogger("r2p.ble")

_SERVICE_UUID = "12345678-1234-1234-1234-1234567890ab"
_CHAR_UUID    = "abcd1234-5678-1234-5678-abcdef123456"
_NAME_PREFIX  = "R2P-"
_SCAN_INTERVAL_S = 5.0     # re-scan for new devices every N seconds
_CONNECT_TIMEOUT = 10.0


class BLEReceiver:

    def __init__(self, queue: asyncio.Queue, scan_interval: float = _SCAN_INTERVAL_S):
        self.queue = queue
        self.scan_interval = scan_interval
        self._connected: dict = {}   # address -> BleakClient
        self._running = False

    async def run(self):
        """Main loop: scan, connect, maintain connections."""
        try:
            from bleak import BleakScanner, BleakClient
            self._BleakClient = BleakClient
        except ImportError:
            log.error("bleak not installed — run: pip install bleak")
            return

        self._running = True
        log.info("BLE receiver started — scanning for R2P devices")
        while self._running:
            await self._scan_and_connect()
            await asyncio.sleep(self.scan_interval)

    def stop(self):
        self._running = False

    @property
    def connected_sensors(self) -> list:
        return list(self._connected.keys())

    # ------------------------------------------------------------------
    async def _scan_and_connect(self):
        try:
            from bleak import BleakScanner
            devices = await BleakScanner.discover(timeout=3.0)
        except Exception as exc:
            log.warning(f"BLE scan failed: {exc}")
            return

        for dev in devices:
            name = dev.name or ""
            if not name.startswith(_NAME_PREFIX):
                continue
            addr = dev.address
            if addr in self._connected:
                continue
            # Parse sensor id from device name: R2P-R_THIGH-A3F2 -> r_thigh
            parts = name.split("-")
            sensor_id = parts[1].lower() if len(parts) >= 2 else "unknown"
            log.info(f"Found {name} ({addr}) -> sensor '{sensor_id}'")
            asyncio.ensure_future(self._connect_and_listen(addr, sensor_id))

    async def _connect_and_listen(self, address: str, sensor_id: str):
        BleakClient = self._BleakClient
        try:
            async with BleakClient(address, timeout=_CONNECT_TIMEOUT) as client:
                self._connected[address] = client
                log.info(f"Connected to {address} ({sensor_id})")

                def _on_notify(_, data: bytearray):
                    raw = data.decode("utf-8", errors="replace").strip()
                    if not raw:
                        return
                    try:
                        reading = parse_packet(raw, timestamp=time.time())
                        self.queue.put_nowait(reading)
                    except PacketParseError as e:
                        log.debug(f"Parse error from {sensor_id}: {e}")

                await client.start_notify(_CHAR_UUID, _on_notify)
                # Hold connection open until client disconnects or we stop
                while self._running and client.is_connected:
                    await asyncio.sleep(0.5)
                await client.stop_notify(_CHAR_UUID)
        except Exception as exc:
            log.warning(f"BLE {address} error: {exc}")
        finally:
            self._connected.pop(address, None)
            log.info(f"Disconnected from {address} ({sensor_id})")
