"""
Serial/USB receiver for R2P IMU nodes.

Reads lines from a COM port (or /dev/ttyUSB*) and parses R2P packets.
Useful for single-device testing without BLE.

Usage:
    queue = asyncio.Queue()
    recv  = SerialReceiver("COM3", queue)
    await recv.run()
"""

import asyncio
import logging
import time
import threading
from typing import Optional

from .packet_parser import parse_packet, PacketParseError

log = logging.getLogger("r2p.serial")

_BAUD = 115200


class SerialReceiver:

    def __init__(self, port: str, queue: asyncio.Queue, baud: int = _BAUD):
        self.port  = port
        self.baud  = baud
        self.queue = queue
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def run(self):
        try:
            import serial
        except ImportError:
            log.error("pyserial not installed — run: pip install pyserial")
            return

        self._running = True
        self._loop = asyncio.get_event_loop()
        log.info(f"Serial receiver on {self.port} @ {self.baud}")

        # Run blocking serial reads on a background thread
        thread = threading.Thread(target=self._read_loop, daemon=True)
        thread.start()
        while self._running:
            await asyncio.sleep(0.1)
        thread.join(timeout=2.0)

    def stop(self):
        self._running = False

    def _read_loop(self):
        try:
            import serial
            with serial.Serial(self.port, self.baud, timeout=1.0) as ser:
                log.info(f"Serial port {self.port} opened")
                while self._running:
                    try:
                        raw = ser.readline().decode("utf-8", errors="replace").strip()
                    except Exception as exc:
                        log.warning(f"Serial read error: {exc}")
                        break
                    if not raw:
                        continue
                    # Filter out firmware debug lines that don't start with LOC
                    if not raw.startswith("LOC"):
                        log.debug(f"Serial non-data: {raw}")
                        continue
                    try:
                        reading = parse_packet(raw, timestamp=time.time())
                        if self._loop and self._loop.is_running():
                            asyncio.run_coroutine_threadsafe(
                                self.queue.put(reading), self._loop
                            )
                    except PacketParseError as e:
                        log.debug(f"Parse error: {e}")
        except Exception as exc:
            log.error(f"Serial {self.port} failed: {exc}")
