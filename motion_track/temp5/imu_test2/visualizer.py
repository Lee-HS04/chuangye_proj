"""
visualizer.py
-------------
Entry point. Starts the WebSocket/BLE server then opens the browser.

Usage:
    py visualizer.py               # connect to real BLE hardware
    py visualizer.py --simulate    # use built-in IMU simulator (no hardware needed)
"""

import argparse
import asyncio
import threading
import time
import webbrowser
from pathlib import Path

from backend.ws_server import WSServer, HOST, PORT


def main():
    parser = argparse.ArgumentParser(description="R2P IMU 3D viewer")
    parser.add_argument("--simulate", action="store_true",
                        help="Use the built-in IMU simulator instead of real BLE hardware.")
    args = parser.parse_args()

    html_path = (Path(__file__).parent / "frontend" / "index.html").resolve()
    url = html_path.as_uri()

    server = WSServer(simulate=args.simulate)

    # Open the browser after a short delay to give the WS server time to bind
    def open_browser():
        time.sleep(1.2)
        print(f"Opening browser: {url}")
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()

    print("Starting IMU visualizer  (Ctrl+C to quit)")
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()