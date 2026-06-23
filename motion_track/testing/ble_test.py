import asyncio
from bleak import BleakClient, BleakScanner

DEVICE_NAME = "R2P-PELVIS"
CHAR_UUID   = "abcd1234-5678-1234-5678-abcdef123456"

def parse(data):
    try:
        text = data.decode("utf-8").strip()
    except UnicodeDecodeError:
        print(f"[WARN] Truncated packet ({len(data)} bytes), skipping")
        return

    try:
        parts = text.split()
        if len(parts) != 10:
            print(f"[WARN] Incomplete packet ({len(parts)}/10 fields): {text!r}")
            return

        values = {}
        for p in parts:
            values[p[:2]] = float(p[2:])

        print(
            f"Accel  X={values['AX']:+7.3f}  Y={values['AY']:+7.3f}  Z={values['AZ']:+7.3f}  |  "
            f"Gyro   X={values['GX']:+7.3f}  Y={values['GY']:+7.3f}  Z={values['GZ']:+7.3f}  |  "
            f"Quat   W={values['QW']:+6.3f}  X={values['QX']:+6.3f}  Y={values['QY']:+6.3f}  Z={values['QZ']:+6.3f}"
        )
    except Exception as e:
        print(f"[WARN] Parse error: {e} — raw: {text!r}")


async def run():
    print(f"Scanning for '{DEVICE_NAME}'...")

    # ✅ FIXED PART (replaces find_device_by_name)
    devices = await BleakScanner.discover(timeout=10.0)

    target = None
    for d in devices:
        if d.name and d.name.startswith(DEVICE_NAME):
            target = d
            break

    if target is None:
        print("Device not found — make sure it is powered on and advertising.")
        return

    print(f"Found {target.name} ({target.address})")

    def on_disconnect(client):
        print("Disconnected from device.")

    async with BleakClient(target.address, disconnected_callback=on_disconnect) as client:
        print("Connected — receiving IMU data...")
        await client.start_notify(CHAR_UUID, lambda s, d: parse(d))

        try:
            while True:
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            await client.stop_notify(CHAR_UUID)


asyncio.run(run())