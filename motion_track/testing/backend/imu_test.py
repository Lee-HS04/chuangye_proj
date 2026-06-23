import serial
import struct

ser = serial.Serial("COM10", 115200, timeout=1)

print("=== CORRECT IMU DECODER ===\n")

SYNC1 = 0x7E
SYNC2 = 0x23


def read(n):
    d = ser.read(n)
    return d if len(d) == n else None


while True:

    if ser.read(1)[0] != SYNC1:
        continue
    if ser.read(1)[0] != SYNC2:
        continue

    length = ser.read(1)[0]
    cmd = ser.read(1)[0]

    data = read(length - 4)
    if not data:
        continue

    # ================= RAW SENSOR BUNDLE =================
    if cmd == 0x04:
        print("[RAW SENSOR FRAME]")
        print("HEX:", data.hex())

    # ================= QUATERNION =================
    elif cmd == 0x16:
        if len(data) >= 16:
            w, x, y, z = struct.unpack("<4f", data[:16])
            print(f"[QUAT] w={w:.3f} x={x:.3f} y={y:.3f} z={z:.3f}")

    # ================= EULER =================
    elif cmd == 0x26:
        if len(data) >= 12:
            r, p, y = struct.unpack("<3f", data[:12])
            print(f"[EULER] roll={r:.2f} pitch={p:.2f} yaw={y:.2f}")

    # ================= UNKNOWN =================
    else:
        print(f"[0x{cmd:02X}] HEX:", data.hex())