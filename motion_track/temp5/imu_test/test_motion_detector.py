"""
test_motion_detector.py — quick regression for the pelvis-only detector.

Runs scripted accelerometer patterns through MotionDetector and asserts the
gate behaves: bouncing in place must NOT walk; bounce + directional horizontal
accel must walk/run/strafe in the right direction.

    python test_motion_detector.py
"""
import numpy as np
from motion_detector import MotionDetector, GRAVITY

FPS = 50
DT = 1 / FPS
I = np.array([1.0, 0, 0, 0])   # identity orientation: sensor frame == world


def simulate(fwd=0.0, lat=0.0, bounce_amp=3.0, bounce_hz=1.8, n=160, quat=I):
    """fwd: +Z is backward, -Z is forward (body forward = -Z).
       lat: +X is right. bounce_amp/hz: vertical oscillation."""
    det = MotionDetector(fps=FPS)
    t = 0.0
    last = None
    for _ in range(n):
        t += DT
        vb = bounce_amp * np.sin(2 * np.pi * bounce_hz * t)
        a = np.array([lat, GRAVITY + vb, fwd])
        last = det.update(quat, a, np.zeros(3), t)
    return det, last


def main():
    cases = [
        # name,           fwd,  lat, bounce, hz,  expected
        ("idle (still)",   0.0,  0.0, 0.0,   0.0, {"idle"}),
        ("bounce in place",0.0,  0.0, 3.0,   1.8, {"march_in_place"}),
        ("walk forward",  -2.5,  0.0, 3.0,   1.8, {"walk"}),
        ("run forward",   -2.5,  0.0, 3.5,   3.0, {"run"}),
        ("walk backward",  2.5,  0.0, 3.0,   1.8, {"walk_back"}),
        ("strafe right",   0.0,  2.5, 3.0,   1.8, {"strafe_right"}),
        ("strafe left",    0.0, -2.5, 3.0,   1.8, {"strafe_left"}),
    ]
    ok = True
    for name, fwd, lat, amp, hz, expected in cases:
        det, st = simulate(fwd=fwd, lat=lat, bounce_amp=amp, bounce_hz=hz)
        passed = st.action in expected
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'}  {name:18s} → {st.action:15s} "
              f"cad={st.cadence_hz:4.1f} dir={st.move_dir} conf={st.confidence:.2f}")
    print("\nALL PASS ✓" if ok else "\nSOME FAILED ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())