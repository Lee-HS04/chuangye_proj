"""
R2P Inertial Motion Capture Suit — Main Entry Point

Usage examples:
  python main.py                              # walk, consumer, live view
  python main.py --motion run --mode pro      # run, professional (15 IMUs)
  python main.py --motion walk --export bvh   # export captured.bvh
  python main.py --motion seq                 # walk->run->squat sequence
  python main.py --no-viz --fps 100 --dur 10  # headless benchmark
  python main.py --list-motions               # show all motion types
"""

import argparse
import sys
import time
from typing import List, Tuple

import numpy as np

try:
    # Running as  python -m r2p_mocap.main  (package mode)
    from .core.config              import R2PConfig, SuitMode
    from .engine                   import R2PEngine
    from .simulation               import IMUSimulator, MotionType
    from .visualization            import SkeletonRenderer
    from .output                   import SkeletonOutput, BVHExporter
    from .core.skeleton            import SkeletonPose
    from .biomechanics.foot_contact import FootState
except ImportError:
    # Running as  python main.py  from inside the r2p_mocap folder
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from r2p_mocap.core.config              import R2PConfig, SuitMode
    from r2p_mocap.engine                   import R2PEngine
    from r2p_mocap.simulation               import IMUSimulator, MotionType
    from r2p_mocap.visualization            import SkeletonRenderer
    from r2p_mocap.output                   import SkeletonOutput, BVHExporter
    from r2p_mocap.core.skeleton            import SkeletonPose
    from r2p_mocap.biomechanics.foot_contact import FootState


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="r2p_mocap",
        description="R2P Inertial Motion Capture Suit Reconstruction Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--motion",  default="walk",
                   choices=["idle", "walk", "run", "squat", "lunge",
                            "side_step", "pivot", "seq"],
                   help="Motion type to simulate (default: walk)")
    p.add_argument("--mode",    default="consumer",
                   choices=["consumer", "professional"],
                   help="Suit mode — consumer=6 IMUs, professional=15 IMUs")
    p.add_argument("--dur",     type=float, default=8.0,
                   help="Duration in seconds (default: 8)")
    p.add_argument("--fps",     type=float, default=100.0,
                   help="Simulation sample rate Hz (default: 100)")
    p.add_argument("--height",  type=float, default=1.75,
                   help="Subject height in metres (default: 1.75)")
    p.add_argument("--no-viz",  action="store_true",
                   help="Disable real-time visualisation")
    p.add_argument("--export",  choices=["json", "bvh", "both"],
                   help="Export captured session")
    p.add_argument("--noise",   action="store_true", default=True,
                   help="Enable IMU noise (default: on)")
    p.add_argument("--no-noise",action="store_true",
                   help="Disable IMU noise (cleaner output)")
    p.add_argument("--list-motions", action="store_true",
                   help="Print available motion types and exit")
    return p


# ── Helpers ────────────────────────────────────────────────────────────────────

def print_banner():
    print("""
+------------------------------------------------------------------+
|        R2P  Inertial Motion Capture Suit Framework               |
|        Real-time skeletal reconstruction from IMU sensors        |
+------------------------------------------------------------------+
|  15-sensor full-body  |  6-sensor lower-body consumer mode       |
|  Madgwick AHRS fusion |  Biomechanical constraints               |
|  Foot contact IK      |  BVH / JSON export                       |
+------------------------------------------------------------------+
""")


def list_motions():
    print("Available motion types:")
    descriptions = {
        "idle":      "Standing still with natural micro-sway",
        "walk":      "Normal walking gait (~1.4 m/s)",
        "run":       "Running gait (~3.5 m/s)",
        "squat":     "Bilateral squat (controlled descent/ascent)",
        "lunge":     "Forward lunge cycle",
        "side_step": "Lateral shuffle steps",
        "pivot":     "Rotational pivot / direction change",
        "seq":       "Automatic sequence: idle->walk->run->squat->walk",
    }
    for name, desc in descriptions.items():
        print(f"  {name:<12} {desc}")


def build_motion_sequence(dur: float) -> List[Tuple[float, MotionType]]:
    """Time-stamped motion transitions for the 'seq' mode."""
    return [
        (0.0,         MotionType.IDLE),
        (dur * 0.10,  MotionType.WALK),
        (dur * 0.35,  MotionType.RUN),
        (dur * 0.60,  MotionType.WALK),
        (dur * 0.72,  MotionType.SQUAT),
        (dur * 0.88,  MotionType.WALK),
    ]


def print_stats(engine: R2PEngine, total_frames: int, elapsed_s: float):
    s = engine.stats
    sep = "-" * 54
    print(f"\n{sep}")
    print(f"  Frames processed : {s.frames_processed}")
    print(f"  Real-time factor : {total_frames / max(elapsed_s, 0.001) / engine.config.sample_rate:.2f}x")
    print(f"  Avg process time : {s.avg_process_ms:.2f} ms/frame")
    print(f"  Peak latency     : {s.last_process_ms:.2f} ms")
    print(f"  Active sensors   : {s.active_sensors}")
    print(f"  Last motion      : {s.motion_state}")
    print(f"{sep}\n")


# ── Main run loop ──────────────────────────────────────────────────────────────

def run(args: argparse.Namespace):
    print_banner()

    # Config
    mode   = SuitMode.PROFESSIONAL if args.mode == "professional" else SuitMode.CONSUMER
    config = R2PConfig(
        mode=mode,
        sample_rate=args.fps,
        body_height=args.height,
        enable_ik=True,
        enable_foot_contact=True,
        enable_drift_correction=True,
    )

    print(f"  Mode    : {mode.value.upper()}  ({len(config.active_sensors)} sensors)")
    print(f"  Motion  : {args.motion}")
    print(f"  Duration: {args.dur:.1f} s  @ {args.fps:.0f} Hz")
    print(f"  Subject : {args.height:.2f} m\n")

    # Engine
    engine = R2PEngine(config)
    engine.auto_calibrate(forward_yaw_deg=0.0)
    print("  [OK] Engine initialised")
    print("  [OK] T-pose calibration applied (identity)\n")

    # Simulator
    use_noise = args.noise and not args.no_noise
    sim = IMUSimulator(
        sample_rate=args.fps,
        mode=args.mode,
        noise=use_noise,
    )

    # Motion sequence
    if args.motion == "seq":
        motion_seq = build_motion_sequence(args.dur)
        primary_motion = MotionType.IDLE
        print("  Motion sequence:")
        for t, m in motion_seq:
            print(f"    {t:5.1f}s -> {m.value}")
        print()
    else:
        motion_seq = None
        primary_motion = MotionType[args.motion.upper()]

    # Visualiser
    renderer: SkeletonRenderer = None
    if not args.no_viz:
        try:
            renderer = SkeletonRenderer(
                title=f"R2P Mocap — {args.motion.upper()}  [{mode.value}]",
                fps=int(min(args.fps, 60)),
            )
            renderer.start()
            print("  [OK] Visualiser started\n")
        except Exception as exc:
            print(f"  [!] Visualiser failed to start: {exc}")
            print("      Running headless.\n")
            renderer = None

    # ── Main loop ────────────────────────────────────────────────────────
    poses: List[SkeletonPose] = []
    t_start = time.perf_counter()
    target_dt = 1.0 / args.fps

    try:
        for frame in sim.stream(args.dur, primary_motion, motion_seq):
            loop_start = time.perf_counter()

            pose = engine.process(frame)
            poses.append(pose)

            # Determine current motion label for visualiser
            label = engine.stats.motion_state

            if renderer is not None:
                foot_states = None
                if engine.output.frames:
                    last = engine.output.frames[-1]
                    foot_states = {
                        side: type("FS", (), {"contact": v})()
                        for side, v in last.foot_contact.items()
                    }
                renderer.update(pose, foot_states, label)

            # Console progress every second
            if engine.stats.frames_processed % int(args.fps) == 0:
                t_now = time.perf_counter() - t_start
                pct = min(100, t_now / args.dur * 100)
                bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
                print(
                    f"\r  [{bar}] {pct:5.1f}%  "
                    f"{t_now:.1f}s/{args.dur:.1f}s  "
                    f"motion={label:<12}  "
                    f"lat={engine.stats.avg_process_ms:.1f}ms",
                    end="", flush=True,
                )

            # Real-time pacing (headless mode only — visualiser already paces)
            if renderer is None:
                elapsed = time.perf_counter() - loop_start
                wait = target_dt - elapsed
                if wait > 0:
                    time.sleep(wait)

    except KeyboardInterrupt:
        print("\n  [!] Interrupted by user")

    print()
    elapsed_total = time.perf_counter() - t_start
    print_stats(engine, len(poses), elapsed_total)

    # ── Export ────────────────────────────────────────────────────────────
    if args.export in ("json", "both"):
        path = "captured.json"
        engine.output.save(path)
        print(f"  [OK] Exported JSON  -> {path}")

    if args.export in ("bvh", "both"):
        path = "captured.bvh"
        BVHExporter().save(engine.output, path, body_height=args.height)
        print(f"  [OK] Exported BVH   -> {path}")

    # ── Post-session replay ───────────────────────────────────────────────
    if renderer is not None and poses:
        print(f"\n  Replaying {len(poses)} frames... (close window to exit)\n")
        mcframes = engine.output.frames
        foot_seq = [
            {side: type("FS", (), {"contact": v})()
             for side, v in f.foot_contact.items()}
            for f in mcframes
        ]
        labels = [f.motion_state for f in mcframes]
        try:
            renderer.render_sequence(
                poses, foot_seq, labels,
                interval_ms=int(1000 / min(args.fps, 60)),
            )
        except Exception:
            pass

    print("  Done.\n")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.list_motions:
        list_motions()
        sys.exit(0)

    run(args)


if __name__ == "__main__":
    main()
