"""
R2P Mocap WebSocket Server

Connects to live IMU sensors (BLE or Serial), runs the reconstruction
engine, and streams skeleton poses to a browser via WebSocket.

Usage:
    # BLE mode (auto-discovers all R2P devices)
    python -m r2p_mocap.server

    # Serial mode (single USB device)
    python -m r2p_mocap.server --serial COM3

    # Simulation only (no hardware)
    python -m r2p_mocap.server --sim-only

    # Options
    python -m r2p_mocap.server --port 8000 --mode professional
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Set, Optional, List

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("r2p.server")

# ── Imports ────────────────────────────────────────────────────────────────────

try:
    from .core.config      import R2PConfig, SuitMode, CONSUMER_SENSORS, PROFESSIONAL_SENSORS
    from .core.imu_data    import IMUReading, SuitFrame
    from .engine           import R2PEngine
    from .simulation       import IMUSimulator, MotionType
    from .input.ble_receiver    import BLEReceiver
    from .input.serial_receiver import SerialReceiver
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from r2p_mocap.core.config      import R2PConfig, SuitMode, CONSUMER_SENSORS, PROFESSIONAL_SENSORS
    from r2p_mocap.core.imu_data    import IMUReading, SuitFrame
    from r2p_mocap.engine           import R2PEngine
    from r2p_mocap.simulation       import IMUSimulator, MotionType
    from r2p_mocap.input.ble_receiver    import BLEReceiver
    from r2p_mocap.input.serial_receiver import SerialReceiver

_STATIC_DIR      = Path(__file__).parent / "static"
_RECORDINGS_DIR  = Path(__file__).parent / "recordings"


# ── Engine state (shared between tasks) ───────────────────────────────────────

class EngineState:
    def __init__(self, config: R2PConfig):
        self.config     = config
        self.engine     = R2PEngine(config)
        self.sim        = IMUSimulator(
            sample_rate=config.sample_rate,
            mode=config.mode.value,
            noise=True,
        )
        self.engine.auto_calibrate()

        self.live_sensors:  Dict[str, IMUReading] = {}   # sensor_id -> latest reading
        self.last_pose_json: Optional[str]        = None
        self.frame_count:   int  = 0
        self.fps:           float = 0.0
        self._fps_t:        float = time.time()
        self._fps_frames:   int  = 0
        self.calibrating:   bool = False
        self._calib_frames: int  = 0
        self._calib_target: int  = 80

        try:
            from .grading.scorer import MotionScorer
            self.scorer: Optional[object] = MotionScorer(
                history_len=int(config.sample_rate * 12)
            )
        except Exception as e:
            log.warning(f"Grading module unavailable: {e}")
            self.scorer = None

        # Recording state
        self._recording:   bool       = False
        self._rec_name:    str        = "take_1"
        self._rec_frames:  List[dict] = []
        self._rec_start:   float      = 0.0

        # Real-time motion energy (sent in every WS frame)
        self._prev_quats:    Dict[str, list] = {}
        self.motion_energy:  float           = 0.0

    # ── Recording ──────────────────────────────────────────────────────────────

    def start_recording(self, name: str):
        self._rec_name   = name.replace(" ", "_") or "take_1"
        self._rec_frames = []
        self._rec_start  = time.time()
        self._recording  = True
        log.info(f"Recording started: {self._rec_name}")

    def stop_recording(self) -> dict:
        self._recording = False
        n = len(self._rec_frames)
        if n == 0:
            return {"frames": 0, "name": self._rec_name, "error": "no frames captured"}
        data = {
            "name":        self._rec_name,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "sample_rate": self.config.sample_rate,
            "sensor_ids":  sorted(self.live_sensors.keys()),
            "frame_count": n,
            "frames":      self._rec_frames,
        }
        _RECORDINGS_DIR.mkdir(exist_ok=True)
        path = _RECORDINGS_DIR / f"{self._rec_name}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        self._rec_frames = []
        log.info(f"Recording saved: {path} ({n} frames)")
        return {"frames": n, "name": self._rec_name}

    def ingest(self, reading: IMUReading):
        """Accept a live sensor reading."""
        self.live_sensors[reading.sensor_id] = reading

    def tick(self) -> Optional[str]:
        """
        Build a SuitFrame, fill missing sensors with simulation,
        run the engine, and return the JSON pose string.
        """
        t_now = time.time()
        sim_motion = MotionType.WALK if self.live_sensors else MotionType.IDLE

        # Get simulated frame to fill gaps
        sim_frame = self.sim.generate_frame(sim_motion)

        # Build combined frame: live sensors override simulated ones
        frame = SuitFrame(timestamp=t_now, frame_id=self.frame_count)
        for sid, reading in sim_frame.readings.items():
            frame.add(reading)
        for sid, reading in self.live_sensors.items():
            frame.add(reading)   # overrides simulation for this sensor

        # Calibration accumulation
        if self.calibrating:
            orientations = {
                sid: r.quaternion if r.quaternion is not None
                     else np.array([1.0, 0.0, 0.0, 0.0])
                for sid, r in frame.readings.items()
            }
            done = self.engine.feed_calibration_frame(orientations)
            self._calib_frames += 1
            if done or self._calib_frames >= self._calib_target:
                self.engine.finish_calibration(t_now)
                self.calibrating    = False
                self._calib_frames  = 0
                log.info("Calibration complete")

        pose = self.engine.process(frame)
        raw_quats = getattr(self.engine, "last_orientations", {})
        self.frame_count += 1

        # FPS estimate
        self._fps_frames += 1
        elapsed = t_now - self._fps_t
        if elapsed >= 1.0:
            self.fps = self._fps_frames / elapsed
            self._fps_frames = 0
            self._fps_t = t_now

        # Build JSON payload
        active  = sorted(self.live_sensors.keys())
        all_ids = (
            PROFESSIONAL_SENSORS
            if self.config.mode == SuitMode.PROFESSIONAL
            else CONSUMER_SENSORS
        )
        simmed = sorted(s for s in all_ids if s not in active)

        bones_dict = {}
        for bone, bt in pose.bones.items():
            bones_dict[bone.value] = {
                "pos": [round(float(v), 4) for v in bt.position],
                "rot": [round(float(v), 4) for v in bt.rotation],
            }

        mcf = (self.engine.output.frames[-1]
               if self.engine.output.frames else None)
        foot_contact = mcf.foot_contact if mcf else {}
        motion       = mcf.motion_state  if mcf else "idle"

        # Only send live sensor quaternions — browser falls back to T-pose for missing ones
        live_set = set(active)
        quats_dict = {
            sid: [round(float(v), 5) for v in q]
            for sid, q in raw_quats.items()
            if sid in live_set
        }

        payload = {
            "type":           "pose",
            "frame_id":       self.frame_count,
            "timestamp":      round(t_now, 3),
            "motion":         motion,
            "fps":            round(self.fps, 1),
            "active_sensors": active,
            "sim_sensors":    simmed,
            "foot_contact":   foot_contact,
            "calibrating":    self.calibrating,
            "motion_energy":  round(self.motion_energy, 5),
            "recording":      self._recording,
            "quats":          quats_dict,
        }
        self.last_pose_json = json.dumps(payload)

        # Real-time motion energy — mean quaternion delta across live sensors
        import math
        if self._prev_quats and quats_dict:
            dists = []
            for sid, q in quats_dict.items():
                prev = self._prev_quats.get(sid)
                if prev:
                    dot = max(-1.0, min(1.0, sum(a*b for a,b in zip(q, prev))))
                    dists.append(2.0 * math.acos(abs(dot)))
            self.motion_energy = float(sum(dists) / len(dists)) if dists else 0.0
        else:
            self.motion_energy = 0.0
        self._prev_quats = dict(quats_dict)

        if self._recording and quats_dict:
            self._rec_frames.append({
                "t":     round(t_now - self._rec_start, 4),
                "quats": quats_dict,
            })

        if self.scorer:
            self.scorer.push_frame(quats_dict)

        return self.last_pose_json

    def start_calibration(self):
        self.engine.begin_calibration()
        self.calibrating   = True
        self._calib_frames = 0
        log.info("Calibration started — hold T-pose")


# ── WebSocket manager ──────────────────────────────────────────────────────────

class WSManager:
    def __init__(self):
        self._clients: Set = set()

    def add(self, ws):
        self._clients.add(ws)

    def remove(self, ws):
        self._clients.discard(ws)

    async def broadcast(self, message: str):
        dead = set()
        for ws in self._clients:
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self._clients.discard(ws)

    @property
    def count(self) -> int:
        return len(self._clients)


# ── FastAPI app factory ────────────────────────────────────────────────────────

def make_app(state: EngineState, ws_manager: WSManager):
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
        from fastapi.responses import HTMLResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError:
        log.error("fastapi not installed — run: pip install fastapi uvicorn")
        sys.exit(1)

    app = FastAPI(title="R2P Mocap Server")

    # Serve static files if the directory exists
    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html_file = _STATIC_DIR / "index.html"
        if html_file.exists():
            return HTMLResponse(html_file.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>R2P Mocap Server</h1><p>index.html not found.</p>")

    @app.get("/api/status")
    async def status():
        return JSONResponse({
            "active_sensors":  sorted(state.live_sensors.keys()),
            "fps":             round(state.fps, 1),
            "frame_count":     state.frame_count,
            "motion":          state.engine.stats.motion_state,
            "mode":            state.config.mode.value,
            "ws_clients":      ws_manager.count,
            "calibrating":     state.calibrating,
        })

    @app.post("/api/calibrate")
    async def calibrate():
        state.start_calibration()
        return JSONResponse({"status": "calibrating", "frames_needed": state._calib_target})

    @app.post("/api/recording/start")
    async def recording_start(name: str = Query(default="take_1")):
        state.start_recording(name)
        return JSONResponse({"status": "recording", "name": name})

    @app.post("/api/recording/stop")
    async def recording_stop():
        return JSONResponse({"status": "saved", **state.stop_recording()})

    @app.get("/api/recordings")
    async def recordings_list():
        _RECORDINGS_DIR.mkdir(exist_ok=True)
        names = sorted(f.stem for f in _RECORDINGS_DIR.glob("*.json"))
        return JSONResponse(names)

    @app.get("/api/recording/{name}")
    async def recording_get(name: str):
        from fastapi.responses import FileResponse
        path = _RECORDINGS_DIR / f"{name}.json"
        if not path.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(str(path), media_type="application/json")

    @app.post("/api/segment/{name}")
    async def segment_recording(
        name:         str,
        energy_high:  float = Query(default=0.055),
        energy_low:   float = Query(default=0.028),
        min_action_s: float = Query(default=0.25),
        min_pause_s:  float = Query(default=0.40),
    ):
        path = _RECORDINGS_DIR / f"{name}.json"
        if not path.exists():
            return JSONResponse({"error": "recording not found"}, status_code=404)
        import json as _json
        rec = _json.loads(path.read_text(encoding="utf-8"))
        try:
            from .segmentation.motion_segmenter import segment
        except ImportError:
            from r2p_mocap.segmentation.motion_segmenter import segment
        result = segment(
            rec["frames"],
            sample_rate   = rec.get("sample_rate", 50.0),
            energy_high   = energy_high,
            energy_low    = energy_low,
            min_action_s  = min_action_s,
            min_pause_s   = min_pause_s,
        )
        result.pop("reps_with_frames", None)   # don't send all frames in HTTP response
        return JSONResponse(result)

    @app.post("/api/segment/{name}/extract-reps")
    async def extract_reps(name: str):
        """Segment recording and save each rep as its own recording file."""
        path = _RECORDINGS_DIR / f"{name}.json"
        if not path.exists():
            return JSONResponse({"error": "recording not found"}, status_code=404)
        import json as _json
        rec = _json.loads(path.read_text(encoding="utf-8"))
        try:
            from .segmentation.motion_segmenter import segment, save_reps
        except ImportError:
            from r2p_mocap.segmentation.motion_segmenter import segment, save_reps
        result  = segment(rec["frames"], sample_rate=rec.get("sample_rate", 50.0))
        saved   = save_reps(result, name, _RECORDINGS_DIR,
                            sample_rate=rec.get("sample_rate", 50.0),
                            recorded_at=rec.get("recorded_at", ""))
        return JSONResponse({
            "rep_count": result["rep_count"],
            "summary":   result["summary"],
            "saved":     [p.split("\\")[-1].split("/")[-1] for p in saved],
        })

    @app.delete("/api/recording/{name}")
    async def recording_delete(name: str):
        path = _RECORDINGS_DIR / f"{name}.json"
        if path.exists():
            path.unlink()
        return JSONResponse({"status": "deleted", "name": name})

    @app.post("/api/record-reference")
    async def record_reference(action: str = Query(default="start")):
        if state.scorer is None:
            return JSONResponse({"error": "grading module not loaded"}, status_code=503)
        if action == "start":
            state.scorer.start_recording()
            return JSONResponse({"status": "recording"})
        frames = state.scorer.stop_recording()
        return JSONResponse({
            "status":        "done",
            "frames":        frames,
            "has_reference": state.scorer.has_reference,
        })

    @app.post("/api/grade")
    async def grade_motion():
        if state.scorer is None:
            return JSONResponse({"error": "grading module not loaded"}, status_code=503)
        return JSONResponse(state.scorer.grade())

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        ws_manager.add(ws)
        log.info(f"WS client connected ({ws_manager.count} total)")
        try:
            while True:
                # Keep connection alive — actual data is pushed by broadcast task
                await asyncio.sleep(30)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            ws_manager.remove(ws)
            log.info(f"WS client disconnected ({ws_manager.count} total)")

    return app


# ── Background tasks ───────────────────────────────────────────────────────────

async def engine_loop(state: EngineState, ws_manager: WSManager, target_fps: float):
    """Runs the R2P engine at target_fps and broadcasts to all WS clients."""
    dt = 1.0 / target_fps
    while True:
        t0 = time.monotonic()
        pose_json = state.tick()
        if ws_manager.count > 0 and pose_json:
            await ws_manager.broadcast(pose_json)
        elapsed = time.monotonic() - t0
        await asyncio.sleep(max(0.0, dt - elapsed))


async def ble_ingest(state: EngineState, queue: asyncio.Queue):
    """Pulls IMUReadings from the BLE/serial queue into engine state."""
    while True:
        reading = await queue.get()
        state.ingest(reading)


# ── Main ───────────────────────────────────────────────────────────────────────

def build_args():
    p = argparse.ArgumentParser(description="R2P Mocap WebSocket Server")
    p.add_argument("--port",     type=int,   default=8000)
    p.add_argument("--host",     default="0.0.0.0")
    p.add_argument("--mode",     default="consumer", choices=["consumer", "professional"])
    p.add_argument("--fps",      type=float, default=50.0, help="Engine tick rate (Hz)")
    p.add_argument("--serial",   default=None, metavar="PORT", help="Serial port (e.g. COM3)")
    p.add_argument("--sim-only", action="store_true", help="No hardware — simulation only")
    p.add_argument("--height",   type=float, default=1.75)
    return p


async def _async_main(args):
    mode   = SuitMode.PROFESSIONAL if args.mode == "professional" else SuitMode.CONSUMER
    config = R2PConfig(mode=mode, sample_rate=args.fps, body_height=args.height)
    state      = EngineState(config)
    ws_manager = WSManager()
    queue: asyncio.Queue = asyncio.Queue(maxsize=500)

    tasks = [
        asyncio.ensure_future(engine_loop(state, ws_manager, args.fps)),
        asyncio.ensure_future(ble_ingest(state, queue)),
    ]

    if not args.sim_only:
        if args.serial:
            recv = SerialReceiver(args.serial, queue)
            tasks.append(asyncio.ensure_future(recv.run()))
            log.info(f"Serial receiver on {args.serial}")
        else:
            ble = BLEReceiver(queue)
            tasks.append(asyncio.ensure_future(ble.run()))
            log.info("BLE receiver scanning for R2P devices")
    else:
        log.info("Simulation-only mode (no hardware)")

    try:
        import uvicorn
    except ImportError:
        log.error("uvicorn not installed — run: pip install uvicorn")
        sys.exit(1)

    app = make_app(state, ws_manager)

    log.info(f"Server starting on http://{args.host}:{args.port}")
    log.info("Open http://localhost:%d in your browser", args.port)

    cfg = uvicorn.Config(app, host=args.host, port=args.port, log_level="warning")
    server = uvicorn.Server(cfg)

    await asyncio.gather(server.serve(), *tasks)


def main():
    args = build_args().parse_args()
    try:
        asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        log.info("Server stopped")


if __name__ == "__main__":
    main()
