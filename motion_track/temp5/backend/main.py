"""
backend/main.py  (v4 — root_integrator v7: smooth velocity-glide locomotion)
-----------------------------------------------------------------------------
FastAPI server — full flow:

  /              → frontend/calibrate.html  (entry point)
  /select        → frontend/select.html
  /test          → frontend/index.html      (live session runner)
  /review        → frontend/review.html
  /settings      → frontend/settings.html

WebSocket routes:
  /ws/imu      — real-time skeleton + biomechanics broadcast
  /ws/camera   — browser sends camera frames; server responds with YOLO pose state

REST routes:
  POST /pose/start?mode=calibration|<test_id>
  POST /pose/reset
  POST /session/start?test_id=...
  POST /session/stop
  GET  /session/check?test_id=...
  GET  /session/{id}/summary
  POST /calibrate/upload    (manual video upload → GVHMR)
  POST /review/annotate
  POST /ble/tpose
  GET  /ble/status
  POST /ble/scan
  GET  /protocols
  GET  /protocol/{id}
  GET  /profile
  GET  /profiles
  POST /profile/create
  POST /profile/switch
  DELETE /profile/{id}
  PATCH /profile/body
  GET  /profile/history
  POST /motion_calib/begin           (NEW v7: guided motion calibration)
  POST /motion_calib/phase?phase=... (NEW v7)
  POST /motion_calib/finish          (NEW v7)
  PATCH /debug/integrator            (legacy tuning shims — still work)
  PATCH /debug/integrator_v7         (NEW v7: smoothness / gait tuning)
  POST /debug/recalibrate_forward
  POST /debug/flip_forward

NOTE: MAC-to-sensor assignment is no longer needed.
The ESP32 firmware (main.cpp) embeds the sensor location in every BLE packet
via the LOC field ("LOCCHEST", "LOCL_THIGH", etc.) and in the device name
("R2P-CHEST-A3F2"). ble_receiver.py maps LOC strings to internal sensor IDs
automatically via LOC_TO_SENSOR_ID. No manual assignment is required.

v2 FIX: on_sensor_packet now passes packet.accel to bio_engine.update()
when sensor_id == "pelvis". BiomechanicsEngine._compute_sway() uses this
as a real-time sway proxy before SkeletonEngine is calibrated.

v3 FIX: RootIntegrator added. Pelvis accelerometer is processed (step
dead-reckoning) and fed into SkeletonEngine.update_root_translation() so
the 3D skeleton visibly moves through world space, not just rotates in place.

v4 CHANGE (root_integrator v7): the integrator now emits a 3-tuple
(dx, dz, dy) with smooth velocity-glide locomotion (NBA2K-style continuous
motion), walk/jog/run gait grading, omnidirectional walk, and jump/squat
vertical primitives. on_sensor_packet unpacks the 3-tuple and forwards dy to
SkeletonEngine.update_root_translation(dx, dz, dy). A motion-calibration flow
(begin/phase/finish) drives the integrator's per-user CalibrationCapture so
detection thresholds adapt to the athlete's body. skeleton.py and
biomechanics.py are UNCHANGED: vertical root motion is invisible to the XZ
sway metric, and update_root_translation already composes dy.
"""

from __future__ import annotations

import asyncio
import base64
import json
import shutil
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import BackgroundTasks, FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from typing import Optional

from ble_receiver import BLEReceiver, SensorPacket
from camera_stream import CameraManager
from pose_detector import PoseDetector
from skeleton import SkeletonEngine
from biomechanics import BiomechanicsEngine
from root_integrator import RootIntegrator          # ← v7
from action_classifier import ActionClassifier       # ← v7.2 action recognition
from sports.running import RunningEngine
from test_protocols import PROTOCOLS, get_protocol, can_run, get_missing_sensors
from gvhmr_calibration import run_calibration_video, run_annotated_video
from profile_manager import ProfileManager

TEMP3_DIR = Path(__file__).resolve().parents[2] / "temp3"
if str(TEMP3_DIR) not in sys.path:
    sys.path.insert(0, str(TEMP3_DIR))

from body_tracking import get_yolo26_keypoints as temp3_get_yolo26_keypoints
from core.state_machine import StateMachineFSM

# Global profile manager — loads last active profile on startup
profile_mgr = ProfileManager()


# ─────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.limb_lengths: Optional[dict] = None
        self.skeleton_engine: Optional[SkeletonEngine]    = None
        self.bio_engine: Optional[BiomechanicsEngine] = None
        self.running_engine: Optional[RunningEngine]      = None
        self.root_integrator: Optional[RootIntegrator]    = None   # ← v7
        self.action_classifier: Optional[ActionClassifier] = None  # ← v7.2
        self.motion_mode: str = "test"   # "test" (pelvis bounce) | "user" (full)

        self.active_test_id: Optional[str]  = None
        self.session_id: Optional[str]  = None
        self.session_frames: list[dict]  = []
        self.vertical_events: list[dict] = []
        self.recording: bool             = False
        self.connected_sensors: set[str] = set()

        self.pose_detector: Optional[PoseDetector] = None
        self.camera_manager: CameraManager       = CameraManager()
        self.calibration_done: bool              = False
        self.pose_mode: str                      = "calibration"


app_state = AppState()


# ─────────────────────────────────────────────
# WS MANAGERS
# ─────────────────────────────────────────────

class WSManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        msg  = json.dumps(data, default=_json_default)
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


imu_ws    = WSManager()
camera_ws = WSManager()


def _json_default(obj):
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.integer):  return int(obj)
    raise TypeError(type(obj))


def _profile_summary() -> dict:
    p = profile_mgr.current
    return {
        "id":         p["id"],
        "name":       p["name"],
        "calibrated": p["limb_lengths"]["thigh_l"] is not None,
        "body":       p["body"],
        "derived":    p["derived"],
    }


_event_loop: Optional[asyncio.AbstractEventLoop] = None


# ─────────────────────────────────────────────
# MOTION EVENT CALLBACK  (v7)
# ─────────────────────────────────────────────

def _on_vertical_event(ev: dict):
    """
    Called by RootIntegrator when a jump / squat / squat_cancelled is detected.
    ev["type"] in {"jump", "squat", "squat_cancelled"} plus measured fields
    (est_height, depth, flight_time, …). Forwarded to the frontend so the
    coaching feed / HUD can react. A consumer that honours "squat_cancelled"
    nets to the correct squat count (a jump's load-phase crouch is retracted).
    """
    if app_state.recording:
        app_state.vertical_events.append(dict(ev))

    if _event_loop is not None:
        asyncio.run_coroutine_threadsafe(
            imu_ws.broadcast({"type": "motion_event", "event": ev}),
            _event_loop,
        )


# ─────────────────────────────────────────────
# BLE CALLBACK
# ─────────────────────────────────────────────

def on_sensor_packet(packet: SensorPacket):
    """
    Called by BLEReceiver for every parsed IMU packet.
    packet.sensor_id is already resolved from the LOC field in the firmware
    packet — no MAC-to-sensor mapping needed.

    Pelvis packet processing:
      1. skeleton_engine.update()          — updates rotation quaternion
      2. root_integrator.update()          — velocity-glide → (dx, dz, dy)
      3. skeleton_engine.update_root_translation(dx, dz, dy) — moves skeleton
         in world space (horizontal walk + vertical jump/squat)
      4. bio_engine.update() with raw_accel — sway metrics (unchanged from v2)
    """
    if app_state.skeleton_engine is None:
        return

    app_state.skeleton_engine.update(packet.sensor_id, packet.quaternion, packet.timestamp)
    app_state.connected_sensors.add(packet.sensor_id)

    # ── Action classifier: feed EVERY sensor (multi-IMU biomechanical fusion) ─
    if app_state.action_classifier is not None:
        app_state.action_classifier.update_sensor(
            packet.sensor_id, packet.quaternion, packet.accel, packet.gyro,
            packet.timestamp,
        )

    # ── Root translation (pelvis sensor only) ────────────────────────────────
    # v7: update() returns (dx, dz, dy). dy carries jump/squat vertical motion.
    if packet.sensor_id == "pelvis" and app_state.root_integrator is not None:
        dx, dz, dy = app_state.root_integrator.update(packet.quaternion, packet.accel)
        app_state.skeleton_engine.update_root_translation(dx=dx, dz=dz, dy=dy)

    # ── Multi-IMU translation gate (leg sensors) ─────────────────────────────
    # Feed thigh gyros so the integrator can confirm REAL locomotion (anti-phase
    # leg swing) versus the pelvis just bobbing in place. With no leg sensors
    # connected this is simply never called and the gate falls back to
    # pelvis-only behaviour (flagged translation_confirmed=False).
    elif packet.sensor_id in ("thigh_l", "thigh_r") and app_state.root_integrator is not None:
        app_state.root_integrator.update_limb(packet.sensor_id, packet.gyro)

    if not app_state.skeleton_engine.is_ready():
        return

    joints    = app_state.skeleton_engine.get_joints()
    active    = app_state.skeleton_engine.get_active_sensors()
    estimated = getattr(app_state.skeleton_engine, '_estimated', {})

    # ── Compute metrics ──────────────────────────────────────────────────────
    # Read translation speed AND gait once so they can be passed to bio_engine
    # AND stored in the frame.  BiomechanicsEngine owns the penalty logic so
    # there is no duplication between main.py and biomechanics.py.
    trans_speed: float = (
        app_state.root_integrator.translation_speed_ms
        if app_state.root_integrator is not None else 0.0
    )
    gait: str = (
        app_state.root_integrator.gait
        if app_state.root_integrator is not None else "idle"
    )

    # Run the action classifier once per pelvis frame (it has just been fed all
    # currently-arrived sensors for this cycle).
    action_state: dict = {}
    if app_state.action_classifier is not None and packet.sensor_id == "pelvis":
        action_state = app_state.action_classifier.tick(packet.timestamp).to_dict()

    if app_state.active_test_id == "running" and app_state.running_engine:
        metrics = app_state.running_engine.update(joints)
        metrics["translation_speed_ms"] = round(trans_speed, 4)

    elif app_state.bio_engine:
        raw_accel: Optional[np.ndarray] = (
            packet.accel if packet.sensor_id == "pelvis" else None
        )
        # translation_speed_ms is forwarded so BiomechanicsEngine applies
        # the locomotion penalty internally — both live metrics AND the
        # session summary reflect the adjusted stability score.
        metrics = app_state.bio_engine.update(
            joints,
            packet.timestamp,
            raw_accel=raw_accel,
            translation_speed_ms=trans_speed,
        )

    else:
        metrics = {"translation_speed_ms": round(trans_speed, 4)}

    # Include root position in frame for frontend HUD / replay
    root_pos = app_state.skeleton_engine.get_root_position()

    frame = {
        "type":      "frame",
        "ts":        packet.timestamp,
        "joints":    joints,
        "estimated": estimated,
        "metrics":   metrics,
        "sensors":   list(app_state.connected_sensors),
        "active":    active,
        "test_id":   app_state.active_test_id,
        "root_pos":  root_pos,                          # [x, y, z]
        "gait":      gait,                              # ← v7: idle/walk/jog/run
        "direction": (                                  # ← v7.1: heading/turn/action/confidence
            app_state.root_integrator.get_direction_state()
            if app_state.root_integrator is not None else {}
        ),
        "action":    action_state,                      # ← v7.2: full action-recognition output
        "motion_mode": app_state.motion_mode,
        "raw": {
            packet.sensor_id: {
                "accel":       packet.accel.tolist(),
                "gyro":        packet.gyro.tolist(),
                "quat":        packet.quaternion.tolist(),
                "mac":         packet.mac,
                "device_name": packet.device_name,
                "loc_raw":     packet.loc_raw,
            }
        },
    }

    if app_state.recording:
        app_state.session_frames.append(frame)

    asyncio.run_coroutine_threadsafe(imu_ws.broadcast(frame), _event_loop)


# ─────────────────────────────────────────────
# LIFESPAN
# ─────────────────────────────────────────────

_ble_receiver: Optional[BLEReceiver] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _event_loop, _ble_receiver
    _event_loop = asyncio.get_running_loop()

    app_state.skeleton_engine  = SkeletonEngine(limb_lengths=None)
    app_state.bio_engine       = BiomechanicsEngine(fps=50)
    app_state.running_engine   = RunningEngine(fps=50)
    # v7: construct with the motion-event callback so jumps/squats reach the UI
    app_state.root_integrator  = RootIntegrator(fps=50, on_event=_on_vertical_event)
    app_state.action_classifier = ActionClassifier(
        fps=50, mode=app_state.motion_mode, enable_experimental=False
    )
    app_state.root_integrator.set_translation_mode(app_state.motion_mode)

    # Restore calibration from last saved profile if available
    saved_ll = profile_mgr.get_limb_lengths()
    if saved_ll:
        app_state.skeleton_engine.set_limb_lengths(saved_ll)
        app_state.limb_lengths = saved_ll
        print(f"Loaded limb lengths from profile: {profile_mgr.current['name']}")

    _ble_receiver = BLEReceiver(on_packet=on_sensor_packet)
    asyncio.create_task(_ble_receiver.run())
    asyncio.create_task(_camera_loop())

    print("R2P backend ready.")
    yield

    if _ble_receiver:
        await _ble_receiver.stop()
    await app_state.camera_manager.stop()


# ─────────────────────────────────────────────
# CAMERA PROCESSING LOOP
# ─────────────────────────────────────────────

async def _camera_loop():
    imu_zeroed_broadcast = False

    while True:
        try:
            frame = await asyncio.wait_for(
                app_state.camera_manager.frame_queue.get(), timeout=0.5
            )
        except asyncio.TimeoutError:
            continue

        detector = app_state.pose_detector
        if detector is None:
            imu_zeroed_broadcast = False
            continue

        result = detector.process_frame(frame)

        annotated_b64 = ""
        if result.frame_annotated is not None:
            small = CameraManager.resize_for_stream(result.frame_annotated, 480)
            annotated_b64 = CameraManager.frame_to_jpeg_b64(small, quality=65)

        just_zeroed = (
            detector._imu_zeroed
            and not imu_zeroed_broadcast
            and result.state.value == "recording"
        )
        if just_zeroed:
            imu_zeroed_broadcast = True
            await imu_ws.broadcast({
                "type": "imu_zeroed",
                "msg":  "IMU zero-reference captured — sensors initialised.",
            })

        await camera_ws.broadcast({
            "type":           "pose_frame",
            "state":          result.state.value,
            "corrections":    result.corrections,
            "hold_progress":  round(result.hold_progress, 3),
            "record_seconds": round(result.record_seconds, 2),
            "is_complete":    result.is_complete,
            "imu_zeroed":     detector._imu_zeroed,
            "frame_b64":      annotated_b64,
        })

        if result.is_complete and not app_state.calibration_done and app_state.pose_mode == "calibration":
            app_state.calibration_done = True
            imu_zeroed_broadcast = False
            asyncio.create_task(
                _process_calibration_frames(detector.get_recorded_frames())
            )


async def _process_calibration_frames(frames: list):
    if not frames:
        return
    out_path = str(OUTPUTS_DIR / "calibration_temp.mp4")
    h, w = frames[0].shape[:2]
    writer = None
    for codec in ("avc1", "H264", "XVID", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(out_path, fourcc, 30.0, (w, h))
        if writer.isOpened():
            break
        writer.release(); writer = None
    if writer:
        for f in frames:
            writer.write(f)
        writer.release()

    await imu_ws.broadcast({"type": "calibration_processing", "msg": "Running GVHMR…"})
    try:
        limb_lengths = await asyncio.to_thread(run_calibration_video, out_path)
    except Exception as exc:
        await imu_ws.broadcast({"type": "calibration_error", "msg": str(exc)})
        return

    if limb_lengths:
        app_state.limb_lengths = limb_lengths
        app_state.skeleton_engine.set_limb_lengths(limb_lengths)

        imu_offsets = {
            sid: app_state.skeleton_engine._calibration_offsets[sid].tolist()
            for sid in app_state.skeleton_engine.SENSOR_IDS
            if app_state.skeleton_engine._calibrated
        }
        profile_mgr.save_calibration(limb_lengths, imu_offsets or None)

        await imu_ws.broadcast({
            "type":         "calibrated",
            "limb_lengths": limb_lengths,
            "profile":      _profile_summary(),
        })
    else:
        await imu_ws.broadcast({"type": "calibration_error", "msg": "GVHMR returned no data."})


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

app = FastAPI(title="R2P Biomechanics API v4", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
OUTPUTS_DIR  = Path(__file__).parent.parent / "outputs"
TEMP3_FRONTEND_DIR = Path(__file__).resolve().parents[3] / "Fronted_newest" / "Fronted"
TEMP3_UPLOAD_DIR = TEMP3_DIR / "uploads"
TEMP3_OUTPUT_DIR = TEMP3_DIR / "outputs"
TEMP3_USER_DATA_PATHS = (
    TEMP3_FRONTEND_DIR / "111" / "personaldata_storage" / "user_data.json",
    TEMP3_FRONTEND_DIR / "personaldata_storage" / "user_data.json",
)
OUTPUTS_DIR.mkdir(exist_ok=True)
TEMP3_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TEMP3_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
app.mount("/videos", StaticFiles(directory=str(TEMP3_OUTPUT_DIR)), name="temp3_videos")


# Temp3 frontend compatibility: upload jobs, account JSON, and form-check FSM.
temp3_gpu_semaphore = asyncio.Semaphore(1)
temp3_jobs: dict[str, dict] = {}
temp3_imu_jobs: dict[str, dict] = {}
temp3_imu_job_lock = asyncio.Lock()
_temp3_run_analysis = None


def _frame_to_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def _pad_frame_to_even_dimensions(frame: np.ndarray) -> np.ndarray:
    frame = _frame_to_bgr(frame)
    h, w = frame.shape[:2]
    pad_bottom = h % 2
    pad_right = w % 2
    if pad_bottom or pad_right:
        frame = cv2.copyMakeBorder(
            frame,
            0,
            pad_bottom,
            0,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
    return frame


def _fit_frame_inside_canvas(frame: np.ndarray, canvas_w: int, canvas_h: int) -> np.ndarray:
    frame = _frame_to_bgr(frame)
    h, w = frame.shape[:2]
    if w == canvas_w and h == canvas_h:
        return frame

    scale = min(canvas_w / max(1, w), canvas_h / max(1, h))
    resized_w = max(1, min(canvas_w, int(round(w * scale))))
    resized_h = max(1, min(canvas_h, int(round(h * scale))))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    x = (canvas_w - resized_w) // 2
    y = (canvas_h - resized_h) // 2
    canvas[y:y + resized_h, x:x + resized_w] = resized
    return canvas


def _open_video_writer(output_path: Path, fps: float, frame_size: tuple[int, int]):
    for codec in ("avc1", "H264", "mp4v", "XVID"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
        if writer.isOpened():
            return writer
        writer.release()
    return None


def _normalize_calibration_video_for_gvhmr(input_path: Path, task_id: str) -> Path:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return input_path

    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    output_path = TEMP3_UPLOAD_DIR / f"{task_id}_calibration_normalized.mp4"

    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        cap.release()
        return input_path

    first_frame = _pad_frame_to_even_dimensions(first_frame)
    canvas_h, canvas_w = first_frame.shape[:2]
    writer = _open_video_writer(output_path, fps, (canvas_w, canvas_h))
    if writer is None:
        cap.release()
        return input_path

    writer.write(first_frame)
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame = _fit_frame_inside_canvas(frame, canvas_w, canvas_h)
        writer.write(frame)

    cap.release()
    writer.release()

    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    return input_path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _temp3_user_data_path() -> Path:
    for path in TEMP3_USER_DATA_PATHS:
        if path.exists():
            return path
    return TEMP3_USER_DATA_PATHS[0]


def _read_temp3_user_data() -> dict:
    path = _temp3_user_data_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        data = {"users": []}
    except json.JSONDecodeError:
        data = {"users": []}
    if not isinstance(data, dict):
        data = {"users": []}
    data.setdefault("users", [])
    return data


def _write_temp3_user_data(data: dict) -> None:
    path = _temp3_user_data_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data["last_updated"] = _now_iso()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _find_temp3_user(
    data: dict,
    user_id: Optional[str] = None,
    username: Optional[str] = None,
    email: Optional[str] = None,
) -> Optional[dict]:
    users = data.get("users") or []
    for user in users:
        if user_id and str(user.get("id")) == str(user_id):
            return user
        if email and user.get("email") == email:
            return user
        if username and user.get("username") == username:
            return user
    return None


def _extract_user_limb_lengths(user: Optional[dict]) -> Optional[dict]:
    if not user:
        return None
    candidates = [
        user.get("limb_lengths"),
        (user.get("limb_length_calibration") or {}).get("limb_lengths"),
        (user.get("backend_video_data") or {}).get("limb_lengths"),
    ]
    for value in candidates:
        if isinstance(value, dict) and value:
            return {k: float(v) for k, v in value.items() if v is not None}
    return None


def _apply_limb_lengths(limb_lengths: Optional[dict]) -> None:
    if not limb_lengths:
        return
    app_state.limb_lengths = limb_lengths
    if app_state.skeleton_engine:
        app_state.skeleton_engine.set_limb_lengths(limb_lengths)


def _apply_temp3_user_limb_lengths(
    user_id: Optional[str] = None,
    username: Optional[str] = None,
    email: Optional[str] = None,
) -> Optional[dict]:
    data = _read_temp3_user_data()
    user = _find_temp3_user(data, user_id=user_id, username=username, email=email)
    limb_lengths = _extract_user_limb_lengths(user)
    _apply_limb_lengths(limb_lengths)
    return limb_lengths


def _save_temp3_user_calibration(
    limb_lengths: dict,
    user_id: Optional[str] = None,
    username: Optional[str] = None,
    email: Optional[str] = None,
) -> Optional[dict]:
    data = _read_temp3_user_data()
    user = _find_temp3_user(data, user_id=user_id, username=username, email=email)
    if user is None:
        user = {
            "id": user_id or str(uuid.uuid4()),
            "username": username or "",
            "email": email or "",
            "personal_info": {},
            "upload_records": [],
            "backend_video_data": {},
            "created_at": _now_iso(),
        }
        data["users"].append(user)

    calibrated_at = _now_iso()
    user["limb_lengths"] = limb_lengths
    user["limb_length_calibration"] = {
        "limb_lengths": limb_lengths,
        "calibrated_at": calibrated_at,
        "source": "full_body_video",
    }
    backend_video_data = user.setdefault("backend_video_data", {})
    backend_video_data["limb_lengths"] = limb_lengths
    backend_video_data["limb_lengths_calibrated_at"] = calibrated_at
    user["calibration_video_done"] = True
    user["updated_at"] = calibrated_at

    _write_temp3_user_data(data)
    return user


def _temp3_protocol_for_test(test_type: str) -> str:
    value = (test_type or "").strip().lower()
    if value in {"cmj", "jump", "countermovement-jump", "countermovement_jump"}:
        return "cmj"
    if value in {"sls", "single-leg-stand", "single_leg_stand", "squat", "balance", "assessment"}:
        return "stability"
    return "stability"


def _summarize_temp3_imu_result(task_id: str, protocol_id: str, summary: dict) -> dict:
    result = {
        "status": "completed",
        "task_id": task_id,
        "session_id": task_id,
        "exercise_name": "CMJ" if protocol_id == "cmj" else "Balance",
        "test_id": protocol_id,
        "progress": 100,
        "result_video": "",
        "summary": summary,
        "limb_lengths_used": app_state.limb_lengths,
    }

    if protocol_id == "cmj":
        jumps = [ev for ev in app_state.vertical_events if ev.get("type") == "jump"]
        flight_times = [float(ev.get("flight_time", 0) or 0) for ev in jumps]
        heights_m = [float(ev.get("est_height", 0) or 0) for ev in jumps]
        flight_time = max(flight_times) if flight_times else 0.0
        jump_height_m = max(heights_m) if heights_m else 0.0
        contact_time = max(0.001, float(summary.get("duration_s") or 0) / max(1, len(jumps))) if jumps else 0.001
        result.update({
            "flight_time": round(flight_time, 3),
            "jump_height_cm": round(jump_height_m * 100, 1),
            "rsi": round(jump_height_m / contact_time, 2) if jump_height_m else 0.0,
            "jump_count": len(jumps),
        })
    else:
        sway = summary.get("sway") or {}
        result.update({
            "cv": float(sway.get("cv_pct") or 0.0),
            "stability_score": float(sway.get("stability_score") or 0.0),
        })

    return result


async def _prepare_temp3_imu_session(task_id: str, protocol_id: str) -> None:
    proto = get_protocol(protocol_id)
    app_state.session_id = task_id
    app_state.session_frames = []
    app_state.vertical_events = []
    app_state.active_test_id = protocol_id
    app_state.recording = True

    if app_state.bio_engine:
        app_state.bio_engine.test_id = protocol_id
        app_state.bio_engine.reset()
    if app_state.running_engine:
        app_state.running_engine.reset()
    if app_state.root_integrator:
        app_state.root_integrator.reset()
    if app_state.action_classifier:
        app_state.action_classifier.reset()
    if app_state.skeleton_engine:
        app_state.skeleton_engine.reset_root_translation()

    await imu_ws.broadcast({
        "type": "session_started",
        "session_id": task_id,
        "test_id": protocol_id,
        "protocol": {
            "name": proto.name,
            "instructions": proto.instructions,
            "duration": proto.record_duration,
        },
    })


async def _finish_temp3_imu_session(task_id: str, protocol_id: str) -> dict:
    app_state.recording = False
    summary = (
        app_state.running_engine.get_summary()
        if protocol_id == "running" and app_state.running_engine
        else app_state.bio_engine.get_summary()
        if app_state.bio_engine
        else {}
    )
    summary["test_id"] = protocol_id
    summary["session_id"] = task_id
    summary["vertical_events"] = app_state.vertical_events

    (OUTPUTS_DIR / f"{task_id}_session.json").write_text(
        json.dumps({
            "session_id": task_id,
            "test_id": protocol_id,
            "frames": app_state.session_frames,
            "summary": summary,
        }, default=_json_default)
    )
    profile_mgr.save_test_result(task_id, protocol_id, summary)
    await imu_ws.broadcast({"type": "session_stopped", "session_id": task_id})

    app_state.active_test_id = None
    app_state.pose_detector = None
    return _summarize_temp3_imu_result(task_id, protocol_id, summary)


async def _run_temp3_imu_test_job(task_id: str, protocol_id: str, duration_seconds: float):
    async with temp3_imu_job_lock:
        try:
            temp3_imu_jobs[task_id].update({
                "status": "processing",
                "progress": 0,
            })
            await _prepare_temp3_imu_session(task_id, protocol_id)
            loop = asyncio.get_running_loop()
            started = loop.time()
            while True:
                elapsed = loop.time() - started
                progress = min(95, int((elapsed / max(duration_seconds, 1.0)) * 95))
                temp3_imu_jobs[task_id]["progress"] = progress
                if elapsed >= duration_seconds:
                    break
                await asyncio.sleep(1)

            temp3_imu_jobs[task_id] = await _finish_temp3_imu_session(task_id, protocol_id)
        except Exception as exc:
            print(f"Error processing temp3 IMU task {task_id}: {exc}")
            temp3_imu_jobs[task_id] = {
                "status": "failed",
                "task_id": task_id,
                "test_id": protocol_id,
                "progress": 100,
                "error": str(exc),
            }
        finally:
            if app_state.session_id == task_id:
                app_state.recording = False
                app_state.active_test_id = None
                app_state.pose_detector = None


def _get_temp3_run_analysis():
    global _temp3_run_analysis
    if _temp3_run_analysis is None:
        try:
            from engine import run_analysis
        except ModuleNotFoundError as exc:
            if exc.name in {"paramiko", "scp"}:
                raise RuntimeError(
                    "Temp3 GVHMR upload processing needs the SSH dependencies "
                    "from motion_track/temp5/backend/requirements.txt "
                    f"({exc.name} is missing)."
                ) from exc
            raise
        _temp3_run_analysis = run_analysis
    return _temp3_run_analysis


async def _analyze_temp3_upload(task_id: str, file_path: str, exercise_name: str, f_mm: Optional[int]):
    async with temp3_gpu_semaphore:
        try:
            run_analysis = _get_temp3_run_analysis()
            temp3_jobs[task_id] = {
                "status": "processing",
                "exercise_name": exercise_name,
            }
            loop = asyncio.get_running_loop()
            out_path = await loop.run_in_executor(
                None,
                partial(run_analysis, file_path, task_id, exercise_name, f_mm=f_mm),
            )

            metrics_path = TEMP3_OUTPUT_DIR / f"{task_id}_metrics.csv"
            result_data = {
                "status": "completed",
                "exercise_name": exercise_name,
                "result_video": f"/videos/{task_id}_annotated.webm",
                "output_path": str(out_path),
            }

            if metrics_path.exists():
                import csv

                with metrics_path.open("r", newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, [])
                    row = next(reader, [])
                if "rsi" in header and len(row) >= 3:
                    result_data["rsi"] = float(row[1])
                    result_data["flight_time"] = float(row[2])
                elif "cv" in header and len(row) >= 2:
                    result_data["cv"] = float(row[1])

            temp3_jobs[task_id] = result_data
        except Exception as exc:
            print(f"Error processing temp3 frontend task {task_id}: {exc}")
            temp3_jobs[task_id] = {
                "status": "failed",
                "exercise_name": exercise_name,
                "error": str(exc),
            }


# ─────────────────────────────────────────────
# PAGE ROUTES
# ─────────────────────────────────────────────

@app.get("/")
async def root():            return RedirectResponse(url="/111/auth/app_login_register.html")
@app.get("/select")
async def select_page():     return FileResponse(str(FRONTEND_DIR / "select.html"))
@app.get("/test")
async def test_page():       return FileResponse(str(FRONTEND_DIR / "index.html"))
@app.get("/review")
async def review_page():     return FileResponse(str(FRONTEND_DIR / "review.html"))
@app.get("/settings")
async def settings_page():   return FileResponse(str(FRONTEND_DIR / "settings.html"))


# Temp3 frontend compatibility routes.
@app.post("/upload")
async def upload_video_for_temp3_frontend(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    exercise_name: str = Form("Balance"),
    leg: Optional[str] = Form(None),
    f_mm: Optional[int] = Form(None),
):
    task_id = str(uuid.uuid4())
    safe_name = Path(video.filename or "user_recording.mp4").name
    file_path = TEMP3_UPLOAD_DIR / f"{task_id}_{safe_name}"

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    temp3_jobs[task_id] = {
        "status": "queued",
        "exercise_name": exercise_name,
        "leg": leg,
    }
    background_tasks.add_task(
        _analyze_temp3_upload,
        task_id,
        str(file_path),
        exercise_name,
        f_mm,
    )

    return {
        "task_id": task_id,
        "message": "Video uploaded successfully. Processing in background.",
    }


@app.post("/api/calibrate_limb_lengths")
async def calibrate_limb_lengths_for_temp3_frontend(
    video: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
    username: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
):
    task_id = str(uuid.uuid4())
    safe_name = Path(video.filename or "full_body_calibration.webm").name
    file_path = TEMP3_UPLOAD_DIR / f"{task_id}_calibration_{safe_name}"

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    try:
        normalized_path = await asyncio.to_thread(
            _normalize_calibration_video_for_gvhmr,
            file_path,
            task_id,
        )
        limb_lengths = await asyncio.to_thread(run_calibration_video, str(normalized_path))
    except Exception as exc:
        raise HTTPException(500, f"GVHMR calibration failed: {exc}") from exc

    if not limb_lengths:
        raise HTTPException(502, "GVHMR returned no limb lengths.")

    limb_lengths = {k: float(v) for k, v in limb_lengths.items() if v is not None}
    _apply_limb_lengths(limb_lengths)
    profile_mgr.save_calibration(limb_lengths)
    updated_user = _save_temp3_user_calibration(
        limb_lengths,
        user_id=user_id,
        username=username,
        email=email,
    )

    await imu_ws.broadcast({
        "type": "calibrated",
        "limb_lengths": limb_lengths,
        "profile": _profile_summary(),
    })

    return {
        "status": "ok",
        "task_id": task_id,
        "limb_lengths": limb_lengths,
        "user": updated_user,
        "profile": _profile_summary(),
    }


@app.post("/api/imu-test/start")
async def start_temp3_imu_test(
    request: Request,
    background_tasks: BackgroundTasks,
    test_type: str = Query("assessment"),
    leg: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    username: Optional[str] = Query(None),
    email: Optional[str] = Query(None),
    duration_seconds: Optional[float] = Query(None),
):
    payload = {}
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            payload = await request.json()
        except Exception:
            payload = {}

    test_type = payload.get("test_type") or test_type
    leg = payload.get("leg") or leg
    user_id = payload.get("user_id") or user_id
    username = payload.get("username") or username
    email = payload.get("email") or email
    duration_seconds = payload.get("duration_seconds") or duration_seconds

    protocol_id = _temp3_protocol_for_test(test_type)
    proto = get_protocol(protocol_id)
    duration = float(duration_seconds or proto.record_duration)
    task_id = str(uuid.uuid4())
    limb_lengths = _apply_temp3_user_limb_lengths(
        user_id=user_id,
        username=username,
        email=email,
    )

    temp3_imu_jobs[task_id] = {
        "status": "queued",
        "task_id": task_id,
        "test_id": protocol_id,
        "test_type": test_type,
        "leg": leg,
        "duration_seconds": duration,
        "progress": 0,
        "limb_lengths_used": limb_lengths,
    }
    background_tasks.add_task(_run_temp3_imu_test_job, task_id, protocol_id, duration)
    return {
        "status": "queued",
        "task_id": task_id,
        "test_id": protocol_id,
        "duration_seconds": duration,
        "limb_lengths_used": limb_lengths,
    }


@app.get("/api/imu-test/status/{task_id}")
async def get_temp3_imu_test_status(task_id: str):
    return temp3_imu_jobs.get(task_id, {"status": "not_found"})


@app.get("/status/{task_id}")
async def get_temp3_upload_status(task_id: str):
    if task_id in temp3_jobs:
        return temp3_jobs[task_id]
    if task_id in temp3_imu_jobs:
        return temp3_imu_jobs[task_id]
    return {"status": "not_found"}


@app.get("/batch_status")
async def get_temp3_batch_status(task_ids: str):
    ids = [tid for tid in task_ids.split(",") if tid]
    results = {
        tid: temp3_jobs.get(tid) or temp3_imu_jobs.get(tid) or {"status": "not_found"}
        for tid in ids
    }
    return {
        "all_completed": all(v.get("status") == "completed" for v in results.values()),
        "any_failed": any(v.get("status") == "failed" for v in results.values()),
        "tasks": results,
    }


@app.get("/api/get_user_data")
async def get_temp3_user_data():
    path = _temp3_user_data_path()
    try:
        return JSONResponse(content=json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return {"users": []}


@app.post("/api/update_user_data")
async def update_temp3_user_data(request: Request):
    data = await request.json()
    path = _temp3_user_data_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"success": True, "message": "data updated"}


@app.get("/api/user-videos")
async def get_temp3_user_videos():
    completed = [
        (task_id, job) for task_id, job in temp3_jobs.items()
        if job.get("status") == "completed"
    ]
    completed.reverse()
    videos = {}
    for task_id, job in completed:
        item = {
            "url": job.get("result_video"),
            "timestamp": task_id,
            "exercise_name": job.get("exercise_name"),
        }
        exercise = (job.get("exercise_name") or "").lower()
        if exercise == "cmj" and "verticalJump" not in videos:
            videos["verticalJump"] = item
        elif exercise in {"balance", "sls"} and "singleLegSquat" not in videos:
            videos["singleLegSquat"] = item
    return videos


@app.websocket("/ws/form-check/{exercise}")
async def websocket_temp3_form_check(websocket: WebSocket, exercise: str):
    await websocket.accept()
    fsm = StateMachineFSM(exercise)

    try:
        while True:
            data = await websocket.receive_text()
            try:
                encoded = data.split(",", 1)[1] if "," in data else data
                nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                keypoints_2d_raw = temp3_get_yolo26_keypoints(frame)
                keypoints = [(pt[0], pt[1]) if pt else None for pt in keypoints_2d_raw]
                status, message, current_state = fsm.process_frame(keypoints)

                await websocket.send_json({
                    "status": status,
                    "state": current_state,
                    "hint": message,
                })
            except Exception as exc:
                print(f"Temp3 form-check frame failed: {exc}")
    except WebSocketDisconnect:
        print("Temp3 form-check WebSocket client disconnected")


# ─────────────────────────────────────────────
# WEBSOCKETS
# ─────────────────────────────────────────────

@app.websocket("/ws/imu")
async def ws_imu(ws: WebSocket):
    await imu_ws.connect(ws)
    await ws.send_text(json.dumps({
        "type":      "init",
        "calibrated": app_state.limb_lengths is not None,
        "recording": app_state.recording,
        "sensors":   list(app_state.connected_sensors),
        "test_id":   app_state.active_test_id,
    }))
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        imu_ws.disconnect(ws)


@app.websocket("/ws/camera")
async def ws_camera(ws: WebSocket):
    await camera_ws.connect(ws)
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            if msg.get("type") == "frame":
                await app_state.camera_manager.push_browser_frame(msg["data"])
            elif msg.get("type") == "start_webcam":
                await app_state.camera_manager.start_webcam(device_index=0)
                await ws.send_text(json.dumps({"type": "webcam_started"}))
            elif msg.get("type") == "stop_camera":
                await app_state.camera_manager.stop()
    except WebSocketDisconnect:
        camera_ws.disconnect(ws)


# ─────────────────────────────────────────────
# POSE DETECTOR
# ─────────────────────────────────────────────

@app.post("/pose/start")
async def start_pose(
    mode: str = Query("calibration"),
    record_seconds: float = Query(5.0),
    hold_seconds: float   = Query(3.0),
):
    if mode in PROTOCOLS:
        proto = get_protocol(mode)
        record_seconds = proto.record_duration
        hold_seconds   = proto.hold_duration

    app_state.calibration_done = False
    app_state.pose_mode = mode
    imu_zero_cb = app_state.skeleton_engine.calibrate_tpose if mode == "calibration" else None

    app_state.pose_detector = PoseDetector(
        required_seconds=record_seconds,
        hold_seconds=hold_seconds,
        fps=30.0,
        on_imu_zero=imu_zero_cb,
    )
    return {"status": "ok", "mode": mode, "record_seconds": record_seconds}


@app.post("/pose/reset")
async def reset_pose():
    if app_state.pose_detector:
        app_state.pose_detector.reset()
    app_state.calibration_done = False
    return {"status": "reset"}


# ─────────────────────────────────────────────
# SESSION
# ─────────────────────────────────────────────

@app.post("/session/start")
async def start_session(test_id: str = Query(...)):
    if test_id not in PROTOCOLS:
        raise HTTPException(400, f"Unknown test: {test_id}")
    proto = get_protocol(test_id)

    active  = app_state.skeleton_engine.get_active_sensors() if app_state.skeleton_engine else []
    missing = get_missing_sensors(test_id, active)
    if missing:
        raise HTTPException(409, {
            "error":   "missing_sensors",
            "missing": missing,
            "message": f"Connect these sensors to run {proto.name}: {', '.join(missing)}"
        })

    app_state.session_id     = str(uuid.uuid4())
    app_state.session_frames = []
    app_state.vertical_events = []
    app_state.active_test_id = test_id
    app_state.recording      = True

    # Reset both engines and the root integrator so each session starts
    # at the origin with zero velocity
    if app_state.bio_engine:
        app_state.bio_engine.test_id = test_id
    (app_state.running_engine if test_id == "running" else app_state.bio_engine).reset()
    if app_state.root_integrator:
        app_state.root_integrator.reset()                    # ← resets velocity/gait too
    if app_state.action_classifier:
        app_state.action_classifier.reset()                  # ← v7.2
    if app_state.skeleton_engine:
        app_state.skeleton_engine.reset_root_translation()

    app_state.pose_detector = PoseDetector(
        required_seconds=proto.record_duration,
        hold_seconds=proto.hold_duration,
        fps=30.0,
        on_imu_zero=app_state.skeleton_engine.calibrate_tpose,
    )
    await imu_ws.broadcast({
        "type":       "session_started",
        "session_id": app_state.session_id,
        "test_id":    test_id,
        "protocol": {
            "name":         proto.name,
            "instructions": proto.instructions,
            "duration":     proto.record_duration,
        },
    })
    return {"session_id": app_state.session_id, "test_id": test_id}


@app.get("/session/check")
async def check_session_readiness(test_id: str = Query(...)):
    if test_id not in PROTOCOLS:
        raise HTTPException(404)
    proto   = get_protocol(test_id)
    active  = app_state.skeleton_engine.get_active_sensors() if app_state.skeleton_engine else []
    missing = get_missing_sensors(test_id, active)
    return {
        "test_id":          test_id,
        "can_run":          len(missing) == 0,
        "active_sensors":   active,
        "required_sensors": proto.required_sensors,
        "optional_sensors": proto.optional_sensors,
        "missing_sensors":  missing,
    }


@app.post("/session/stop")
async def stop_session():
    app_state.recording = False
    sid = app_state.session_id
    summary = (
        app_state.running_engine.get_summary()
        if app_state.active_test_id == "running"
        else app_state.bio_engine.get_summary()
        if app_state.bio_engine
        else {}
    )
    summary["test_id"]    = app_state.active_test_id
    summary["session_id"] = sid
    summary["vertical_events"] = app_state.vertical_events
    if sid:
        (OUTPUTS_DIR / f"{sid}_session.json").write_text(
            json.dumps({
                "session_id": sid,
                "test_id":    app_state.active_test_id,
                "frames":     app_state.session_frames,
                "summary":    summary,
            }, default=_json_default)
        )
        profile_mgr.save_test_result(sid, app_state.active_test_id or "unknown", summary)
    await imu_ws.broadcast({"type": "session_stopped", "session_id": sid})
    app_state.active_test_id = None
    app_state.pose_detector  = None
    return {"session_id": sid, "frames": len(app_state.session_frames)}


@app.get("/session/{session_id}/summary")
async def get_summary(session_id: str):
    p = OUTPUTS_DIR / f"{session_id}_session.json"
    if not p.exists():
        raise HTTPException(404, "Session not found")
    return JSONResponse(content=json.loads(p.read_text()).get("summary", {}))


# ─────────────────────────────────────────────
# PROTOCOLS
# ─────────────────────────────────────────────

@app.get("/protocols")
async def list_protocols():
    active = app_state.skeleton_engine.get_active_sensors() if app_state.skeleton_engine else []
    result = {}
    for pid, p in PROTOCOLS.items():
        missing = get_missing_sensors(pid, active)
        result[pid] = {
            "id": p.id, "name": p.name, "group": p.group, "sport": p.sport,
            "icon": p.icon, "description": p.description,
            "instructions": p.instructions, "record_duration": p.record_duration,
            "required_sensors": p.required_sensors,
            "optional_sensors": p.optional_sensors,
            "can_run":          len(missing) == 0,
            "missing_sensors":  missing,
        }
    return result


@app.get("/protocol/{test_id}")
async def get_proto(test_id: str):
    if test_id not in PROTOCOLS:
        raise HTTPException(404)
    p = get_protocol(test_id)
    return {
        "id": p.id, "name": p.name, "group": p.group, "icon": p.icon,
        "description": p.description, "instructions": p.instructions,
        "record_duration": p.record_duration, "hold_duration": p.hold_duration,
        "metrics": [
            {
                "key":              m.key,
                "label":            m.label,
                "unit":             m.unit,
                "good_range":       m.good_range,
                "warn_range":       m.warn_range,
                "higher_is_better": m.higher_is_better,
            }
            for m in p.metrics
        ],
    }


# ─────────────────────────────────────────────
# CALIBRATION / REVIEW
# ─────────────────────────────────────────────

@app.post("/calibrate/upload")
async def calibrate_upload(video: UploadFile = File(...)):
    tmp = OUTPUTS_DIR / f"calib_{video.filename}"
    tmp.write_bytes(await video.read())
    ll = await asyncio.to_thread(run_calibration_video, str(tmp))
    if not ll:
        raise HTTPException(500, "GVHMR returned no data")
    app_state.limb_lengths = ll
    app_state.skeleton_engine.set_limb_lengths(ll)
    await imu_ws.broadcast({"type": "calibrated", "limb_lengths": ll})
    return {"status": "ok", "limb_lengths": ll}


@app.post("/review/annotate")
async def annotate_video(video: UploadFile = File(...), session_id: str = Query("")):
    tmp = OUTPUTS_DIR / f"review_{video.filename}"
    tmp.write_bytes(await video.read())
    sm = None
    if session_id:
        p = OUTPUTS_DIR / f"{session_id}_session.json"
        if p.exists():
            sm = json.loads(p.read_text()).get("summary")
    out = await asyncio.to_thread(run_annotated_video, str(tmp), str(OUTPUTS_DIR), sm)
    return {"status": "ok", "video_path": out, "session_id": session_id}


# ─────────────────────────────────────────────
# PROFILE / SETTINGS
# ─────────────────────────────────────────────

@app.get("/profile")
async def get_current_profile():
    return JSONResponse(content=profile_mgr.current)

@app.get("/profiles")
async def list_profiles():
    return {"profiles": profile_mgr.list_profiles()}

@app.post("/profile/create")
async def create_profile(name: str = Query("New Athlete")):
    profile = profile_mgr.create_profile(name)
    app_state.limb_lengths = None
    app_state.skeleton_engine.set_limb_lengths(None)
    return profile

@app.post("/profile/switch")
async def switch_profile(profile_id: str = Query(...)):
    profile = profile_mgr.switch_profile(profile_id)
    if not profile:
        raise HTTPException(404, "Profile not found")
    saved_ll = profile_mgr.get_limb_lengths()
    if saved_ll:
        app_state.skeleton_engine.set_limb_lengths(saved_ll)
        app_state.limb_lengths = saved_ll
    else:
        app_state.skeleton_engine.set_limb_lengths(None)
        app_state.limb_lengths = None
    await imu_ws.broadcast({"type": "profile_switched", "profile": _profile_summary()})
    return profile

@app.delete("/profile/{profile_id}")
async def delete_profile(profile_id: str):
    ok = profile_mgr.delete_profile(profile_id)
    if not ok:
        raise HTTPException(404, "Profile not found")
    return {"status": "deleted"}

@app.patch("/profile/body")
async def update_body_info(data: dict):
    profile = profile_mgr.update_body_info(**data)
    return profile

@app.get("/profile/history")
async def get_test_history(test_id: Optional[str] = Query(None)):
    return {"history": profile_mgr.get_test_history(test_id)}


# ─────────────────────────────────────────────
# BLE
# ─────────────────────────────────────────────

@app.get("/ble/status")
async def ble_status():
    active = app_state.skeleton_engine.get_active_sensors() if app_state.skeleton_engine else []
    return {
        "connected_sensors":  list(app_state.connected_sensors),
        "active_sensors":     active,
        "expected_sensors":   BLEReceiver.SENSOR_IDS,
        "ready":              app_state.skeleton_engine.is_ready() if app_state.skeleton_engine else False,
        "connected_devices":  _ble_receiver.get_connected_devices() if _ble_receiver else [],
    }


@app.post("/ble/scan")
async def ble_scan():
    if not _ble_receiver:
        return {"devices": []}
    devices = await _ble_receiver.scan(timeout=5.0)
    return {"devices": devices}


@app.post("/ble/tpose")
async def ble_tpose():
    """
    Manual IMU zero-reference override.
    Also resets root integrator and skeleton root position so the
    skeleton starts at the origin after re-zeroing.
    """
    if app_state.skeleton_engine:
        app_state.skeleton_engine.calibrate_tpose()   # also calls reset_root_translation
    if app_state.root_integrator:
        app_state.root_integrator.reset()
    await imu_ws.broadcast({"type": "imu_zeroed", "msg": "IMU zero set manually."})
    return {"status": "ok"}


# ─────────────────────────────────────────────
# MOTION CALIBRATION  (v7 — guided walk-4-way / jump / squat)
# ─────────────────────────────────────────────

# Phase order the frontend should walk the user through.
MOTION_CALIB_PHASES = ["walk_forward", "walk_back", "walk_left", "walk_right", "jump", "squat"]


@app.post("/motion_calib/begin")
async def motion_calib_begin():
    """
    Start a guided motion-calibration capture session.

    Flow:
      1. POST /motion_calib/begin
      2. For each phase: POST /motion_calib/phase?phase=<name>, have the user
         perform the movement for ~4–5 s, then advance to the next phase.
      3. POST /motion_calib/finish — derives & applies per-user thresholds.
    """
    ri = app_state.root_integrator
    if ri is None:
        raise HTTPException(503, "Root integrator not initialised")
    ri.begin_calibration()
    await imu_ws.broadcast({"type": "calib_begin", "phases": MOTION_CALIB_PHASES})
    return {
        "status":  "ok",
        "phases":  MOTION_CALIB_PHASES,
        "message": "Set each phase, perform the movement, then call /motion_calib/finish.",
    }


@app.post("/motion_calib/phase")
async def motion_calib_phase(phase: str = Query(...)):
    """
    Set the active calibration phase. Valid phases:
      walk_forward, walk_back, walk_left, walk_right, jump, squat
    Call this, have the user perform the movement for a few seconds, then set
    the next phase.
    """
    ri = app_state.root_integrator
    if ri is None:
        raise HTTPException(503, "Root integrator not initialised")
    if phase not in MOTION_CALIB_PHASES:
        raise HTTPException(400, f"Unknown calibration phase: {phase}")
    try:
        ri.set_calibration_phase(phase)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    await imu_ws.broadcast({"type": "calib_phase", "phase": phase})
    return {"status": "ok", "phase": phase}


@app.post("/motion_calib/finish")
async def motion_calib_finish():
    """
    Finish calibration. Auto-derives and applies per-user thresholds
    (step / jump-launch / squat-descend) and returns the captured signatures.

    To persist across sessions, store the returned thresholds on the profile
    and re-apply them on startup.
    """
    ri = app_state.root_integrator
    if ri is None:
        raise HTTPException(503, "Root integrator not initialised")
    summary = ri.finish_calibration()
    await imu_ws.broadcast({"type": "calib_complete", "summary": summary})
    return {"status": "ok", "summary": summary}


# ─────────────────────────────────────────────
# DIRECTIONAL INITIALISATION  (v7.1 — explicit body frame)
# ─────────────────────────────────────────────

def _current_pelvis_quat():
    """Latest pelvis world quaternion [w,x,y,z], or None if not seen yet."""
    se = app_state.skeleton_engine
    if se is None:
        return None
    q = getattr(se, "_quats", {}).get("pelvis")
    return q


@app.post("/direction/set_forward")
async def direction_set_forward():
    """
    Capture the CURRENT held pelvis orientation as 'facing forward'.
    The user should stand still facing their intended forward direction, then
    call this. Locks the forward axis deterministically (no walking needed) and
    disables fragile auto-detection.
    """
    ri = app_state.root_integrator
    if ri is None:
        raise HTTPException(503, "Root integrator not initialised")
    q = _current_pelvis_quat()
    if q is None:
        raise HTTPException(409, "No pelvis IMU data yet — connect the pelvis sensor.")
    res = ri.set_forward_reference(q)
    if res.get("status") != "ok":
        raise HTTPException(400, res.get("message", "Could not set forward."))
    await imu_ws.broadcast({"type": "direction_forward_set", "forward_axis": res["forward_axis"]})
    return res


@app.post("/direction/set_right")
async def direction_set_right():
    """
    Capture a 'turned ~90° to the right' pose to confirm the yaw sign.
    Call AFTER /direction/set_forward. Removes any left/right ambiguity.
    """
    ri = app_state.root_integrator
    if ri is None:
        raise HTTPException(503, "Root integrator not initialised")
    q = _current_pelvis_quat()
    if q is None:
        raise HTTPException(409, "No pelvis IMU data yet — connect the pelvis sensor.")
    res = ri.set_right_reference(q)
    if res.get("status") != "ok":
        raise HTTPException(400, res.get("message", "Could not set right reference."))
    await imu_ws.broadcast({"type": "direction_right_set", "axis_flipped": res["axis_flipped"]})
    return res


@app.get("/direction/state")
async def direction_state():
    """
    Live directional + action readout for the 3D model / HUD:
    heading_deg, turn_rate_dps, turn_dir (cw/ccw/none), move_dir
    (forward/backward/strafe_left/strafe_right), gait, action
    (jumping/squatting/none), and the multi-IMU translation confidence.
    """
    ri = app_state.root_integrator
    if ri is None:
        raise HTTPException(503, "Root integrator not initialised")
    return ri.get_direction_state()


# ─────────────────────────────────────────────
# MOTION MODE  (test ↔ user)
# ─────────────────────────────────────────────

@app.post("/motion_mode")
async def set_motion_mode(mode: str = Query(..., description="'test' or 'user'")):
    """
    Switch between:
      test — pelvis-only: avatar translates on pelvis bounce (single-sensor
             testing). Action classifier also reports WALK on bounce.
      user — full multi-IMU biomechanical classification; translation requires
             limb confirmation (won't drift on a stationary bounce).
    """
    if mode not in ("test", "user"):
        raise HTTPException(400, "mode must be 'test' or 'user'")
    app_state.motion_mode = mode
    if app_state.root_integrator is not None:
        app_state.root_integrator.set_translation_mode(mode)
    if app_state.action_classifier is not None:
        app_state.action_classifier.set_mode(mode)
    await imu_ws.broadcast({"type": "motion_mode", "mode": mode})
    return {"status": "ok", "mode": mode}


@app.get("/motion_mode")
async def get_motion_mode():
    return {"mode": app_state.motion_mode}


@app.post("/motion_mode/experimental")
async def set_experimental_states(enabled: bool = Query(...)):
    """
    Enable/disable the STUBBED action states (strafe, walk-backward, hybrids).
    Off by default — keep off until the leg IMUs are connected and these states
    are tuned on real multi-IMU data. The validated states (walk/run/jump/
    squat/idle) are unaffected either way.
    """
    if app_state.action_classifier is None:
        raise HTTPException(503, "Action classifier not initialised")
    app_state.action_classifier.enable_experimental_states(enabled)
    return {"status": "ok", "experimental_states_enabled": enabled}


# ─────────────────────────────────────────────
# OUTPUTS
# ─────────────────────────────────────────────

@app.get("/outputs/{filename}")
async def get_output(filename: str):
    p = OUTPUTS_DIR / filename
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(str(p))


# ─────────────────────────────────────────────
# INTEGRATOR TUNING ENDPOINTS (dev convenience)
# ─────────────────────────────────────────────

@app.patch("/debug/integrator")
async def tune_integrator(
    hp_alpha:        Optional[float] = Query(None, description="High-pass / smoothing alpha 0.90–0.99"),
    velocity_decay:  Optional[float] = Query(None, description="Heading smoothing 0.05–0.95"),
    accel_threshold: Optional[float] = Query(None, description="Stationary threshold (legacy units)"),
    scale:           Optional[float] = Query(None, description="Displacement / stride scale multiplier"),
):
    """
    Adjust RootIntegrator parameters at runtime without restarting (legacy
    shims — preserved for backward compatibility with v3/v4/v5 callers).

    Example:
        PATCH /debug/integrator?scale=0.5&velocity_decay=0.88
    """
    ri = app_state.root_integrator
    if ri is None:
        raise HTTPException(503, "Root integrator not initialised")
    if hp_alpha        is not None: ri._hp._alpha         = float(hp_alpha)
    if velocity_decay  is not None: ri._velocity_decay    = float(velocity_decay)
    if accel_threshold is not None: ri._accel_threshold   = float(accel_threshold)
    if scale           is not None: ri._scale             = float(scale)
    return {
        "hp_alpha":        ri._hp._alpha,
        "velocity_decay":  ri._velocity_decay,
        "accel_threshold": ri._accel_threshold,
        "scale":           ri._scale,
    }


@app.patch("/debug/integrator_v7")
async def tune_integrator_v7(
    move_tau:      Optional[float] = Query(None, description="Velocity ease time-constant s (lower=snappier, higher=heavier)"),
    stop_decay:    Optional[float] = Query(None, description="Target decay per frame when stopped 0.80–0.99"),
    max_speed:     Optional[float] = Query(None, description="Speed cap m/s"),
    stride_length: Optional[float] = Query(None, description="Stride length m (speed = stride / cadence)"),
):
    """
    Tune the v7 velocity-glide smoothness and gait behaviour live.

    - move_tau is the single biggest 'feel' knob: 0.18 s is a responsive
      game-character glide; raise toward 0.30 s for heavier momentum, lower
      toward 0.10 s for snappier starts.
    - Walk vs run = stride_length / cadence, capped at max_speed.

    Example:
        PATCH /debug/integrator_v7?move_tau=0.22&max_speed=5.0
    """
    ri = app_state.root_integrator
    if ri is None:
        raise HTTPException(503, "Root integrator not initialised")
    if move_tau      is not None: ri._move_tau      = float(move_tau)
    if stop_decay    is not None: ri._stop_decay    = float(stop_decay)
    if max_speed     is not None: ri._max_speed     = float(max_speed)
    if stride_length is not None: ri._stride_length = float(stride_length)
    return ri.get_debug_info()


@app.post("/debug/recalibrate_forward")
async def recalibrate_forward():
    """
    Re-arm forward-axis auto-detection mid-session.
    Position, gravity, and step count are all preserved — only the
    forward-axis detection window is reset.
    Have the user walk straight ~5 steps after calling this.
    """
    ri = app_state.root_integrator
    if ri is None:
        raise HTTPException(503, "Root integrator not initialised")
    ri.recalibrate_forward()
    return {
        "status":  "ok",
        "message": "Walk straight ~5 steps to re-detect forward axis.",
    }


@app.post("/debug/flip_forward")
async def flip_forward():
    """
    Reverse the detected forward axis (front <-> back).
    Call once if the skeleton is walking backward after auto-detection.
    Safe to call multiple times — each call toggles the direction.
    """
    ri = app_state.root_integrator
    if ri is None:
        raise HTTPException(503, "Root integrator not initialised")
    ri.flip_forward()
    return {
        "status":       "ok",
        "forward_axis": ri._forward_axis.tolist(),
        "message":      "Forward axis reversed.",
    }


@app.get("/debug/integrator_state")
async def integrator_state():
    """Return the full RootIntegrator debug snapshot (model, gait, counts, axes)."""
    ri = app_state.root_integrator
    if ri is None:
        raise HTTPException(503, "Root integrator not initialised")
    return ri.get_debug_info()


app.mount("/", StaticFiles(directory=str(TEMP3_FRONTEND_DIR), html=True), name="temp3_frontend")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
