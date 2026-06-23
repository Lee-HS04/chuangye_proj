"""
test_main.py  —  R2P Full-Body IMU Reconstruction Engine
=========================================================

Wires together the complete pipeline defined in the system spec:

    BLE packets
        ↓  ble_receiver.py     (raw packet parsing)
        ↓  sensor_fusion.py    (complementary filter + confidence)
        ↓  motion_detector.py  (action + squat/jump/strafe detection)
        ↓  gait_analyzer.py    (stride phase, cadence, symmetry)
        ↓  skeleton.py         (FK + constraint + IK foot planting)
        ↓  analytics.py        (stability, fatigue, balance, scores)
        ↓  WebSocket broadcast → skeleton_viewer.html

Output per frame matches the system spec JSON:
  joints, root_pos, root_rot, movement_state,
  tracking_confidence, analytics scores, gait state

Run
---
    pip install fastapi uvicorn bleak numpy
    python test_main.py
    open http://localhost:8000/
"""

from __future__ import annotations

import asyncio
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

# ── paths ─────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).resolve().parent          # imu_test/
FRONTEND_DIR = HERE / "frontend"
BACKEND_DIR  = HERE.parent / "backend"
if not (BACKEND_DIR / "ble_receiver.py").exists():
    print(f"[WARN] ble_receiver.py not found in {BACKEND_DIR}")

# IMPORTANT: BACKEND_DIR is inserted first, then HERE is inserted at position 0.
# Final order: [HERE, BACKEND_DIR, ...stdlib...]
# This guarantees imu_test/skeleton.py is resolved before backend/skeleton.py,
# and likewise for any other module that exists in both directories.
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(HERE))           # ← position 0 = highest priority

from ble_receiver    import BLEReceiver, SensorPacket   # noqa: E402
from sensor_fusion   import FusionManager               # noqa: E402
from motion_detector import MotionDetector              # noqa: E402
from gait_analyzer   import GaitAnalyzer                # noqa: E402
from root_solver     import RootSolver                  # noqa: E402
from skeleton        import SkeletonSolver              # noqa: E402
from analytics       import AnalyticsEngine             # noqa: E402

# ── sensor tiers ──────────────────────────────────────────────────────────────
PELVIS_ONLY = ['pelvis']
CONSUMER    = ['pelvis','chest','thigh_l','thigh_r','shin_l','shin_r']
PRO         = CONSUMER + [
    'head','l_shoulder','r_shoulder',
    'l_upper_arm','r_upper_arm',
    'l_forearm','r_forearm',
    'l_foot','r_foot',
]
TIERS       = {'pelvis_only': PELVIS_ONLY, 'consumer': CONSUMER, 'pro': PRO}
ACCEPTED    = set(PRO)
STALE_S     = 0.4

# ── axis remap ────────────────────────────────────────────────────────────────
REMAP_PRESETS = {
    'identity':              [(0,+1),(1,+1),(2,+1)],
    'swap_yz':               [(0,+1),(2,+1),(1,+1)],
    'swap_yz_negz':          [(0,+1),(2,+1),(1,-1)],
    'swap_yz_negz_negx':     [(0,-1),(2,+1),(1,-1)],   # old (lateral tilt flipped)
    'swap_yz_negz_negx_v2':  [(0,-1),(2,+1),(1,+1)],   # fixed default
    'swap_yz_negyz':         [(0,+1),(2,-1),(1,-1)],
    'swap_xy':               [(1,+1),(0,+1),(2,+1)],
    'swap_xz':               [(2,+1),(1,+1),(0,+1)],
    'neg_all':               [(0,-1),(1,-1),(2,-1)],
}

def _norm(q): n=np.linalg.norm(q); return q/n if n>1e-9 else np.array([1.0,0,0,0])
def _qmul(a,b):
    w1,x1,y1,z1=a; w2,x2,y2,z2=b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2,w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2])
def _qconj(q): return np.array([q[0],-q[1],-q[2],-q[3]])

def remap_quat(q, mapping):
    w=q[0]; vec=q[1:]
    out=np.array([sign*vec[src] for src,sign in mapping])
    return _norm(np.array([w,out[0],out[1],out[2]]))

# ── state ─────────────────────────────────────────────────────────────────────
class Cfg:
    def __init__(self):
        self.preset   = 'swap_yz_negz_negx_v2'
        self.mapping  = REMAP_PRESETS['swap_yz_negz_negx_v2']
        self.zero_inv: dict[str,np.ndarray] = {}
        self._pending_zero: set[str] = set()
        self.mode           = 'auto'
        self.preset_override= False

cfg = Cfg()
_loop: Optional[asyncio.AbstractEventLoop] = None
_ble:  Optional[BLEReceiver]               = None
_last_seen:  dict[str,float] = {}

# ── engine instances ──────────────────────────────────────────────────────────
_fusion    = FusionManager()
_detector  = MotionDetector(fps=50.0)
_gait      = GaitAnalyzer(fps=50.0)
_skeleton  = SkeletonSolver(constraint_hardness=0.82, ik_enabled=True)
_analytics = AnalyticsEngine(fps=50.0)

# Smoothed root position (integrated from detector output)
_root_pos = np.zeros(3)
_root_y   = 0.0
_MOVE_SPEED = {'walk':1.4,'walk_fwd':1.4,'walk_back':1.0,
               'strafe_left':1.1,'strafe_right':1.1,'run':3.4}
_ROOT_MAX = 3.5

# Store latest smoothed quaternions for skeleton solver
_smooth_quats: dict[str,np.ndarray] = {}
_calibrating = False

# ── logger ────────────────────────────────────────────────────────────────────
LOG_DIR    = HERE / 'logs'
LOG_DIR.mkdir(exist_ok=True)
CSV_HEADER = [
    'timestamp','sensor','label',
    'raw_qw','raw_qx','raw_qy','raw_qz',
    'proc_qw','proc_qx','proc_qy','proc_qz',
    'yaw_deg','pitch_deg','roll_deg',
    'accel_x','accel_y','accel_z',
    'gyro_x','gyro_y','gyro_z',
    'remap_preset','zeroed',
]

def _quat_to_euler(q):
    w,x,y,z = q
    yaw   = np.degrees(np.arctan2(2*(w*y+z*x),1-2*(x*x+y*y)))
    sp    = np.clip(2*(w*x-y*z),-1,1)
    pitch = np.degrees(np.arcsin(float(sp)))
    roll  = np.degrees(np.arctan2(2*(w*z+x*y),1-2*(x*x+z*z)))
    return float(yaw), float(pitch), float(roll)

class Logger:
    def __init__(self):
        self._file=None; self._writer=None; self._path=None
        self._label=''; self.active=False; self._rows=0

    def start(self, label='') -> str:
        if self.active: self.stop()
        ts=datetime.now().strftime('%Y%m%d_%H%M%S')
        self._path=LOG_DIR/f'imu_{ts}.csv'
        self._file=open(self._path,'w',newline='',encoding='utf-8')
        self._writer=csv.writer(self._file)
        self._writer.writerow(CSV_HEADER)
        self._label=label; self.active=True; self._rows=0
        print(f'[Logger] Recording → {self._path}')
        return str(self._path)

    def stop(self) -> dict:
        if not self.active: return {'active':False}
        self.active=False
        if self._file: self._file.flush(); self._file.close(); self._file=None
        info={'path':str(self._path),'rows':self._rows}
        print(f'[Logger] Stopped. {self._rows} rows → {self._path}')
        self._path=None; self._writer=None
        return info

    def set_label(self, lbl): self._label=lbl

    def write(self, sid, q_raw, q_out, accel, gyro, ts, preset, zeroed):
        if not self.active or not self._writer: return
        yaw,pitch,roll = _quat_to_euler(q_out)
        row=[f'{ts:.4f}',sid,self._label,
             f'{q_raw[0]:.5f}',f'{q_raw[1]:.5f}',f'{q_raw[2]:.5f}',f'{q_raw[3]:.5f}',
             f'{q_out[0]:.5f}',f'{q_out[1]:.5f}',f'{q_out[2]:.5f}',f'{q_out[3]:.5f}',
             f'{yaw:.2f}',f'{pitch:.2f}',f'{roll:.2f}',
             f'{accel[0]:.3f}',f'{accel[1]:.3f}',f'{accel[2]:.3f}',
             f'{gyro[0]:.3f}', f'{gyro[1]:.3f}', f'{gyro[2]:.3f}',
             preset,int(zeroed)]
        self._writer.writerow(row)
        self._rows+=1
        if self._rows%50==0: self._file.flush()

    @property
    def status(self): return {'active':self.active,'path':str(self._path) if self._path else None,'rows':self._rows,'label':self._label}

_logger = Logger()

# ── helpers ───────────────────────────────────────────────────────────────────

def live_sensors() -> list[str]:
    now=time.time()
    return [s for s,t in _last_seen.items() if now-t<STALE_S]

def detect_tier(live: list[str]) -> str:
    ls=set(live)
    if set(CONSUMER).issubset(ls):
        return 'pro' if set(PRO).issubset(ls) else 'consumer'
    return 'pelvis_only'

def effective_mode() -> str:
    return detect_tier(live_sensors()) if cfg.mode=='auto' else cfg.mode

# ── WebSocket manager ─────────────────────────────────────────────────────────
class WSManager:
    def __init__(self): self.active: list[WebSocket] = []
    async def connect(self, ws):
        await ws.accept(); self.active.append(ws)
    def disconnect(self, ws):
        if ws in self.active: self.active.remove(ws)
    async def broadcast(self, data):
        msg=json.dumps(data); dead=[]
        for ws in self.active:
            try: await ws.send_text(msg)
            except: dead.append(ws)
        for ws in dead: self.disconnect(ws)

imu_ws = WSManager()

# ── packet pipeline ───────────────────────────────────────────────────────────

def on_sensor_packet(packet: SensorPacket):
    global _root_pos, _root_y, _calibrating
    sid = packet.sensor_id
    if sid not in ACCEPTED or _loop is None: return

    now = time.time()
    _last_seen[sid] = now
    q_raw   = _norm(packet.quaternion.astype(float))
    accel   = packet.accel.astype(float)
    gyro    = packet.gyro.astype(float)

    # ── zero offset ───────────────────────────────────────────────────────
    if sid in cfg._pending_zero:
        cfg.zero_inv[sid] = _qconj(q_raw)
        cfg._pending_zero.discard(sid)
        asyncio.run_coroutine_threadsafe(
            imu_ws.broadcast({'type':'zeroed','sensor':sid}), _loop)

    # ── remap + zero ──────────────────────────────────────────────────────
    q_mapped = remap_quat(q_raw, cfg.mapping)
    zinv     = cfg.zero_inv.get(sid)
    if zinv is not None:
        q_out = _norm(_qmul(remap_quat(zinv, cfg.mapping), q_mapped))
    else:
        q_out = q_mapped

    # ── sensor fusion filter ──────────────────────────────────────────────
    q_smooth = _fusion.update(sid, q_out, accel, gyro, packet.timestamp)
    _smooth_quats[sid] = q_smooth
    conf_sid = _fusion.confidence(sid)

    # ── log ───────────────────────────────────────────────────────────────
    _logger.write(sid, q_raw, q_smooth, accel, gyro,
                  packet.timestamp, cfg.preset, zinv is not None)

    # ── per-sensor analytics feeds ────────────────────────────────────────
    if sid in ('shin_l', 'shin_r'):
        side = 'l' if sid == 'shin_l' else 'r'
        _gait.update_shin(side, gyro, accel, q_smooth, packet.timestamp)

    if sid in ('thigh_l','thigh_r'):
        _analytics.update_limb(sid, gyro, q_smooth, packet.timestamp)

    # ── pelvis is the primary driver ──────────────────────────────────────
    detected_dict = None
    gait_dict     = None
    skel_dict     = None
    analytics_dict= None

    if sid == 'pelvis':
        # Motion detection
        det_state = _detector.update(q_smooth, accel, gyro, packet.timestamp)
        detected_dict = det_state.to_dict()

        # Analytics
        an_state = _analytics.update(q_smooth, accel, gyro, packet.timestamp, det_state.action)
        analytics_dict = an_state.to_dict()

        if _calibrating:
            _detector.calibrate_from_window()

        # Root position integration
        _integrate_root(det_state, q_smooth, 0.02)

        # Squat vertical offset
        if det_state.is_squat:
            _root_y = max(-0.35, _root_y - 0.003)
        else:
            _root_y = min(0.0, _root_y + 0.005)

        # Gait summary
        gait_dict = _gait.state.to_dict()

        # Skeleton FK solve (only meaningful when pelvis is live)
        live = live_sensors()
        try:
            skel_state = _skeleton.solve(
                sensor_quats  = _smooth_quats,
                root_pos      = _root_pos,
                root_y_offset = _root_y,
                live_sensors  = live,
                dt            = 0.02,
                action        = det_state.action,
                cadence_hz    = det_state.cadence_hz,
            )
            skel_dict = skel_state.to_dict()
        except Exception as e:
            print(f'[skeleton] error: {e}')

    # ── broadcast ─────────────────────────────────────────────────────────
    frame = {
        'type':             'frame',
        'ts':               packet.timestamp,
        'sensor':           sid,
        'quat':             q_smooth.tolist(),
        'preset':           cfg.preset,
        'zeroed':           zinv is not None,
        'confidence':       round(conf_sid, 3),
        'mode':             effective_mode(),
        'mode_setting':     cfg.mode,
        'preset_override':  cfg.preset_override,
        'detected':         detected_dict,
        'gait':             gait_dict,
        'skeleton':         skel_dict,
        'analytics':        analytics_dict,
        'accel':            accel.tolist(),
        'gyro':             gyro.tolist(),
        'calibrating':      _calibrating,
    }
    asyncio.run_coroutine_threadsafe(imu_ws.broadcast(frame), _loop)


def _integrate_root(det, qp: np.ndarray, dt: float):
    global _root_pos
    move_dir = det.move_dir
    if not move_dir or (move_dir[0]==0 and move_dir[1]==0): return
    if det.action in ('idle','squat','jump','tilt_fwd'): return
    if det.in_place: return

    speed = _MOVE_SPEED.get(det.action, 1.0)
    # Pelvis yaw → world-frame direction
    w,x,y,z = qp
    yaw_rad  = np.arctan2(2*(w*y+z*x), 1-2*(x*x+y*y))
    sinY, cosY = np.sin(yaw_rad), np.cos(yaw_rad)
    bR = move_dir[0]; bF = -move_dir[1]
    wX = bR*cosY + bF*sinY
    wZ = -bR*sinY + bF*cosY

    _root_pos[0] = float(np.clip(_root_pos[0]+wX*speed*dt, -_ROOT_MAX, _ROOT_MAX))
    _root_pos[2] = float(np.clip(_root_pos[2]+wZ*speed*dt, -_ROOT_MAX, _ROOT_MAX))


# ── lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loop, _ble
    _loop = asyncio.get_running_loop()
    _ble  = BLEReceiver(on_packet=on_sensor_packet)
    asyncio.create_task(_ble.run())
    asyncio.create_task(_mode_watcher())
    print('R2P engine ready →  http://localhost:8000/')
    yield
    if _ble: await _ble.stop()

async def _mode_watcher():
    prev = None
    while True:
        await asyncio.sleep(0.3)
        snap = (effective_mode(), tuple(sorted(live_sensors())))
        if snap != prev:
            prev = snap
            await imu_ws.broadcast({'type':'mode_update','mode':snap[0],'mode_setting':cfg.mode,'live':list(snap[1])})

app = FastAPI(title='R2P IMU Reconstruction Engine', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
app.mount('/static', StaticFiles(directory=str(FRONTEND_DIR)), name='static')

# ── routes ────────────────────────────────────────────────────────────────────
@app.get('/')
async def index(): return FileResponse(str(FRONTEND_DIR/'skeleton_viewer.html'))

@app.websocket('/ws/imu')
async def ws_imu(ws: WebSocket):
    await imu_ws.connect(ws)
    await ws.send_text(json.dumps({
        'type':'init','preset':cfg.preset,'presets':list(REMAP_PRESETS.keys()),
        'tiers':{k:v for k,v in TIERS.items()},
        'mode':effective_mode(),'mode_setting':cfg.mode,
        'preset_override':cfg.preset_override,'live':live_sensors(),
        'zeroed':sorted(cfg.zero_inv.keys()),
    }))
    try:
        while True: await ws.receive_text()
    except WebSocketDisconnect:
        imu_ws.disconnect(ws)

@app.post('/zero')
async def set_zero(sensor: Optional[str]=Query(None)):
    if sensor:
        if sensor not in ACCEPTED: raise HTTPException(400, f"Unknown sensor '{sensor}'")
        cfg._pending_zero.add(sensor); targets=[sensor]
    else:
        targets=live_sensors() or [PELVIS_ONLY[0]]
        cfg._pending_zero.update(targets)
    return {'status':'ok','zeroing':targets}

@app.post('/unzero')
async def clear_zero(sensor: Optional[str]=Query(None)):
    if sensor: cfg.zero_inv.pop(sensor,None); await imu_ws.broadcast({'type':'unzeroed','sensor':sensor})
    else: cfg.zero_inv.clear(); await imu_ws.broadcast({'type':'unzeroed','sensor':'all'})
    return {'status':'ok'}

@app.post('/remap')
async def set_remap(preset: str=Query(...)):
    if preset not in REMAP_PRESETS: raise HTTPException(400,f'Unknown preset. Options: {list(REMAP_PRESETS)}')
    cfg.preset=preset; cfg.mapping=REMAP_PRESETS[preset]
    await imu_ws.broadcast({'type':'remap','preset':preset})
    return {'status':'ok','preset':preset}

@app.post('/mode')
async def set_mode(mode: str=Query(...)):
    if mode not in ('auto','pelvis_only','consumer','pro'): raise HTTPException(400,'mode must be auto|pelvis_only|consumer|pro')
    cfg.mode=mode
    await imu_ws.broadcast({'type':'mode_update','mode':effective_mode(),'mode_setting':cfg.mode,'live':live_sensors()})
    return {'status':'ok','mode_setting':cfg.mode,'effective':effective_mode()}

@app.post('/preset_override')
async def set_override(enabled: bool=Query(...)):
    cfg.preset_override=enabled
    await imu_ws.broadcast({'type':'preset_override','enabled':enabled})
    return {'status':'ok','preset_override':enabled}

@app.post('/detect/params')
async def set_detect_params(
    travel_thresh:  Optional[float]=Query(None),
    bounce_thresh:  Optional[float]=Query(None),
    run_cadence_hz: Optional[float]=Query(None),
    walk_cadence_hz:Optional[float]=Query(None),
    dir_stability:  Optional[float]=Query(None),
):
    _detector.set_params(travel_thresh=travel_thresh,bounce_thresh=bounce_thresh,
                         run_cadence_hz=run_cadence_hz,walk_cadence_hz=walk_cadence_hz,
                         dir_stability=dir_stability)
    await imu_ws.broadcast({'type':'detect_params','params':_detector.get_params()})
    return {'status':'ok','params':_detector.get_params()}

@app.post('/detect/calibrate')
async def calibrate_detector(enabled: bool=Query(True)):
    global _calibrating
    _calibrating=enabled
    if not enabled:
        thr=_detector.calibrate_from_window()
        await imu_ws.broadcast({'type':'detect_params','params':_detector.get_params()})
        return {'status':'ok','calibrating':False,'travel_thresh':thr}
    return {'status':'ok','calibrating':True}

@app.post('/detect/reset')
async def reset_detector():
    _detector.reset(); _gait.reset(); _analytics.reset()
    return {'status':'ok'}

@app.post('/log/start')
async def log_start(label: str=Query('')):
    path=_logger.start(label=label)
    await imu_ws.broadcast({'type':'log_started','path':path,'label':label})
    return {'status':'ok','path':path}

@app.post('/log/stop')
async def log_stop():
    info=_logger.stop()
    await imu_ws.broadcast({'type':'log_stopped',**info})
    return {'status':'ok',**info}

@app.post('/log/label')
async def log_label(label: str=Query(...)):
    _logger.set_label(label)
    await imu_ws.broadcast({'type':'log_label','label':label})
    return {'status':'ok','label':label}

@app.get('/log/status')
async def log_status(): return _logger.status

@app.get('/log/files')
async def log_files():
    files=sorted(LOG_DIR.glob('*.csv'),reverse=True)
    return {'log_dir':str(LOG_DIR),'files':[{'name':f.name,'size_kb':round(f.stat().st_size/1024,1)} for f in files]}

@app.get('/config')
async def get_config():
    return {'preset':cfg.preset,'mapping':cfg.mapping,'presets':list(REMAP_PRESETS.keys()),
            'mode_setting':cfg.mode,'effective_mode':effective_mode(),
            'preset_override':cfg.preset_override,'tiers':TIERS,
            'live':live_sensors(),'zeroed':sorted(cfg.zero_inv.keys()),
            'detect_params':_detector.get_params(),'log':_logger.status}

@app.get('/analytics')
async def get_analytics():
    return {**_analytics.state.to_dict(), 'gait': _gait.state.to_dict()}

if __name__ == '__main__':
    uvicorn.run('test_main:app', host='0.0.0.0', port=8000, reload=False)