"""
calibration.py  —  IMU-to-Bone Calibration System
===================================================

Handles the full calibration pipeline that professional mocap suits require:

1. POSE CAPTURE  — record sensor orientations while the user holds a known pose
                   (T-pose or A-pose)
2. OFFSET SOLVE  — compute q_offset = q_ref_inv * q_current for each sensor,
                   where q_ref is the canonical bone orientation in the chosen pose
3. FORWARD SOLVE — determine the body's forward direction in world space from
                   the pelvis sensor, establishing the global reference frame
4. BIND          — store per-sensor calibration quaternions; all future motion
                   is expressed relative to these

Why per-sensor calibration is critical
---------------------------------------
The IMU chip inside a sensor housing may not be perfectly aligned with the
body segment it's strapped to.  Even 5–10° of mounting misalignment causes
the knee to appear bent in the rest pose, the shoulders to be raised, etc.
The calibration offset corrects this per-sensor mounting error so the rest
pose maps to a canonical skeletal neutral.

T-pose vs A-pose
----------------
T-pose: arms horizontal, legs vertical, body fully extended.
        Easiest for upper body (shoulder/elbow alignment).
A-pose: arms at ~45° from sides, knees very slightly flexed.
        More comfortable; reduces shoulder impingement during calibration.
        Preferred for walking/running capture sessions.

The canonical orientations for each bone in T-pose (Y-up, -Z forward):
  - Pelvis:      identity (body upright, facing -Z)
  - Chest:       identity
  - Head:        identity
  - Thighs:      pointing straight down (-Y)
  - Shins:       pointing straight down (-Y)
  - Feet:        pointing forward (-Z)
  - Upper arms:  pointing sideways (±X for T-pose, ±45° for A-pose)
  - Forearms:    pointing sideways (same as upper arms in both poses)
  - Shoulders:   identity (absorb clavicle angle)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import ALL_SENSORS, DEFAULT_LIMB_LENGTHS


# ── math ─────────────────────────────────────────────────────────────────────

def _norm(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-9 else np.array([1.0, 0, 0, 0])


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1,x1,y1,z1 = a;  w2,x2,y2,z2 = b
    return np.array([
        w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
        w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2,
    ])


def _qconj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _ax_angle(axis: np.ndarray, deg: float) -> np.ndarray:
    h = np.radians(deg) / 2.0
    s = np.sin(h)
    return _norm(np.array([np.cos(h), axis[0]*s, axis[1]*s, axis[2]*s]))


def _qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.array([0.0, v[0], v[1], v[2]])
    return _qmul(_qmul(q, qv), _qconj(q))[1:]


# ── canonical reference orientations ─────────────────────────────────────────
# These define the expected sensor orientation when the user is in a perfect
# T-pose, expressed in world space (Y-up, -Z forward).
# The calibration offset = conjugate(q_current) * q_canonical.
_I = np.array([1.0, 0, 0, 0])

T_POSE_REFS: Dict[str, np.ndarray] = {
    'pelvis':      _I,
    'chest':       _I,
    'head':        _I,
    'thigh_l':     _ax_angle(np.array([1,0,0],dtype=float),  0.0),   # straight down
    'thigh_r':     _ax_angle(np.array([1,0,0],dtype=float),  0.0),
    'shin_l':      _ax_angle(np.array([1,0,0],dtype=float),  0.0),
    'shin_r':      _ax_angle(np.array([1,0,0],dtype=float),  0.0),
    'l_foot':      _ax_angle(np.array([1,0,0],dtype=float), -90.0),  # pointing forward
    'r_foot':      _ax_angle(np.array([1,0,0],dtype=float), -90.0),
    'l_shoulder':  _I,
    'r_shoulder':  _I,
    'l_upper_arm': _ax_angle(np.array([0,0,1],dtype=float),  90.0),  # arms out sideways
    'r_upper_arm': _ax_angle(np.array([0,0,1],dtype=float), -90.0),
    'l_forearm':   _ax_angle(np.array([0,0,1],dtype=float),  90.0),
    'r_forearm':   _ax_angle(np.array([0,0,1],dtype=float), -90.0),
}

# A-pose: arms at 45° from sides
A_POSE_REFS: Dict[str, np.ndarray] = {
    **T_POSE_REFS,
    'l_upper_arm': _ax_angle(np.array([0,0,1],dtype=float),  45.0),
    'r_upper_arm': _ax_angle(np.array([0,0,1],dtype=float), -45.0),
    'l_forearm':   _ax_angle(np.array([0,0,1],dtype=float),  45.0),
    'r_forearm':   _ax_angle(np.array([0,0,1],dtype=float), -45.0),
    # Slight knee flex in A-pose
    'thigh_l':     _ax_angle(np.array([1,0,0],dtype=float),  5.0),
    'thigh_r':     _ax_angle(np.array([1,0,0],dtype=float),  5.0),
}

POSE_REFS = {'t_pose': T_POSE_REFS, 'a_pose': A_POSE_REFS}


# ── data types ────────────────────────────────────────────────────────────────

@dataclass
class CalibrationResult:
    pose_type:        str                         # 't_pose' | 'a_pose'
    offsets:          Dict[str, List[float]]      # sensor_id → [w,x,y,z]
    forward_vector:   List[float]                 # world-space [x,y,z] body forward
    up_vector:        List[float]                 # world-space [x,y,z] body up
    timestamp:        float = field(default_factory=time.time)
    limb_scale:       float = 1.0                 # overall scale vs default limb lengths
    notes:            str = ''

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'CalibrationResult':
        d['offsets'] = {k: np.array(v) for k, v in d['offsets'].items()}
        return cls(**d)

    def get_offset(self, sensor_id: str) -> np.ndarray:
        raw = self.offsets.get(sensor_id)
        if raw is None:
            return np.array([1.0, 0, 0, 0])
        return _norm(np.array(raw))


# ── calibration engine ────────────────────────────────────────────────────────

class CalibrationEngine:
    """
    Manages the T-pose / A-pose calibration workflow.

    Workflow
    --------
    1. Call begin(pose_type)           — enters collection mode
    2. Feed update(sensor_id, q, ts)  — accumulates samples for each sensor
       for COLLECTION_DURATION seconds while user holds the pose
    3. Call finalize()                 — solves offsets, returns CalibrationResult
    4. Pass result to apply(q, sid)    — removes mounting offset from any quaternion

    The collect-then-solve approach (rather than single-frame snapshot) is
    more robust: it averages out motion during the hold and ignores sensors
    that arrive slightly late.
    """

    COLLECTION_DURATION = 2.0   # seconds to accumulate samples

    def __init__(self):
        self._active     = False
        self._pose_type  = 't_pose'
        self._start_ts   = 0.0
        self._samples:   Dict[str, List[np.ndarray]] = {}
        self._result:    Optional[CalibrationResult] = None

    # ── public API ────────────────────────────────────────────────────────────

    def begin(self, pose_type: str = 't_pose') -> None:
        """Start a calibration session. pose_type: 't_pose' | 'a_pose'"""
        if pose_type not in POSE_REFS:
            pose_type = 't_pose'
        self._active    = True
        self._pose_type = pose_type
        self._start_ts  = time.time()
        self._samples   = {sid: [] for sid in ALL_SENSORS}
        print(f'[Calibration] Started {pose_type} — hold pose for {self.COLLECTION_DURATION:.0f}s')

    def update(self, sensor_id: str, q: np.ndarray, ts: float) -> bool:
        """
        Feed a sensor quaternion during calibration.
        Returns True once collection is complete and finalize() can be called.
        """
        if not self._active:
            return False
        if sensor_id in self._samples:
            self._samples[sensor_id].append(_norm(q))
        elapsed = time.time() - self._start_ts
        return elapsed >= self.COLLECTION_DURATION

    def finalize(self) -> Optional[CalibrationResult]:
        """
        Solve calibration offsets from collected samples.
        Each sensor's offset = conjugate(mean_q) * canonical_ref_q.
        Returns CalibrationResult, or None if insufficient data.
        """
        if not self._active:
            return None
        self._active = False
        refs = POSE_REFS[self._pose_type]
        offsets: Dict[str, List[float]] = {}
        pelvis_q = None

        for sid, samples in self._samples.items():
            if not samples:
                offsets[sid] = [1.0, 0.0, 0.0, 0.0]
                continue
            # Quaternion averaging using the first sample as reference
            # (avoids sign-flip issues in naive averaging)
            ref = samples[0]
            aligned = []
            for s in samples:
                if np.dot(ref, s) < 0:
                    aligned.append(-s)
                else:
                    aligned.append(s)
            mean_q = _norm(np.mean(aligned, axis=0))

            # Offset = inv(mean_current) * canonical_target
            canonical = refs.get(sid, np.array([1.0, 0, 0, 0]))
            offset = _norm(_qmul(_qconj(mean_q), canonical))
            offsets[sid] = offset.tolist()
            if sid == 'pelvis':
                pelvis_q = mean_q

        # Forward vector: direction the pelvis faces during calibration
        if pelvis_q is not None:
            fwd_w = _qrot(pelvis_q, np.array([0.0, 0.0, -1.0]))
            up_w  = _qrot(pelvis_q, np.array([0.0, 1.0,  0.0]))
        else:
            fwd_w = np.array([0.0, 0.0, -1.0])
            up_w  = np.array([0.0, 1.0,  0.0])

        self._result = CalibrationResult(
            pose_type      = self._pose_type,
            offsets        = offsets,
            forward_vector = fwd_w.tolist(),
            up_vector      = up_w.tolist(),
        )
        print(f'[Calibration] Complete. Forward: {[round(x,2) for x in fwd_w]}')
        return self._result

    @property
    def active(self) -> bool:
        return self._active

    @property
    def result(self) -> Optional[CalibrationResult]:
        return self._result

    @property
    def elapsed(self) -> float:
        return time.time() - self._start_ts if self._active else 0.0

    @property
    def progress(self) -> float:
        """0.0 – 1.0 collection progress."""
        return min(1.0, self.elapsed / self.COLLECTION_DURATION)

    # ── apply calibration ─────────────────────────────────────────────────────

    @staticmethod
    def apply(q_raw: np.ndarray, offset: np.ndarray) -> np.ndarray:
        """
        Apply a stored offset to a live quaternion.
        q_corrected = q_raw * q_offset
        This removes the sensor mounting error so the bone is in canonical pose
        at rest.
        """
        return _norm(_qmul(q_raw, offset))

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        if self._result is None:
            raise ValueError('No calibration result to save.')
        data = self._result.to_dict()
        # Convert numpy arrays to plain lists for JSON
        data['offsets'] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                           for k, v in data['offsets'].items()}
        path.write_text(json.dumps(data, indent=2))
        print(f'[Calibration] Saved → {path}')

    def load(self, path: Path) -> CalibrationResult:
        data = json.loads(path.read_text())
        data['offsets'] = {k: np.array(v) for k, v in data['offsets'].items()}
        self._result = CalibrationResult(**data)
        print(f'[Calibration] Loaded ← {path}')
        return self._result