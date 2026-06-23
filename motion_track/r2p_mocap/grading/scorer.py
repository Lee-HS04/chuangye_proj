"""
MotionScorer

Records reference motion clips from live sensor quaternions and
grades subsequent live windows by DTW comparison.

Usage:
    scorer = MotionScorer(history_len=500)
    # every engine tick:
    scorer.push_frame(quats_dict)

    # record a reference:
    scorer.start_recording()
    # ... wait while user performs the move ...
    scorer.stop_recording()

    # grade:
    result = scorer.grade()
    # result = {"score": 84, "grade": "B", "distance": 0.12, "per_sensor": {...}}
"""
from __future__ import annotations
from collections import deque
from typing import Dict, List, Optional
import numpy as np

from .dtw import dtw_distance, score_from_distance

# Sensor IDs used as features — must match what server sends in quats dict.
GRADE_SENSORS = [
    "pelvis", "chest", "head",
    "l_thigh", "r_thigh",
    "l_shin",  "r_shin",
    "l_foot",  "r_foot",
    "l_upper_arm", "r_upper_arm",
    "l_forearm",   "r_forearm",
]

_IDENT = [1.0, 0.0, 0.0, 0.0]


def _frame_vec(quats: Dict[str, List[float]]) -> np.ndarray:
    """Flatten all sensor quaternions into one feature vector (52 dims)."""
    parts: List[float] = []
    for sid in GRADE_SENSORS:
        q = quats.get(sid, _IDENT)
        parts.extend(q[:4])
    return np.array(parts, dtype=np.float64)


class MotionScorer:

    def __init__(self, history_len: int = 500):
        self._history:   deque          = deque(maxlen=history_len)
        self._reference: Optional[np.ndarray] = None   # (T, 52)
        self._recording: bool           = False
        self._rec_buf:   List[np.ndarray] = []

    # ── Feed ───────────────────────────────────────────────────────────────

    def push_frame(self, quats: Dict[str, List[float]]):
        """Call once per engine tick with the current sensor quats dict."""
        vec = _frame_vec(quats)
        self._history.append(vec)
        if self._recording:
            self._rec_buf.append(vec)

    # ── Reference recording ────────────────────────────────────────────────

    def start_recording(self):
        self._rec_buf  = []
        self._recording = True

    def stop_recording(self) -> int:
        """Finalise and store reference. Returns number of captured frames."""
        self._recording = False
        if self._rec_buf:
            self._reference = np.array(self._rec_buf, dtype=np.float64)
        return len(self._rec_buf)

    @property
    def has_reference(self) -> bool:
        return self._reference is not None

    @property
    def reference_duration_s(self) -> float:
        if self._reference is None:
            return 0.0
        return len(self._reference) / 50.0   # assume 50 Hz

    # ── Grading ────────────────────────────────────────────────────────────

    def grade(self, window_frames: Optional[int] = None) -> Dict:
        """
        Compare the most recent `window_frames` of history to the reference.
        If window_frames is None, uses len(reference) frames.
        """
        if not self.has_reference:
            return {"score": None, "grade": None,
                    "error": "no reference recorded — use /api/record-reference first"}
        if len(self._history) < 10:
            return {"score": None, "grade": None,
                    "error": "insufficient motion history"}

        ref  = self._reference
        hist = np.array(list(self._history), dtype=np.float64)
        n    = window_frames or len(ref)
        live = hist[-n:] if len(hist) >= n else hist

        dist         = dtw_distance(live, ref)
        score, grade = score_from_distance(dist)

        # Per-sensor breakdown (4 dims each)
        per_sensor = {}
        dim = 4
        for i, sid in enumerate(GRADE_SENSORS):
            a = live[:, i*dim:(i+1)*dim]
            b = ref[:,  i*dim:(i+1)*dim]
            d = dtw_distance(a, b)
            s, g = score_from_distance(d)
            per_sensor[sid] = {"score": s, "grade": g, "distance": round(d, 4)}

        return {
            "score":      score,
            "grade":      grade,
            "distance":   round(dist, 4),
            "ref_frames": len(ref),
            "live_frames": len(live),
            "per_sensor": per_sensor,
        }
