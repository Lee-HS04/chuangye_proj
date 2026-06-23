"""
IMU Motion Segmenter
====================
Takes a saved recording (list of {t, quats} frames) and splits it into
action and pause segments using a hysteresis energy detector.

Works with any subset of sensors — 3 IMUs (pelvis + both thighs) is enough
for turns, jumps, and walking because they all produce large pelvis/thigh
quaternion changes relative to a rest state.

Algorithm
---------
1. Per-frame energy = mean angular displacement (rad) across all active sensors
   between consecutive frames — this is the quaternion derivative proxy.
2. Smooth with a sliding window to suppress sensor noise.
3. Hysteresis threshold: enter ACTION state when energy > HIGH,
   return to PAUSE state only when energy < LOW (prevents rapid toggling).
4. Merge adjacent actions separated by pauses shorter than min_pause.
5. Discard action segments shorter than min_action (noise / accidental twitches).
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Any

# ── Tuneable defaults ──────────────────────────────────────────────────────────
SMOOTH_WINDOW = 9       # frames  (~180 ms at 50 Hz)
ENERGY_HIGH   = 0.055   # rad/frame — cross upward  → ACTION
ENERGY_LOW    = 0.028   # rad/frame — cross downward → PAUSE  (hysteresis gap)
MIN_ACTION_S  = 0.25    # seconds  — ignore sub-quarter-second blips
MIN_PAUSE_S   = 0.40    # seconds  — merge reps closer than this


# ── Core helpers ───────────────────────────────────────────────────────────────

def _qdist(a: List[float], b: List[float]) -> float:
    """Angular distance between two [w,x,y,z] quaternions (radians)."""
    dot = max(-1.0, min(1.0, a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]))
    return 2.0 * float(np.arccos(abs(dot)))


def compute_energy(
    frames: List[Dict],
    smooth_window: int = SMOOTH_WINDOW,
) -> np.ndarray:
    """
    Returns a (N,) array of smoothed per-frame motion energy (rad/frame).
    Higher = more movement across all active sensors.
    """
    n   = len(frames)
    raw = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        dists = []
        for sid, q in frames[i]["quats"].items():
            prev = frames[i - 1]["quats"].get(sid)
            if prev:
                dists.append(_qdist(q, prev))
        if dists:
            raw[i] = float(np.mean(dists))

    kernel = np.ones(smooth_window) / smooth_window
    return np.convolve(raw, kernel, mode="same")


# ── Segmentation ───────────────────────────────────────────────────────────────

def segment(
    frames:        List[Dict],
    sample_rate:   float = 50.0,
    energy_high:   float = ENERGY_HIGH,
    energy_low:    float = ENERGY_LOW,
    min_action_s:  float = MIN_ACTION_S,
    min_pause_s:   float = MIN_PAUSE_S,
) -> Dict[str, Any]:
    """
    Segment a recording into alternating pause/action periods.

    Parameters
    ----------
    frames       : list of {"t": float, "quats": {sensor_id: [w,x,y,z]}}
    sample_rate  : Hz (used for frame-count → duration conversions)
    energy_high  : threshold to enter ACTION state
    energy_low   : threshold to return to PAUSE state (hysteresis)
    min_action_s : discard action segments shorter than this
    min_pause_s  : merge consecutive actions if pause between them is shorter

    Returns
    -------
    {
      energy     : [float, ...]   per-frame energy for visualisation
      segments   : [{type, start_frame, end_frame, start_t, end_t, duration}, ...]
      reps       : action segments only, each with rep number + extracted frames
      rep_count  : int
      summary    : human-readable string
    }
    """
    if not frames:
        return {"energy": [], "segments": [], "reps": [],
                "rep_count": 0, "summary": "Empty recording"}

    energy = compute_energy(frames, SMOOTH_WINDOW)
    n      = len(energy)
    min_action_f = max(1, int(min_action_s * sample_rate))
    min_pause_f  = max(1, int(min_pause_s  * sample_rate))

    # ── Hysteresis state machine ───────────────────────────────────────────────
    in_action = False
    raw_segs: List[tuple] = []      # (start_frame, end_frame)
    seg_start = 0

    for i, e in enumerate(energy):
        if not in_action and e >= energy_high:
            in_action = True
            seg_start = i
        elif in_action and e < energy_low:
            in_action = False
            raw_segs.append((seg_start, i))
    if in_action:
        raw_segs.append((seg_start, n - 1))

    # ── Merge reps separated by short pauses ──────────────────────────────────
    merged: List[tuple] = []
    for start, end in raw_segs:
        if merged and (start - merged[-1][1]) < min_pause_f:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    # ── Drop too-short actions ─────────────────────────────────────────────────
    actions = [(s, e) for s, e in merged if (e - s) >= min_action_f]

    # ── Build interleaved segment list (pauses fill the gaps) ─────────────────
    full_segs: List[Dict] = []
    prev = 0
    for s, e in actions:
        if s > prev:
            full_segs.append(_make_seg("pause",  frames, prev, s))
        full_segs.append(_make_seg("action", frames, s, e))
        prev = e
    if prev < n - 1:
        full_segs.append(_make_seg("pause", frames, prev, n - 1))

    # ── Number reps and attach extracted frames ────────────────────────────────
    rep_num = 0
    reps    = []
    for seg in full_segs:
        if seg["type"] == "action":
            rep_num += 1
            rep = dict(rep=rep_num, **seg)
            rep["frames"] = frames[seg["start_frame"] : seg["end_frame"] + 1]
            reps.append(rep)

    # ── Summary ───────────────────────────────────────────────────────────────
    durations = [r["duration"] for r in reps]
    if durations:
        summary = (
            f"{len(reps)} rep(s) detected — "
            f"avg {np.mean(durations):.2f}s, "
            f"range {min(durations):.2f}–{max(durations):.2f}s"
        )
    else:
        summary = "No action segments detected — try lowering the energy threshold"

    return {
        "energy":     [round(float(v), 5) for v in energy],
        "segments":   full_segs,
        "reps":       [{k: v for k, v in r.items() if k != "frames"} for r in reps],
        "reps_with_frames": reps,   # full payload (large — use only when saving rep clips)
        "rep_count":  len(reps),
        "summary":    summary,
        "thresholds": {"high": energy_high, "low": energy_low},
    }


def _make_seg(kind: str, frames: List[Dict], start: int, end: int) -> Dict:
    return {
        "type":        kind,
        "start_frame": start,
        "end_frame":   end,
        "start_t":     frames[start]["t"],
        "end_t":       frames[end]["t"],
        "duration":    round(frames[end]["t"] - frames[start]["t"], 3),
        "frame_count": end - start,
    }


# ── Convenience: save extracted reps as separate recording files ──────────────

def save_reps(
    result:      Dict[str, Any],
    base_name:   str,
    recordings_dir,
    sample_rate: float = 50.0,
    recorded_at: str   = "",
) -> List[str]:
    """
    Save each detected rep as its own JSON recording file.
    Returns list of saved file paths.
    """
    import json
    from pathlib import Path
    out_dir = Path(recordings_dir)
    out_dir.mkdir(exist_ok=True)
    saved = []
    for rep in result["reps_with_frames"]:
        name = f"{base_name}_rep{rep['rep']}"
        data = {
            "name":        name,
            "recorded_at": recorded_at,
            "sample_rate": sample_rate,
            "source":      base_name,
            "rep":         rep["rep"],
            "duration":    rep["duration"],
            "frame_count": rep["frame_count"],
            "frames":      rep["frames"],
        }
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        saved.append(str(path))
    return saved
