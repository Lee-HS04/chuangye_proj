"""
backend/gvhmr_calibration.py
-----------------------------
GVHMR is no longer in the critical real-time path.
It is called only for two optional tasks:
  1. run_calibration_video() — given a short standing video, extract
     accurate limb segment lengths for this specific athlete.
  2. run_annotated_video()   — given a session recording, produce an
     annotated video overlaid with skeleton and biomechanics metrics
     for the post-session review page.

Reuses the existing SSH-based remote processing pipeline.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

# Reuse the existing SSH pipeline from the previous codebase
# (remote_ssh_pipeline.py must be present in backend/)
try:
    from remote_ssh_pipeline import process_video_on_remote
    SSH_AVAILABLE = True
except ImportError:
    SSH_AVAILABLE = False
    print("WARNING: remote_ssh_pipeline not found. GVHMR will be simulated.")


# SMPL-to-segment mapping (COCO-17 joint indices)
# For lower body: Hip=11/12, Knee=13/14, Ankle=15/16
SEGMENT_PAIRS = {
    "thigh_l": (11, 13),   # left hip → left knee
    "shin_l":  (13, 15),   # left knee → left ankle
    "thigh_r": (12, 14),
    "shin_r":  (14, 16),
}


# ─────────────────────────────────────────────
# CALIBRATION: extract limb lengths from video
# ─────────────────────────────────────────────

def run_calibration_video(video_path: str) -> Optional[dict]:
    """
    Processes a short (~5 s) standing video with GVHMR to extract
    per-athlete limb segment lengths.

    Returns a dict like:
      {"thigh_l": 0.44, "shin_l": 0.41, "thigh_r": 0.44, "shin_r": 0.41,
       "hip_width": 0.20, "pelvis_height": 0.92}
    or None on failure.
    """
    if not SSH_AVAILABLE:
        print("GVHMR not available — returning default limb lengths.")
        return _default_lengths()

    print(f"Running GVHMR calibration on: {video_path}")
    try:
        result = process_video_on_remote(video_path, f_mm=24)
    except Exception as exc:
        print(f"GVHMR calibration failed: {exc}")
        raise RuntimeError(f"GVHMR calibration failed: {exc}") from exc

    if result is None:
        raise RuntimeError(
            "GVHMR remote processing returned no result. Check the backend terminal "
            "for the SSH/GVHMR error printed above."
        )

    joints_3d = result.get("joints_3d_global")  # shape (T, 44, 3)
    if joints_3d is None:
        raise RuntimeError("GVHMR result did not include decoded joints_3d_global data.")

    if isinstance(joints_3d, torch.Tensor):
        joints_3d = joints_3d.numpy()

    # Use the median frame (most stable posture) for length extraction
    T = joints_3d.shape[0]
    if T == 0:
        raise RuntimeError("GVHMR returned zero frames for calibration.")
    mid = T // 2
    # Average over a 1-second window around the middle
    fps_approx = 30
    start = max(0, mid - fps_approx // 2)
    end   = min(T, mid + fps_approx // 2)
    stable_joints = joints_3d[start:end]   # (window, 44, 3)

    # SMPL (44) → COCO-17 mapping via existing helper
    try:
        from body_tracking import smpl_to_coco17
        coco_frames = np.array([smpl_to_coco17(f) for f in stable_joints])  # (window, 17, 3)
    except ImportError:
        # Fallback: use joints directly if body_tracking not present
        coco_frames = stable_joints[:, :17, :]

    # Compute median segment lengths across the stable window
    lengths = {}
    for seg_name, (j1_idx, j2_idx) in SEGMENT_PAIRS.items():
        segs = np.linalg.norm(
            coco_frames[:, j1_idx, :] - coco_frames[:, j2_idx, :], axis=1
        )
        lengths[seg_name] = float(np.median(segs))

    # Hip width: distance between left and right hip joints
    hip_dists = np.linalg.norm(
        coco_frames[:, 11, :] - coco_frames[:, 12, :], axis=1
    )
    lengths["hip_width"] = float(np.median(hip_dists)) / 2   # half-width offset

    # Pelvis height: median Y of mid-hip
    mid_hip_y = (coco_frames[:, 11, 1] + coco_frames[:, 12, 1]) / 2
    lengths["pelvis_height"] = float(np.median(np.abs(mid_hip_y)))

    print(f"Calibration complete: {lengths}")
    return lengths


# ─────────────────────────────────────────────
# ANNOTATED VIDEO for post-session review
# ─────────────────────────────────────────────

def run_annotated_video(
    video_path: str,
    output_dir: str,
    session_metrics: Optional[dict] = None,
    f_mm: int = 24,
) -> Optional[str]:
    """
    Runs GVHMR on a recorded session video and produces an annotated
    output video with:
      - Skeleton overlay
      - Per-frame biomechanics metrics (from session_metrics if provided)
      - Summary overlay in final seconds

    Returns the local path to the annotated video, or None on failure.
    """
    if not SSH_AVAILABLE:
        print("GVHMR not available — returning input video as-is.")
        return video_path

    print(f"Running GVHMR annotation on: {video_path}")

    try:
        result = process_video_on_remote(video_path, f_mm=f_mm)
    except Exception as exc:
        print(f"GVHMR annotation failed: {exc}")
        return None

    if result is None:
        return None

    try:
        from body_tracking import project_3d_to_2d, smpl_to_coco17
    except ImportError:
        print("body_tracking module not found. Cannot annotate video.")
        return None

    joints_3d_incam = result["joints_3d_incam"]
    K_fullimg        = result["K_fullimg"]

    cap = cv2.VideoCapture(video_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_name = Path(video_path).stem
    output_path = str(Path(output_dir) / f"{video_name}_annotated.mp4")

    writer = None
    for codec in ("avc1", "H264", "XVID", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))
        if writer.isOpened():
            break
        writer.release()
        writer = None

    if writer is None:
        cap.release()
        return None

    num_gvhmr = len(joints_3d_incam)

    for idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        idx_b = min(idx, num_gvhmr - 1)
        K = K_fullimg[idx_b] if K_fullimg.ndim == 3 else K_fullimg

        joints_2d = smpl_to_coco17(project_3d_to_2d(joints_3d_incam[idx_b], K))
        frame = _draw_skeleton_2d(frame, joints_2d)

        # Overlay metrics text
        frame = _overlay_metrics(frame, session_metrics, idx, total, fps)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Annotated video saved: {output_path}")
    return output_path


# ─────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────

_SKELETON_EDGES = [
    (0,1),(0,2),(1,3),(2,4),
    (5,7),(7,9),(6,8),(8,10),
    (5,6),(5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

def _draw_skeleton_2d(frame: np.ndarray, keypoints: list) -> np.ndarray:
    for pt in keypoints:
        if pt is not None:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
    for p1, p2 in _SKELETON_EDGES:
        k1, k2 = keypoints[p1], keypoints[p2]
        if k1 is not None and k2 is not None:
            cv2.line(frame, (int(k1[0]), int(k1[1])), (int(k2[0]), int(k2[1])), (0, 0, 255), 2)
    return frame


def _overlay_metrics(
    frame: np.ndarray,
    metrics: Optional[dict],
    frame_idx: int,
    total_frames: int,
    fps: float,
) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Progress bar at bottom
    bar_w = int(frame.shape[1] * frame_idx / max(1, total_frames - 1))
    cv2.rectangle(frame, (0, frame.shape[0] - 6), (bar_w, frame.shape[0]), (0, 200, 255), -1)

    if metrics is None:
        return frame

    y = 30
    def _put(text, color=(255, 255, 255)):
        nonlocal y
        cv2.putText(frame, text, (12, y), font, 0.65, color, 2)
        y += 28

    sway = metrics.get("sway", {})
    sym  = metrics.get("symmetry", {})
    fat  = metrics.get("fatigue", {})

    _put(f"Stability: {sway.get('stability_score', '--')}")
    _put(f"Asym: {sym.get('mean_knee_asym_deg', '--')}° ({sym.get('grade', '')})")
    _put(f"Fatigue: {'detected' if fat.get('detected') else 'normal'}",
         color=(0, 80, 255) if fat.get("detected") else (255, 255, 255))

    reps = metrics.get("total_reps", "--")
    _put(f"Reps: {reps}")

    return frame


def _default_lengths() -> dict:
    return {
        "thigh_l": 0.42, "shin_l": 0.40,
        "thigh_r": 0.42, "shin_r": 0.40,
        "hip_width": 0.19, "pelvis_height": 0.95,
    }
