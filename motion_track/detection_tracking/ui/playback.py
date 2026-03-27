"""
ui/playback.py
--------------
Playback loops for both uploaded video and webcam.
Each function receives the shared `process_fn` callable so it stays
decoupled from frame_processor internals.
"""
from __future__ import annotations

import time
from typing import Callable

import cv2
import numpy as np
import streamlit as st

from ui.frame_processor import resize_if_needed


def run_video_playback(
    frame_placeholder: st.delta_generator.DeltaGenerator,
    process_fn: Callable[[np.ndarray], np.ndarray],
) -> None:
    """
    Play or show a single frame of the uploaded video.

    - If `playing` is True  → advance through frames until end or paused.
    - If `playing` is False → render only the current frame_index.
    """
    cap_path = st.session_state["cap_path"]
    total    = st.session_state["total_frames"]
    fps      = st.session_state["video_fps"]
    cap      = cv2.VideoCapture(cap_path)

    if st.session_state["playing"]:
        _play_loop(cap, frame_placeholder, process_fn, total, fps)
        cap.release()
        st.rerun()
    else:
        _show_single_frame(cap, frame_placeholder, process_fn)
        cap.release()


def run_webcam_playback(
    frame_placeholder: st.delta_generator.DeltaGenerator,
    process_fn: Callable[[np.ndarray], np.ndarray],
) -> None:
    """Stream from the default webcam until the Stop button is pressed."""
    cap      = cv2.VideoCapture(0)
    stop_btn = st.button("⏹ Stop Webcam")

    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret:
            break
        frame = resize_if_needed(frame)
        frame_placeholder.image(process_fn(frame), use_container_width=True)

    cap.release()


# ── private ──────────────────────────────────────────────────────────────────

def _play_loop(
    cap:               cv2.VideoCapture,
    frame_placeholder: st.delta_generator.DeltaGenerator,
    process_fn:        Callable[[np.ndarray], np.ndarray],
    total:             int,
    fps:               float,
) -> None:
    while st.session_state["playing"] and st.session_state["frame_index"] < total - 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state["frame_index"])
        ret, frame = cap.read()
        if not ret:
            st.session_state["playing"] = False
            break

        frame = resize_if_needed(frame)
        frame_placeholder.image(process_fn(frame), use_container_width=True)

        speed = st.session_state.get("playback_speed", 1.0)
        step  = max(1, round(speed)) if speed >= 2.0 else 1
        delay = (1.0 / fps / speed)  if speed < 2.0  else 0.001

        st.session_state["frame_index"] = min(
            total - 1, st.session_state["frame_index"] + step
        )
        time.sleep(max(0.001, delay))

    if st.session_state["frame_index"] >= total - 1:
        st.session_state["playing"] = False


def _show_single_frame(
    cap:               cv2.VideoCapture,
    frame_placeholder: st.delta_generator.DeltaGenerator,
    process_fn:        Callable[[np.ndarray], np.ndarray],
) -> None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state["frame_index"])
    ret, frame = cap.read()
    if ret:
        frame = resize_if_needed(frame)
        frame_placeholder.image(process_fn(frame), use_container_width=True)