"""
ui/frame_processor.py
---------------------
Thread-safe version (FIXED: session_state access)
"""
from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

from body_tracking import get_keypoints
from posture_analysis import (
    extract_features,
    evaluate_multiple_rules,
    draw_feedback,
)
from analysis.cmj     import CMJAnalyser, CMJPhase
from analysis.sls     import SLSAnalyser
from analysis.balance import BalanceAnalyser


# ── Shared util ───────────────────────────────────────────────────────────────

def resize_if_needed(frame: np.ndarray, max_width: int = 800) -> np.ndarray:
    if frame.shape[1] > max_width:
        scale = max_width / frame.shape[1]
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    return frame


# ── Main dispatcher ───────────────────────────────────────────────────────────

def process_frame(
    frame:          np.ndarray,
    rules_all:      dict       = None,
    selected_rules: list[str]  = None,
    counter                    = None,
    sway_tracker               = None,
    cmj_analyser:   CMJAnalyser    = None,
    sls_analyser:   SLSAnalyser    = None,
    balance_analyser: BalanceAnalyser = None,
) -> np.ndarray:

    # ✅ SAFE access (no crash in thread)
    mode    = st.session_state.get("analysis_mode", "posture")
    floor_y = st.session_state.get("floor_y", 0)
    fps     = st.session_state.get("video_fps", 30.0)

    keypoints, annotated = get_keypoints(frame)
    display = annotated.copy() if annotated is not None else frame.copy()

    if keypoints is not None:
        if mode == "posture":
            display = _run_posture(
                display, keypoints, rules_all or {}, selected_rules or [],
                counter, sway_tracker, floor_y,
            )
        elif mode == "cmj" and cmj_analyser is not None:
            display = _run_cmj(display, keypoints, cmj_analyser, floor_y, fps)
        elif mode == "sls" and sls_analyser is not None:
            display = _run_sls(display, keypoints, sls_analyser)
        elif mode == "balance" and balance_analyser is not None:
            display = _run_balance(display, keypoints, balance_analyser, floor_y)

    _draw_floor_line(display, floor_y)
    return cv2.cvtColor(display, cv2.COLOR_BGR2RGB)


# ── Mode implementations ──────────────────────────────────────────────────────

def _run_posture(display, keypoints, rules_all, selected_rules,
                 counter, sway_tracker, floor_y):
    features = extract_features(keypoints)

    if len(keypoints) > 16 and keypoints[15] and keypoints[16]:
        features["jump_feet"] = floor_y - min(keypoints[15][1], keypoints[16][1])

    if len(keypoints) > 12 and keypoints[11] and keypoints[12]:
        mid_x = (keypoints[11][0] + keypoints[12][0]) / 2
        mid_y = (keypoints[11][1] + keypoints[12][1]) / 2
        features["mid_hip"] = (mid_x, floor_y - mid_y)

    if sway_tracker:
        sway_tracker.update(features.get("mid_hip"))
        sway_velocity             = sway_tracker.get_sway_velocity()
        features["sway_velocity"] = sway_velocity
    else:
        sway_velocity = None

    results    = evaluate_multiple_rules(features, rules_all, selected_rules)
    all_failed = set()
    y = 30

    for name, result in results.items():
        score, failed = result["score"], result["failed"]
        all_failed.update(failed)

        cv2.putText(display, f"{name}: {score}%",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 25

        if failed:
            cv2.putText(display, "Fail: " + ", ".join(failed),
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y += 25

    draw_feedback(display, keypoints, all_failed)

    if counter:
        reps = counter.update(features)
        cv2.putText(display, f"Reps: {reps}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        y += 30

    if sway_velocity is not None:
        baseline = 5
        change   = (sway_velocity - baseline) / baseline * 100

        if change < 10:
            ftxt, fcolor = "Stable", (0, 255, 0)
        elif change < 20:
            ftxt, fcolor = "Slight Fatigue", (0, 255, 255)
        else:
            ftxt, fcolor = "Fatigued", (0, 0, 255)

        cv2.putText(display, f"Sway: {sway_velocity:.2f}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fcolor, 2)
        cv2.putText(display, ftxt,
                    (10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fcolor, 2)

    return display


def _run_cmj(display, keypoints, cmj_analyser: CMJAnalyser, floor_y, fps):
    result = cmj_analyser.update(keypoints, floor_y, fps)

    if result is not None:
        st.session_state["cmj_last_result"] = result

    phase_color = {
        CMJPhase.IDLE:    (180, 180, 180),
        CMJPhase.LOADING: (255, 200,   0),
        CMJPhase.FLIGHT:  (  0, 220, 255),
        CMJPhase.LANDING: (  0, 255,   0),
    }.get(cmj_analyser.phase, (180, 180, 180))

    phase_label = {
        CMJPhase.IDLE:    "IDLE",
        CMJPhase.LOADING: "LOADING",
        CMJPhase.FLIGHT:  "FLIGHT!",
        CMJPhase.LANDING: "LANDED",
    }.get(cmj_analyser.phase, "")

    cv2.putText(display, f"CMJ: {phase_label}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, phase_color, 2)

    last = st.session_state.get("cmj_last_result")
    if last:
        _draw_status_badge(display, last.status, last.rsi_mod, "RSI_mod", y=70)

    _highlight_kps(display, keypoints, [11, 12, 15, 16], phase_color)
    return display


def _run_sls(display, keypoints, sls_analyser: SLSAnalyser):
    side   = st.session_state.get("sls_side", "left")
    result = sls_analyser.update(keypoints, side)

    if result is not None:
        st.session_state["sls_last_result"] = result

    last = st.session_state.get("sls_last_result")

    y = 35
    cv2.putText(display, f"SLS ({side})",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    y += 30

    if last:
        status_color = {"green": (0, 255, 0), "yellow": (0, 200, 255),
                        "red": (0, 0, 255)}.get(last.status, (180, 180, 180))
        cv2.putText(display, f"FPPA: {last.fppa_deg:.1f}deg",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        y += 28
        if not last.camera_ok:
            cv2.putText(display, "! Face camera",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)

    kp_map = {"left": [11, 13, 15], "right": [12, 14, 16]}
    _highlight_kps(display, keypoints, kp_map.get(side, []), (0, 255, 180))
    _draw_sls_lines(display, keypoints, side, last)
    return display


def _run_balance(display, keypoints, balance_analyser: BalanceAnalyser, floor_y):
    active = st.session_state.get("balance_session_active", False)

    live_vel = None
    if active:
        live_vel = balance_analyser.update(keypoints)
        st.session_state["balance_live_velocity"] = live_vel

    y = 35
    rec_color = (0, 0, 220) if active else (150, 150, 150)
    cv2.putText(display, "BAL REC" if active else "BAL IDLE",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, rec_color, 2)
    y += 30

    if live_vel is not None:
        cv2.putText(display, f"Sway: {live_vel:.2f}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    if len(keypoints) > 12 and keypoints[11] and keypoints[12]:
        mid_x = int((keypoints[11][0] + keypoints[12][0]) / 2)
        mid_y = int((keypoints[11][1] + keypoints[12][1]) / 2)
        cv2.circle(display, (mid_x, mid_y), 8, (0, 200, 255), -1)

    return display


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_floor_line(frame, floor_y):
    cv2.line(frame, (0, floor_y), (frame.shape[1], floor_y), (255, 255, 0), 2)


def _draw_status_badge(frame, status, value, label, y=70):
    color = {"green": (0, 220, 0), "yellow": (0, 200, 255),
             "red": (0, 0, 255), "no_baseline": (180, 180, 180)}.get(status, (180,180,180))
    cv2.putText(frame, f"{label}: {value}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)


def _highlight_kps(frame, keypoints, indices, color):
    for i in indices:
        if i < len(keypoints) and keypoints[i] is not None:
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            cv2.circle(frame, (x, y), 10, color, 2)


def _draw_sls_lines(frame, keypoints, side, result):
    idx = {"left": [11, 13, 15], "right": [12, 14, 16]}.get(side, [11, 13, 15])

    if result:
        color = {"green": (0, 220, 0), "yellow": (0, 180, 255),
                 "red": (0, 0, 255)}.get(result.status, (200, 200, 200))
    else:
        color = (200, 200, 200)

    pts = []
    for i in idx:
        if i < len(keypoints) and keypoints[i] is not None:
            pts.append((int(keypoints[i][0]), int(keypoints[i][1])))

    for a, b in zip(pts, pts[1:]):
        cv2.line(frame, a, b, color, 3) 