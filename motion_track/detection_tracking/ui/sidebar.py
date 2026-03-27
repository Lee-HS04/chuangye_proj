"""
ui/sidebar.py
-------------
All sidebar widgets.
Returns a SidebarConfig dataclass so main.py has one clean object to read.
"""
from __future__ import annotations

import tempfile
import cv2
import streamlit as st
from dataclasses import dataclass
from posture_analysis import load_rules


@dataclass
class SidebarConfig:
    analysis_mode:  str          # "posture" | "cmj" | "sls" | "balance"
    selected_rules: list[str]    # only used in "posture" mode
    source_option:  str          # "Webcam" | "Upload MP4 Video"
    run_webcam:     bool
    sls_side:       str          # "left" | "right"


# ── Floor helpers ─────────────────────────────────────────────────────────────

def _nudge_floor(delta: int) -> None:
    st.session_state["floor_y"] = max(0, min(1080, st.session_state["floor_y"] + delta))

def _sync_floor_slider() -> None:
    st.session_state["floor_y"] = st.session_state["_floor_slider"]

def _sync_floor_number() -> None:
    st.session_state["floor_y"] = st.session_state["_floor_number"]


# ── Public ────────────────────────────────────────────────────────────────────

def render_sidebar(rules_path: str = "rules.txt") -> SidebarConfig:
    """Render every sidebar section and return the collected config."""

    # ── Analysis mode ─────────────────────────────────────────────────────
    st.sidebar.markdown("### R2P-Guard")
    mode = st.sidebar.radio(
        "Analysis mode",
        ["posture", "cmj", "sls", "balance"],
        format_func=lambda m: {
            "posture": "🏋️ Posture",
            "cmj":     "💥 CMJ — Jump",
            "sls":     "🦵 SLS — Squat",
            "balance": "⚖️ Balance",
        }[m],
        key="analysis_mode",
    )

    # ── Posture rules (only shown in posture mode) ─────────────────────────
    selected_rules: list[str] = []
    if mode == "posture":
        rules_all  = load_rules(rules_path)
        rule_names = list(rules_all.keys())
        selected_rules = st.sidebar.multiselect(
            "Select Rules", rule_names, default=rule_names[:1]
        )

    # ── SLS side (only shown in SLS mode) ─────────────────────────────────
    sls_side = st.session_state.get("sls_side", "left")
    if mode == "sls":
        sls_side = st.sidebar.radio(
            "Active leg", ["left", "right"],
            index=0 if sls_side == "left" else 1,
            horizontal=True,
            key="sls_side_sidebar",
        )
        st.session_state["sls_side"] = sls_side

    # ── Input source ──────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    source_option = st.sidebar.radio(
        "Input Source", ["Webcam", "Upload MP4 Video"]
    )

    # ── Floor line ────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Floor Line**")

    col_down, col_val, col_up = st.sidebar.columns([1, 2, 1])
    col_down.button("▼", on_click=_nudge_floor, args=(-5,), key="floor_down")
    col_up.button(  "▲", on_click=_nudge_floor, args=(+5,), key="floor_up")
    col_val.number_input(
        "Y",
        min_value=0, max_value=1080,
        value=st.session_state["floor_y"],
        step=1,
        key="_floor_number",
        on_change=_sync_floor_number,
        label_visibility="collapsed",
    )
    st.sidebar.slider(
        "Fine-tune floor", 0, 1080,
        value=st.session_state["floor_y"],
        key="_floor_slider",
        on_change=_sync_floor_slider,
        label_visibility="collapsed",
    )

    # ── Playback speed ────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Playback Speed**")
    st.sidebar.select_slider(
        "Speed",
        options=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0],
        value=st.session_state.get("playback_speed", 1.0),
        format_func=lambda x: f"{x}×",
        key="playback_speed",
    )
    st.sidebar.caption(
        f"Playing at {st.session_state['playback_speed']}× — "
        f"native {st.session_state.get('video_fps', 30):.1f} fps"
    )

    # ── Webcam toggle ─────────────────────────────────────────────────────
    run_webcam = False
    if source_option == "Webcam":
        run_webcam = st.sidebar.checkbox("Start Webcam", value=False)

    # ── Video upload ──────────────────────────────────────────────────────
    if source_option == "Upload MP4 Video":
        _handle_upload()
    else:
        _clear_video_state()

    return SidebarConfig(
        analysis_mode  = mode,
        selected_rules = selected_rules,
        source_option  = source_option,
        run_webcam     = run_webcam,
        sls_side       = st.session_state.get("sls_side", "left"),
    )


# ── Upload helpers ────────────────────────────────────────────────────────────

def _handle_upload() -> None:
    uploaded_file = st.sidebar.file_uploader("Upload MP4 Video", type=["mp4"])

    if uploaded_file is None:
        return

    file_id = getattr(uploaded_file, "file_id", id(uploaded_file))
    if st.session_state["uploaded_file_id"] == file_id:
        return

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.flush()

    cap = cv2.VideoCapture(tfile.name)
    st.session_state.update({
        "cap_path":         tfile.name,
        "uploaded_file_id": file_id,
        "frame_index":      0,
        "playing":          False,
        "total_frames":     int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "video_fps":        cap.get(cv2.CAP_PROP_FPS) or 30.0,
    })
    cap.release()


def _clear_video_state() -> None:
    if st.session_state.get("cap_path"):
        st.session_state.update({
            "cap_path":    None,
            "frame_index": 0,
            "playing":     False,
        })