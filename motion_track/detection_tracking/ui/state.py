"""
ui/state.py
-----------
Centralised session-state defaults.
Call `init_state()` once at the top of main.py.
"""
import streamlit as st

DEFAULTS: dict = {
    # ── Video / playback ──────────────────────────────────────────────────────
    "floor_y":                  400,
    "frame_index":              0,
    "playing":                  False,
    "cap_path":                 None,
    "total_frames":             0,
    "video_fps":                30.0,
    "uploaded_file_id":         None,
    "playback_speed":           1.0,

    # ── Analysis mode ─────────────────────────────────────────────────────────
    # "posture" | "cmj" | "sls" | "balance"
    "analysis_mode":            "posture",

    # ── CMJ ───────────────────────────────────────────────────────────────────
    "cmj_baseline":             None,   # float | None

    # ── SLS ───────────────────────────────────────────────────────────────────
    "sls_side":                 "left", # "left" | "right"

    # ── Balance ───────────────────────────────────────────────────────────────
    "balance_baseline":         None,   # float | None
    "balance_session_active":   False,
    "balance_result":           None,
    "balance_live_velocity":    None,
}


def init_state() -> None:
    """Populate session state with defaults for any missing key."""
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value