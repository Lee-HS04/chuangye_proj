"""
ui/player.py
------------
Video player transport controls (⏮ ⏪ ▶/⏸ ⏩ + progress scrubber).
Call `render_player_controls()` only when a video is loaded.
"""
import streamlit as st


def _sync_progress() -> None:
    st.session_state["frame_index"] = st.session_state["_progress_slider"]
    st.session_state["playing"]     = False


def render_player_controls() -> None:
    """Render the transport bar and progress slider for uploaded video."""
    total = st.session_state["total_frames"]
    fps   = st.session_state["video_fps"]

    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 5])

    with c1:
        if st.button("⏮", help="Go to start"):
            st.session_state["frame_index"] = 0
            st.session_state["playing"]     = False

    with c2:
        if st.button("⏪", help="Previous frame"):
            st.session_state["frame_index"] = max(0, st.session_state["frame_index"] - 1)
            st.session_state["playing"]     = False

    with c3:
        label = "⏸" if st.session_state["playing"] else "▶"
        if st.button(label, help="Play / Pause"):
            st.session_state["playing"] = not st.session_state["playing"]

    with c4:
        if st.button("⏩", help="Next frame"):
            st.session_state["frame_index"] = min(total - 1, st.session_state["frame_index"] + 1)
            st.session_state["playing"]     = False

    with c5:
        st.slider(
            "Progress",
            0, max(1, total - 1),
            value=st.session_state["frame_index"],
            key="_progress_slider",
            on_change=_sync_progress,
            label_visibility="collapsed",
        )

    current_time = st.session_state["frame_index"] / fps
    total_time   = total / fps
    st.caption(
        f"Frame {st.session_state['frame_index']} / {total - 1}  │  "
        f"{current_time:.2f}s / {total_time:.2f}s  │  {fps:.1f} fps"
    )