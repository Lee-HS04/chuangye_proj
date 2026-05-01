import streamlit as st
from core.counters import R2PScorer, RepCounter, SwayTracker, extract_features, CMJTracker, calculate_fppa

def video_controls(total_frames: int, fps: float) -> None:
    """
    Render video transport controls.
    Must be called OUTSIDE the playback loop so controls are always visible.
    All state mutations happen via callbacks so Streamlit reruns pick them up.
    """

    c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 1, 4, 2])

    # ── Go to start ──────────────────────────────────────────────────────────
    def _go_start():
        st.session_state["frame_index"] = 0
        st.session_state["playing"] = False
#reset trackers history
        st.session_state["sway_tracker"].reset()
        st.session_state["sls_counter"].reset()
        st.session_state["cmj_counter"].reset()


    with c1:
        st.button("⏮", key="start_btn", on_click=_go_start)

    # ── Step back one frame ───────────────────────────────────────────────────
    def _prev():
        st.session_state["frame_index"] = max(
            0, int(st.session_state["frame_index"]) - 1
        )
        st.session_state["playing"] = False

    with c2:
        st.button("⏪", key="prev_btn", on_click=_prev)

    # ── Play / Pause toggle ───────────────────────────────────────────────────
    def _toggle_play():
        st.session_state["playing"] = not st.session_state.get("playing", False)

    with c3:
        label = "⏸" if st.session_state.get("playing") else "▶"
        st.button(label, key="play_pause_btn", on_click=_toggle_play)

    # ── Step forward one frame ────────────────────────────────────────────────
    def _next():
        st.session_state["frame_index"] = min(
            total_frames - 1, int(st.session_state["frame_index"]) + 1
        )
        st.session_state["playing"] = False

    with c4:
        st.button("⏩", key="next_btn", on_click=_next)

    # ── Seek slider ───────────────────────────────────────────────────────────
    def _seek():
        st.session_state["frame_index"] = st.session_state["_progress_slider"]
        st.session_state["playing"] = False

    with c5:
        st.slider(
            "Progress",
            0,
            max(1, total_frames - 1),
            value=int(st.session_state.get("frame_index", 0)),
            key="_progress_slider",
            on_change=_seek,
            label_visibility="collapsed",
        )

    # ── Playback speed ────────────────────────────────────────────────────────
    _speeds = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0]
    with c6:
        st.selectbox(
            "Speed",
            _speeds,
            index=_speeds.index(st.session_state.get("playback_speed_video", 1.0)),
            key="playback_speed_video",
        )

    # ── Time readout ──────────────────────────────────────────────────────────
    fidx = int(st.session_state.get("frame_index", 0))
    current_time = fidx / fps if fps else 0
    total_time = total_frames / fps if fps else 0
    st.caption(
        f"Frame {fidx}/{total_frames - 1} | "
        f"{current_time:.2f}s / {total_time:.2f}s | "
        f"{fps:.1f} fps | Speed {st.session_state['playback_speed_video']}x"
    )
# # ui/video_controls.py
# import streamlit as st

# def video_controls(total_frames, fps):
#     c1, c2, c3, c4, c5, c6 = st.columns([1,1,1,1,4,2])

#     # Start
#     with c1:
#         if st.button("⏮", key="start_btn"):
#             st.session_state["frame_index"] = 0
#             st.session_state["playing"] = False

#     # Previous
#     with c2:
#         if st.button("⏪", key="prev_btn"):
#             st.session_state["frame_index"] = max(0, st.session_state["frame_index"] - 1)
#             st.session_state["playing"] = False

#     # Play / Pause
#     with c3:
#         lbl = "⏸" if st.session_state.get("playing") else "▶"
#         if st.button(lbl, key="play_pause_btn"):
#             st.session_state["playing"] = not st.session_state.get("playing", False)

#     # Next
#     with c4:
#         if st.button("⏩", key="next_btn"):
#             st.session_state["frame_index"] = min(total_frames-1, st.session_state["frame_index"] + 1)
#             st.session_state["playing"] = False

#     # Slider
#     def slider_update():
#         st.session_state["frame_index"] = st.session_state["_progress_slider"]
#         st.session_state["playing"] = False

#     with c5:
#         st.slider(
#             "Progress",
#             0, max(1,total_frames-1),
#             value=int(st.session_state.get("frame_index",0)),
#             key="_progress_slider",
#             on_change=slider_update,
#             label_visibility="collapsed"
#         )

#     # Playback speed (unique key)
#     with c6:
#         st.selectbox(
#             "Speed", [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0],
#             index=[0.25,0.5,0.75,1.0,1.5,2.0,4.0,8.0].index(st.session_state.get("playback_speed_video", 1.0)),
#             key="playback_speed_video"
#         )

#     # Time display
#     current_time = st.session_state.get("frame_index",0) / fps
#     total_time = total_frames / fps
#     st.caption(f"Frame {int(st.session_state.get('frame_index',0))}/{total_frames-1} | "
#                f"{current_time:.2f}s / {total_time:.2f}s | {fps:.1f} fps | Speed {st.session_state['playback_speed_video']}x")