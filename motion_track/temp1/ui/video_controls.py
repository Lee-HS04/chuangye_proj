import streamlit as st

def render_video_controls(total, fps):
    ss = st.session_state
    c1, c2, c3, c4, c5 = st.columns([1,1,1,1,5])

    with c1:
        if st.button("⏮", key="start_video"):
            ss["frame_index"] = 0
            ss["playing"] = False
    with c2:
        if st.button("⏪", key="prev_frame"):
            ss["frame_index"] = max(0, ss["frame_index"]-1)
            ss["playing"] = False
    with c3:
        lbl = "⏸" if ss["playing"] else "▶"
        if st.button(lbl, key="play_pause"):
            ss["playing"] = not ss["playing"]
    with c4:
        if st.button("⏩", key="next_frame"):
            ss["frame_index"] = min(total-1, ss["frame_index"]+1)
            ss["playing"] = False
    with c5:
        def sync_progress():
            ss["frame_index"] = ss["_progress_slider"]
            ss["playing"] = False
        st.slider("Progress", 0, max(1,total-1),
                  value=ss["frame_index"], key="_progress_slider",
                  on_change=sync_progress, label_visibility="collapsed")
    current_time = ss["frame_index"] / fps
    st.caption(f"Frame {ss['frame_index']} / {total-1}  │  {current_time:.2f}s / {total/fps:.2f}s  │  {fps:.1f} fps")