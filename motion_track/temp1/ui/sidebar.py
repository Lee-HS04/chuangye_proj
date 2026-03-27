import streamlit as st

def render_rules_sidebar(rules_all):
    rule_names = list(rules_all.keys())
    selected_rules = st.sidebar.multiselect(
        "Select Rules", rule_names, default=rule_names[:1]
    )
    return selected_rules

def render_source_sidebar():
    source_option = st.sidebar.radio("Input Source", ["Webcam", "Upload MP4 Video"])
    run_webcam = False
    if source_option == "Webcam":
        run_webcam = st.sidebar.checkbox("Start Webcam", value=False)
    return source_option, run_webcam

def render_floor_sidebar(floor_y):
    ss = st.session_state

    def nudge_floor(delta: int):
        ss["floor_y"] = max(0, min(1080, ss["floor_y"] + delta))
    def sync_from_slider():
        ss["floor_y"] = ss["_floor_slider"]
    def sync_from_number():
        ss["floor_y"] = ss["_floor_number"]

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Floor Line**")
    col_down, col_val, col_up = st.sidebar.columns([1,2,1])
    col_down.button("▼", on_click=nudge_floor, args=(-5,))
    col_up.button("▲", on_click=nudge_floor, args=(5,))
    col_val.number_input(
        "Y", min_value=0, max_value=1080, value=floor_y,
        step=1, key="_floor_number", on_change=sync_from_number, label_visibility="collapsed"
    )
    st.sidebar.slider("Fine-tune floor", 0, 1080, value=floor_y,
                      key="_floor_slider", on_change=sync_from_slider,
                      label_visibility="collapsed")

def render_playback_speed_sidebar(video_fps, playback_speed):
    speed = st.sidebar.select_slider(
        "Speed", options=[0.25,0.5,0.75,1.0,1.5,2.0,4.0,8.0],
        value=playback_speed, format_func=lambda x: f"{x}×"
    )
    st.sidebar.caption(f"Playing at {speed}× — native {video_fps:.1f} fps")
    return speed