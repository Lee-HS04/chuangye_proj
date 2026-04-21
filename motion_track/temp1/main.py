import streamlit as st
import cv2
import tempfile
import time
from config import DEFAULTS
from ui import sidebar, video_controls, frame_display
from core.counters import RepCounter, SwayTracker
from core.posture import process_keypoints
from posture_analysis import load_rules
from core.utils import resize_if_needed

# ─────────────────────────────────────────────
# Session state defaults
for k,v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.set_page_config(layout="wide", page_title="Gym AI Posture Analysis")
st.title("🏋️ Gym AI Posture & Balance Analysis")

# ─────────────────────────────────────────────
# Load rules
rules_all = load_rules("assets/rules.txt")
selected_rules = sidebar.render_rules_sidebar(rules_all)

# ─────────────────────────────────────────────
# Sidebar controls
source_option, run_webcam = sidebar.render_source_sidebar()
st.session_state["playback_speed"] = sidebar.render_playback_speed_sidebar(
    st.session_state["video_fps"], st.session_state["playback_speed"]
)
sidebar.render_floor_sidebar(st.session_state["floor_y"])

# ─────────────────────────────────────────────
# Persistent trackers
if "counter" not in st.session_state:
    st.session_state["counter"] = RepCounter(
        exercise="Proper Squat", feature="left_knee", min_angle=70, max_angle=110
    )
if "sway_tracker" not in st.session_state:
    st.session_state["sway_tracker"] = SwayTracker()

FRAME = st.empty()

# ─────────────────────────────────────────────
# Video source handling
cap_path = st.session_state.get("cap_path")
if source_option == "Upload MP4 Video":
    uploaded_file = st.sidebar.file_uploader("Upload MP4 Video", type=["mp4"])
    if uploaded_file:
        file_id = getattr(uploaded_file, "file_id", id(uploaded_file))
        if st.session_state.get("uploaded_file_id") != file_id:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.flush()
            st.session_state["cap_path"] = tfile.name
            st.session_state["uploaded_file_id"] = file_id
            st.session_state["frame_index"] = 0
            st.session_state["playing"] = False
            _cap = cv2.VideoCapture(tfile.name)
            st.session_state["total_frames"] = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            st.session_state["video_fps"] = _cap.get(cv2.CAP_PROP_FPS) or 30
            _cap.release()

# ─────────────────────────────────────────────
# Frame rendering
if source_option == "Upload MP4 Video" and cap_path:
    total = st.session_state["total_frames"]
    fps = st.session_state["video_fps"]
    video_controls.render_video_controls(total, fps)

    cap = cv2.VideoCapture(cap_path)
    frame_index = st.session_state["frame_index"]

    if st.session_state["playing"]:
        while st.session_state["playing"] and frame_index < total-1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                st.session_state["playing"] = False
                break
            frame = resize_if_needed(frame)
            FRAME.image(frame_display.render_frame(
                frame, st.session_state["floor_y"], rules_all, selected_rules,
                st.session_state["counter"], st.session_state["sway_tracker"]
            ), use_container_width=True)
            step = max(1, round(st.session_state["playback_speed"]))
            frame_index += step
            st.session_state["frame_index"] = frame_index
            time.sleep(max(0.0, 1.0 / fps / st.session_state["playback_speed"]))
        cap.release()
        st.rerun()
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame = resize_if_needed(frame)
            FRAME.image(frame_display.render_frame(
                frame, st.session_state["floor_y"], rules_all, selected_rules,
                st.session_state["counter"], st.session_state["sway_tracker"]
            ), use_container_width=True)

elif source_option == "Webcam" and run_webcam:
    cap = cv2.VideoCapture(0)
    stop_btn = st.button("⏹ Stop Webcam")
    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret: break
        frame = resize_if_needed(frame)
        FRAME.image(frame_display.render_frame(
            frame, st.session_state["floor_y"], rules_all, selected_rules,
            st.session_state["counter"], st.session_state["sway_tracker"]
        ), use_container_width=True)
    cap.release()
elif source_option == "Webcam" and not run_webcam:
    st.info("Enable **Start Webcam** in the sidebar to begin.")