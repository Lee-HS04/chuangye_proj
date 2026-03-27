import streamlit as st
import cv2
import tempfile
import numpy as np
import time

from body_tracking import get_keypoints
from posture_analysis import (
    load_rules,
    extract_features,
    evaluate_multiple_rules,
    draw_feedback,
    RepCounter,
    SwayTracker,
)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Gym AI Posture Analysis")

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] { padding-top: 1rem; }
        div[data-testid="column"] button { width: 100%; font-size: 1.2rem; padding: 2px 0; }
        input[type=number]::-webkit-inner-spin-button,
        input[type=number]::-webkit-outer-spin-button { -webkit-appearance: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏋️ Gym AI Posture & Balance Analysis")

# ─────────────────────────────────────────────
# Session-state defaults
# ─────────────────────────────────────────────
defaults = {
    "floor_y":          400,
    "frame_index":      0,
    "playing":          False,
    "cap_path":         None,
    "total_frames":     0,
    "video_fps":        30,
    "uploaded_file_id": None,
    "playback_speed":   1.0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# Sidebar – rules
# ─────────────────────────────────────────────
rules_all  = load_rules("rules.txt")
rule_names = list(rules_all.keys())

selected_rules = st.sidebar.multiselect(
    "Select Rules", rule_names, default=rule_names[:1]
)

# ─────────────────────────────────────────────
# Sidebar – source
# ─────────────────────────────────────────────
source_option = st.sidebar.radio(
    "Input Source", ["Webcam", "Upload MP4 Video"]
)

# ─────────────────────────────────────────────
# Sidebar – floor line controls
# Keys are all UNIQUE; callbacks sync back to "floor_y"
# ─────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**Floor Line**")

def nudge_floor(delta: int):
    st.session_state["floor_y"] = max(0, min(1080, st.session_state["floor_y"] + delta))

def sync_from_slider():
    st.session_state["floor_y"] = st.session_state["_floor_slider"]

def sync_from_number():
    st.session_state["floor_y"] = st.session_state["_floor_number"]

col_down, col_val, col_up = st.sidebar.columns([1, 2, 1])
col_down.button("▼", on_click=nudge_floor, args=(-5,), key="floor_down")
col_up.button(  "▲", on_click=nudge_floor, args=(+5,), key="floor_up")
col_val.number_input(
    "Y",
    min_value=0, max_value=1080,
    value=st.session_state["floor_y"],
    step=1,
    key="_floor_number",
    on_change=sync_from_number,
    label_visibility="collapsed",
)
st.sidebar.slider(
    "Fine-tune floor",
    0, 1080,
    value=st.session_state["floor_y"],
    key="_floor_slider",
    on_change=sync_from_slider,
    label_visibility="collapsed",
)

# ─────────────────────────────────────────────
# Sidebar – playback speed
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# Sidebar – webcam toggle
# ─────────────────────────────────────────────
run_webcam = False
if source_option == "Webcam":
    run_webcam = st.sidebar.checkbox("Start Webcam", value=False)

# ─────────────────────────────────────────────
# Video upload handling
# ─────────────────────────────────────────────
if source_option == "Upload MP4 Video":
    uploaded_file = st.sidebar.file_uploader("Upload MP4 Video", type=["mp4"])

    if uploaded_file is not None:
        file_id = getattr(uploaded_file, "file_id", id(uploaded_file))
        if st.session_state["uploaded_file_id"] != file_id:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.flush()
            st.session_state["cap_path"]         = tfile.name
            st.session_state["uploaded_file_id"] = file_id
            st.session_state["frame_index"]      = 0
            st.session_state["playing"]          = False

            _cap = cv2.VideoCapture(tfile.name)
            st.session_state["total_frames"] = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            st.session_state["video_fps"]    = _cap.get(cv2.CAP_PROP_FPS) or 30
            _cap.release()
else:
    if st.session_state.get("cap_path"):
        st.session_state["cap_path"]    = None
        st.session_state["frame_index"] = 0
        st.session_state["playing"]     = False

# ─────────────────────────────────────────────
# Video player controls
# ─────────────────────────────────────────────
FRAME = st.empty()

if source_option == "Upload MP4 Video" and st.session_state["cap_path"]:
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
        lbl = "⏸" if st.session_state["playing"] else "▶"
        if st.button(lbl, help="Play / Pause"):
            st.session_state["playing"] = not st.session_state["playing"]

    with c4:
        if st.button("⏩", help="Next frame"):
            st.session_state["frame_index"] = min(total - 1, st.session_state["frame_index"] + 1)
            st.session_state["playing"]     = False

    with c5:
        def sync_progress():
            st.session_state["frame_index"] = st.session_state["_progress_slider"]
            st.session_state["playing"]     = False

        st.slider(
            "Progress",
            0, max(1, total - 1),
            value=st.session_state["frame_index"],
            key="_progress_slider",
            on_change=sync_progress,
            label_visibility="collapsed",
        )

    current_time = st.session_state["frame_index"] / fps
    total_time   = total / fps
    st.caption(
        f"Frame {st.session_state['frame_index']} / {total - 1}  │  "
        f"{current_time:.2f}s / {total_time:.2f}s  │  {fps:.1f} fps"
    )

# ─────────────────────────────────────────────
# Persistent trackers
# ─────────────────────────────────────────────
if "counter" not in st.session_state:
    st.session_state["counter"] = RepCounter(
        exercise="Proper Squat", feature="left_knee", min_angle=70, max_angle=110
    )
if "sway_tracker" not in st.session_state:
    st.session_state["sway_tracker"] = SwayTracker()

counter      = st.session_state["counter"]
sway_tracker = st.session_state["sway_tracker"]

# ─────────────────────────────────────────────
# Frame processor
# ─────────────────────────────────────────────
def process_frame(frame: np.ndarray) -> np.ndarray:
    floor_y = st.session_state["floor_y"]

    keypoints, annotated = get_keypoints(frame)
    display_frame = annotated.copy() if annotated is not None else frame.copy()

    if keypoints is not None:
        features = extract_features(keypoints)

        if len(keypoints) > 16 and keypoints[15] is not None and keypoints[16] is not None:
            features["jump_feet"] = floor_y - min(keypoints[15][1], keypoints[16][1])
        else:
            features["jump_feet"] = None

        if len(keypoints) > 12 and keypoints[11] is not None and keypoints[12] is not None:
            mid_hip = (
                (keypoints[11][0] + keypoints[12][0]) / 2,
                (keypoints[11][1] + keypoints[12][1]) / 2,
            )
            features["mid_hip"] = (mid_hip[0], floor_y - mid_hip[1])
        else:
            features["mid_hip"] = None

        sway_tracker.update(features.get("mid_hip"))
        sway_velocity             = sway_tracker.get_sway_velocity()
        features["sway_velocity"] = sway_velocity

        results    = evaluate_multiple_rules(features, rules_all, selected_rules)
        all_failed = set()
        y = 30

        for name, result in results.items():
            score  = result["score"]
            failed = result["failed"]
            all_failed.update(failed)
            cv2.putText(display_frame, f"{name}: {score}%",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 25
            if failed:
                cv2.putText(display_frame, "Fail: " + ", ".join(failed),
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y += 25

        draw_feedback(display_frame, keypoints, all_failed)

        reps = counter.update(features)
        cv2.putText(display_frame, f"Reps: {reps}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        y += 30

        if sway_velocity is not None:
            baseline = 5
            change   = (sway_velocity - baseline) / baseline * 100
            if change < 10:
                ftxt, fcolor = "Stable",         (0, 255, 0)
            elif change < 20:
                ftxt, fcolor = "Slight Fatigue", (0, 255, 255)
            else:
                ftxt, fcolor = "Fatigued",       (0, 0, 255)

            cv2.putText(display_frame, f"Sway: {sway_velocity:.2f}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fcolor, 2)
            y += 25
            cv2.putText(display_frame, ftxt,
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fcolor, 2)

    cv2.line(display_frame,
             (0, floor_y), (display_frame.shape[1], floor_y),
             (255, 255, 0), 2)

    return cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────
# Playback – uploaded video
# ─────────────────────────────────────────────
if source_option == "Upload MP4 Video" and st.session_state["cap_path"]:
    cap   = cv2.VideoCapture(st.session_state["cap_path"])
    total = st.session_state["total_frames"]
    fps   = st.session_state["video_fps"]

    if st.session_state["playing"]:
        while st.session_state["playing"] and st.session_state["frame_index"] < total - 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state["frame_index"])
            ret, frame = cap.read()
            if not ret:
                st.session_state["playing"] = False
                break

            if frame.shape[1] > 800:
                scale = 800 / frame.shape[1]
                frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

            FRAME.image(process_frame(frame), use_container_width=True)
            speed = st.session_state.get("playback_speed", 1.0)
            # At high speeds, skip frames instead of sleeping negative values
            step  = max(1, round(speed)) if speed >= 2.0 else 1
            delay = max(0.0, (1.0 / fps) / speed) if speed < 2.0 else 0.0
            st.session_state["frame_index"] += step
            if delay > 0:
                time.sleep(delay)

        if st.session_state["frame_index"] >= total - 1:
            st.session_state["playing"] = False

        cap.release()
        st.rerun()

    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state["frame_index"])
        ret, frame = cap.read()
        cap.release()

        if ret:
            if frame.shape[1] > 800:
                scale = 800 / frame.shape[1]
                frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            FRAME.image(process_frame(frame), use_container_width=True)

# ─────────────────────────────────────────────
# Playback – webcam
# ─────────────────────────────────────────────
elif source_option == "Webcam" and run_webcam:
    cap      = cv2.VideoCapture(0)
    stop_btn = st.button("⏹ Stop Webcam")

    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.shape[1] > 800:
            scale = 800 / frame.shape[1]
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        FRAME.image(process_frame(frame), use_container_width=True)

    cap.release()

elif source_option == "Webcam" and not run_webcam:
    st.info("Enable **Start Webcam** in the sidebar to begin.")