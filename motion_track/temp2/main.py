# main.py
import time
import cv2
import streamlit as st

from config import DEFAULTS
from ui.sidebar import setup_sidebar
from ui.video_controls import video_controls
from ui.video_upload import handle_video_upload
from ui.frame_display import process_frame
from core.counters import RepCounter, SwayTracker
from body_tracking import get_keypoints
from posture_analysis import load_rules

# ────────────── Page Config ──────────────
st.set_page_config(layout="wide", page_title="R2P Ready-to-Play Guard")
st.title("R2P Ready-to-Play Guard")

# ────────────── Session Defaults ──────────────
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ────────────── Load Rules ──────────────
rules_all = load_rules("assets/rules.txt")
default_rules = list(rules_all.keys())[:3]

# ────────────── Sidebar ──────────────
selected_rules, source_option, run_webcam = setup_sidebar(rules_all, default_rules)

# ────────────── Video Upload ──────────────
if source_option == "Upload MP4 Video":
    handle_video_upload()

# ────────────── Persistent Counters / Trackers ──────────────
if "cmj_counter" not in st.session_state:
    st.session_state["cmj_counter"] = RepCounter(
        "CMJ", "jump_feet", min_angle=10, max_angle=50
    )
if "sls_counter" not in st.session_state:
    st.session_state["sls_counter"] = RepCounter(
        "SLS", "FPPA_left", min_angle=70, max_angle=110
    )
if "balance_tracker" not in st.session_state:
    st.session_state["balance_tracker"] = SwayTracker()

cmj_counter     = st.session_state["cmj_counter"]
sls_counter     = st.session_state["sls_counter"]
balance_tracker = st.session_state["balance_tracker"]


# ────────────── Helper: annotate + push one frame ──────────────
def _render_frame(cap: cv2.VideoCapture, frame_placeholder) -> bool:
    """
    Read frame at current frame_index, annotate it, push to placeholder.
    Returns False if the frame could not be read.
    """
    idx = int(st.session_state["frame_index"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        return False

    if frame.shape[1] > 800:
        scale = 800 / frame.shape[1]
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

    keypoints, annotated = get_keypoints(frame)
    if annotated is not None:
        frame = annotated

    frame_placeholder.image(
        process_frame(
            frame, keypoints, selected_rules, rules_all,
            st.session_state["floor_y"],
            cmj_counter, sls_counter, balance_tracker,
        ),
        use_container_width=True,
    )
    return True


# ────────────── MP4 Playback ──────────────
if source_option == "Upload MP4 Video" and st.session_state.get("cap_path"):
    total_frames = st.session_state.get("total_frames", 1)
    fps          = st.session_state.get("video_fps", 30)

    # Controls live in a fixed container — rendered once before the loop
    # so they stay visible and clickable throughout playback.
    controls_area = st.container()
    frame_area    = st.empty()

    with controls_area:
        video_controls(total_frames, fps)

    cap = cv2.VideoCapture(st.session_state["cap_path"])

    if st.session_state.get("playing"):
        # ── Tight playback loop ───────────────────────────────────────────
        while True:
            # Re-read speed each frame so live slider changes take effect
            speed          = st.session_state.get("playback_speed_video", 1.0)
            frame_duration = 1.0 / (fps * speed)

            # Bail out if pause/seek happened via a sidebar callback
            if not st.session_state.get("playing"):
                break

            idx = int(st.session_state["frame_index"])
            if idx >= total_frames - 1:
                st.session_state["playing"] = False
                st.session_state["frame_index"] = total_frames - 1
                break

            t0 = time.perf_counter()

            ok = _render_frame(cap, frame_area)
            if not ok:
                st.session_state["playing"] = False
                break

            # Advance index (fractional accumulation handles sub-1x speeds)
            st.session_state["frame_index"] = min(
                total_frames - 1,
                st.session_state["frame_index"] + speed,
            )

            # Sleep for the remainder of the frame budget
            elapsed = time.perf_counter() - t0
            sleep_t = max(0.0, frame_duration - elapsed)
            if sleep_t:
                time.sleep(sleep_t)

    else:
        # ── Paused: render the current frame once ─────────────────────────
        _render_frame(cap, frame_area)

    cap.release()


# ────────────── Webcam ──────────────
elif source_option == "Webcam" and run_webcam:
    frame_area = st.empty()
    stop_btn   = st.button("⏹ Stop Webcam")
    cap        = cv2.VideoCapture(0)

    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret:
            break

        if frame.shape[1] > 800:
            scale = 800 / frame.shape[1]
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        keypoints, annotated = get_keypoints(frame)
        if annotated is not None:
            frame = annotated

        frame_area.image(
            process_frame(
                frame, keypoints, selected_rules, rules_all,
                st.session_state["floor_y"],
                cmj_counter, sls_counter, balance_tracker,
            ),
            use_container_width=True,
        )

    cap.release()

elif source_option == "Webcam" and not run_webcam:
    st.info("Enable **Start Webcam** in the sidebar to begin.")

# # main.py
# import time
# import cv2
# import streamlit as st

# from config import DEFAULTS
# from ui.sidebar import setup_sidebar
# from ui.video_controls import video_controls
# from ui.video_upload import handle_video_upload
# from ui.frame_display import process_frame
# from posture_analysis import load_rules, RepCounter, SwayTracker

# # ────────────── Page Config ──────────────
# st.set_page_config(layout="wide", page_title="R2P Ready-to-Play Guard")
# st.title("R2P Ready-to-Play Guard")

# # ────────────── Session Defaults ──────────────
# for k, v in DEFAULTS.items():
#     if k not in st.session_state:
#         st.session_state[k] = v

# # ────────────── Load Rules ──────────────
# rules_all     = load_rules("assets/rules.txt")
# default_rules = list(rules_all.keys())[:3]

# # ────────────── Sidebar ──────────────
# # setup_sidebar now returns 4 values — exercise_name is new
# selected_rules, source_option, run_webcam, exercise_name = setup_sidebar(
#     rules_all, default_rules
# )

# # ────────────── Video Upload ──────────────
# if source_option == "Upload MP4 Video":
#     handle_video_upload()

# # ────────────── Persistent Counters / Trackers ──────────────
# if "cmj_counter" not in st.session_state:
#     st.session_state["cmj_counter"] = RepCounter(
#         "CMJ", "jump_feet", min_angle=10, max_angle=50
#     )
# if "sls_counter" not in st.session_state:
#     st.session_state["sls_counter"] = RepCounter(
#         "SLS", "FPPA_left", min_angle=70, max_angle=110
#     )
# if "balance_tracker" not in st.session_state:
#     st.session_state["balance_tracker"] = SwayTracker()

# cmj_counter     = st.session_state["cmj_counter"]
# sls_counter     = st.session_state["sls_counter"]
# balance_tracker = st.session_state["balance_tracker"]


# # ────────────── Helper: annotate + push one frame ──────────────
# def _render_frame(cap: cv2.VideoCapture, frame_placeholder) -> bool:
#     idx = int(st.session_state["frame_index"])
#     cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#     ret, frame = cap.read()
#     if not ret:
#         return False

#     if frame.shape[1] > 800:
#         scale = 800 / frame.shape[1]
#         frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

#     from body_tracking import get_keypoints
#     keypoints, annotated = get_keypoints(frame)
#     if annotated is not None:
#         frame = annotated

#     frame_placeholder.image(
#         process_frame(
#             frame, keypoints, selected_rules, rules_all,
#             st.session_state["floor_y"],
#             cmj_counter, sls_counter, balance_tracker,
#             exercise_name=exercise_name,
#         ),
#         use_container_width=True,
#     )
#     return True


# # ────────────── MP4 Playback ──────────────
# if source_option == "Upload MP4 Video" and st.session_state.get("cap_path"):
#     total_frames = st.session_state.get("total_frames", 1)
#     fps          = st.session_state.get("video_fps", 30)

#     controls_area = st.container()
#     frame_area    = st.empty()

#     with controls_area:
#         video_controls(total_frames, fps)

#     cap = cv2.VideoCapture(st.session_state["cap_path"])

#     if st.session_state.get("playing"):
#         while True:
#             speed          = st.session_state.get("playback_speed_video", 1.0)
#             frame_duration = 1.0 / (fps * speed)

#             if not st.session_state.get("playing"):
#                 break

#             idx = int(st.session_state["frame_index"])
#             if idx >= total_frames - 1:
#                 st.session_state["playing"]     = False
#                 st.session_state["frame_index"] = total_frames - 1
#                 break

#             t0 = time.perf_counter()

#             ok = _render_frame(cap, frame_area)
#             if not ok:
#                 st.session_state["playing"] = False
#                 break

#             st.session_state["frame_index"] = min(
#                 total_frames - 1,
#                 st.session_state["frame_index"] + speed,
#             )

#             elapsed = time.perf_counter() - t0
#             sleep_t = max(0.0, frame_duration - elapsed)
#             if sleep_t:
#                 time.sleep(sleep_t)

#     else:
#         _render_frame(cap, frame_area)

#     cap.release()


# # ────────────── Webcam ──────────────
# elif source_option == "Webcam" and run_webcam:
#     from body_tracking import get_keypoints
#     frame_area = st.empty()
#     stop_btn   = st.button("⏹ Stop Webcam")
#     cap        = cv2.VideoCapture(0)

#     while cap.isOpened() and not stop_btn:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if frame.shape[1] > 800:
#             scale = 800 / frame.shape[1]
#             frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

#         keypoints, annotated = get_keypoints(frame)
#         if annotated is not None:
#             frame = annotated

#         frame_area.image(
#             process_frame(
#                 frame, keypoints, selected_rules, rules_all,
#                 st.session_state["floor_y"],
#                 cmj_counter, sls_counter, balance_tracker,
#                 exercise_name=exercise_name,
#             ),
#             use_container_width=True,
#         )

#     cap.release()

# elif source_option == "Webcam" and not run_webcam:
#     st.info("Enable **Start Webcam** in the sidebar to begin.")