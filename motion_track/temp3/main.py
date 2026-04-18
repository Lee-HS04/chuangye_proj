# main.py
import os
import time
import cv2
import streamlit as st

from config import DEFAULTS
from ui.sidebar import setup_sidebar
from ui.video_controls import video_controls
from ui.video_upload import handle_video_upload
from ui.frame_display import process_frame
from core.counters import R2PScorer, RepCounter, SwayTracker, extract_features, CMJTracker, calculate_fppa
from body_tracking import process_video_gvhmr, project_3d_to_2d, smpl_to_coco17, get_yolo26_keypoints
from remote_ssh_pipeline import process_video_on_remote
from posture_analysis import load_rules

# ────────────── Page Config ──────────────
st.set_page_config(layout="wide", page_title="R2P Ready-to-Play Guard")
st.title("R2P Ready-to-Play Guard")

# Global storage for trackers
if "sway_tracker" not in st.session_state:
    st.session_state["sway_tracker"] = SwayTracker(fps=30)

# ────────────── Session Defaults ──────────────
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ────────────── Load Rules ──────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RULES_PATH = os.path.join(BASE_DIR, "assets", "rules.txt")
rules_all = load_rules(RULES_PATH)
default_rules = list(rules_all.keys())[:3]

# ────────────── Sidebar ──────────────
# selected_rules, source_option, run_webcam = setup_sidebar(rules_all, default_rules)
selected_rules, source_option, run_webcam, exercise_name = setup_sidebar(rules_all, default_rules)

# ────────────── Video Upload ──────────────
if source_option == "Upload MP4 Video":
    handle_video_upload()
    
    cap_path = st.session_state.get("cap_path")
    if cap_path:
        f_mm = st.session_state.get("camera_f_mm", 24)
        
        # Conditionally skip GVHMR processing for Balance/Frontal tests
        # To temporarily disable YOLO and test GVHMR, set use_yolo26 = False
        # use_yolo26 = exercise_name in ["SLS", "Balance"] 
        use_yolo26 = False # Uncomment this line to force GVHMR mapping for all exercises
        st.session_state["use_yolo26"] = use_yolo26
        
        if not use_yolo26:
            if st.session_state.get("gvhmr_path") != cap_path or st.session_state.get("gvhmr_f_mm") != f_mm:
                with st.spinner("Processing video remotely on SSH GPU Server... This usually takes ~1 minute. Please wait..."):
                    gvhmr_results = process_video_on_remote(cap_path, f_mm=f_mm) # Uses GPU server now!
                    st.session_state["gvhmr_results"] = gvhmr_results

                    gvhmr_results = process_video_on_remote(cap_path, f_mm=f_mm)

                    print("\n=== GVHMR DEBUG ===")
                    print("num frames:",
                        len(gvhmr_results["joints_3d_global"]))

                    print("sample hip frame 0:",
                        gvhmr_results["joints_3d_global"][0][11],
                        gvhmr_results["joints_3d_global"][0][12])

                    if len(gvhmr_results["joints_3d_global"]) > 10:
                        print("sample hip frame 10:",
                            gvhmr_results["joints_3d_global"][10][11],
                            gvhmr_results["joints_3d_global"][10][12])






                    st.session_state["gvhmr_path"] = cap_path
                    st.session_state["gvhmr_f_mm"] = f_mm
                
                if gvhmr_results is None:
                    st.error("⚠️ GVHMR processing failed. No 3D tracking data was generated.")
        else:
            # Tell the system we are using local YOLO
            st.session_state["gvhmr_results"] = None
            st.info("⚡ Real-time YOLO26 selected for frontal balance tests. Bypassing GVHMR SSH processing.")


# ────────────── Persistent Counters / Trackers ──────────────
if "cmj_counter" not in st.session_state:
    st.session_state["cmj_counter"] = CMJTracker(fps=30)

if "sls_counter" not in st.session_state:
    st.session_state["sls_counter"] = RepCounter(
        exercise="SLS",
        feature="sls_fppa",
        min_angle=5,
        max_angle=15
    )

if "balance_tracker" not in st.session_state:
    st.session_state["balance_tracker"] = SwayTracker(fps=30)

if "r2p_scorer" not in st.session_state:
    st.session_state["r2p_scorer"] = R2PScorer()


cmj_counter = st.session_state["cmj_counter"]
sls_counter = st.session_state["sls_counter"]
balance_tracker = st.session_state["balance_tracker"]
r2p_scorer = st.session_state["r2p_scorer"]

def draw_skeleton(frame, keypoints_2d):
    if not keypoints_2d:
        return frame
    
    edges = [
        (0,1),(0,2),(1,3),(2,4),
        (5,7),(7,9),(6,8),(8,10),
        (5,6), (5,11),(6,12), (11,12),
        (11,13),(13,15), (12,14),(14,16)
    ]
    for pt in keypoints_2d:
        if pt is not None:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
            
    for p1, p2 in edges:
        kp1, kp2 = keypoints_2d[p1], keypoints_2d[p2]
        if kp1 is not None and kp2 is not None:
             cv2.line(frame, (int(kp1[0]), int(kp1[1])), (int(kp2[0]), int(kp2[1])), (0, 0, 255), 2)
    return frame

# ────────────── Helper: annotate + push one frame ──────────────
def _render_frame(cap: cv2.VideoCapture, frame_placeholder) -> bool:
    """
    Read frame at current frame_index, annotate it, push to placeholder.
    Returns False if the frame could not be read.
    """
    idx = int(st.session_state["frame_index"])
    
    # Start timer for frame processing
    frame_perf_start = time.perf_counter()
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        return False

    # Resize for display if too large
    if frame.shape[1] > 800:
        scale = 800 / frame.shape[1]
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    else:
        scale = 1.0

    keypoints_3d = None
    keypoints_2d = None
    annotated = frame.copy()

    # Ensure GVHMR results exist
    gvhmr_results = st.session_state.get("gvhmr_results")
    use_yolo26 = st.session_state.get("use_yolo26", False)
    
    if use_yolo26:
        # Frontal balance test -> Use Local YOLO26
        # Get 2D (x,y) keypoints directly
        keypoints_2d_raw = get_yolo26_keypoints(frame)
        
        # In YOLO 2D tests, our 3D logic simply utilizes X,Y in calculate_frontal_projection_angle()
        # So we can pass the exact same 2D data as 3D (math logic applies indices [0] and [1]).
        # Add a dummy Z dimension (0.0) so the rest of the 3D-oriented system doesn't crash on len().
        keypoints_3d = [(pt[0], pt[1], 0.0) if pt else None for pt in keypoints_2d_raw]
        
        # Scale for display
        if scale != 1.0:
            keypoints_2d = [(int(pt[0]*scale), int(pt[1]*scale)) if pt is not None else None for pt in keypoints_2d_raw]
        else:
            keypoints_2d = [(int(pt[0]), int(pt[1])) if pt is not None else None for pt in keypoints_2d_raw]
            
        if st.session_state.get("show_skeleton", True):
            annotated = draw_skeleton(annotated, keypoints_2d)

    elif gvhmr_results is not None:
        # Standard Profile -> Use Remote GVHMR
        num_frames = len(gvhmr_results["joints_3d_global"])
        idx_bounded = min(idx, num_frames - 1)
        
        smpl_global = gvhmr_results["joints_3d_global"][idx_bounded]
        smpl_incam = gvhmr_results["joints_3d_incam"][idx_bounded]
        
        K = gvhmr_results["K_fullimg"][idx_bounded] if gvhmr_results["K_fullimg"].ndim == 3 else gvhmr_results["K_fullimg"]
        
        keypoints_3d = smpl_to_coco17(smpl_global)
        joints_2d = project_3d_to_2d(smpl_incam, K)
        keypoints_2d = smpl_to_coco17(joints_2d)
        
        if scale != 1.0:
            keypoints_2d_scaled = [(int(pt[0]*scale), int(pt[1]*scale)) if pt is not None else None for pt in keypoints_2d]
            keypoints_2d = keypoints_2d_scaled

        # ONLY DRAW THE SKELETON IF THE TOGGLE IS CHECKED:
        if st.session_state.get("show_skeleton", True):
            annotated = draw_skeleton(annotated, keypoints_2d)

    # ────────────── Ensure R2PScorer exists ──────────────
    if "r2p_scorer" not in st.session_state:
        from core.counters import R2PScorer
        st.session_state["r2p_scorer"] = R2PScorer()

    # ────────────── Render frame with posture scoring ──────────────
    processed_image = process_frame(
        annotated,
        keypoints_2d,
        keypoints_3d,
        selected_rules,
        rules_all,
        st.session_state["floor_y"],
        cmj_counter,
        sls_counter,
        balance_tracker,
        st.session_state["r2p_scorer"],
    )
    
    frame_perf_end = time.perf_counter()
    latency_ms = (frame_perf_end - frame_perf_start) * 1000
    
    # Annotate processing time directly onto the image
    cv2.putText(processed_image, f"Local Render: {latency_ms:.1f}ms", (10, processed_image.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(processed_image, f"Local Render: {latency_ms:.1f}ms", (10, processed_image.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    frame_placeholder.image(
        processed_image,
        width="stretch",
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
    st.error("GVHMR requires whole-video batch processing. Webcam mode is not available with GVHMR at this time.")
    st.stop()

elif source_option == "Webcam" and not run_webcam:
    st.info("Enable **Start Webcam** in the sidebar to begin.")


