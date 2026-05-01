"""
ui/video_upload.py
------------------
Handles MP4 uploads with content-based deduplication.

Videos are stored under  uploads/<sha256_hex>.mp4
so the same file is never written twice across reruns.
"""
import cv2
import hashlib
import os
import streamlit as st

from core.counters import R2PScorer, SLSDetector, SwayTracker, CMJTracker

UPLOAD_DIR = "uploads"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def handle_video_upload() -> None:
    uploaded = st.sidebar.file_uploader("Upload MP4 Video", type=["mp4"])

    # IMPORTANT: must stop here safely
    if uploaded is None:
        return

    raw = uploaded.read()
    file_hash = _sha256(raw)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(UPLOAD_DIR, f"{file_hash}.mp4")

    if not os.path.exists(dest):
        with open(dest, "wb") as f:
            f.write(raw)

    # reset only when new file
    if st.session_state.get("_loaded_hash") != file_hash:
        st.session_state["_loaded_hash"] = file_hash
        st.session_state["cap_path"] = dest
        st.session_state["frame_index"] = 0
        st.session_state["playing"] = False

        # Reset per-video analysis state so Streamlit matches engine's fresh-run behavior.
        st.session_state["video_results"] = []
        st.session_state["cv_saved"] = False
        st.session_state["metrics_saved"] = False
        st.session_state["baseline_feet_y"] = None
        st.session_state["cv_logged"] = False
        st.session_state["cv_debug_count"] = 0
        st.session_state["cv_debug_frame_index"] = 0

        # Recreate trackers/counters for deterministic CV parity across runs.
        st.session_state["sway_tracker"] = SwayTracker(fps=60)
        st.session_state["balance_tracker"] = st.session_state["sway_tracker"]
        st.session_state["cmj_counter"] = CMJTracker(fps=60)
        st.session_state["sls_counter"] = SLSDetector()
        st.session_state["r2p_scorer"] = R2PScorer()

        # Refresh GVHMR cache key to force re-compute for this specific video path.
        st.session_state["gvhmr_path"] = None

        cap = cv2.VideoCapture(dest)
        st.session_state["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.session_state["video_fps"] = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.release()

    # ✅ SAFE NAME HANDLING (ADD THIS HERE)
    file_name = uploaded.name
    clean_name = os.path.splitext(file_name)[0]

    st.session_state["video_name"] = clean_name
    st.session_state["video_name_display"] = file_name
    st.session_state["video_hash"] = file_hash

# def handle_video_upload() -> None:
#     """
#     Render the file-uploader widget and persist the video to disk exactly once.

#     Sets / updates session_state keys:
#         cap_path        – absolute path to the saved .mp4
#         frame_index     – reset to 0 on a *new* file
#         playing         – reset to False on a new file
#         total_frames    – frame count of the video
#         video_fps       – frame-rate of the video
#     """
    

#     uploaded = st.sidebar.file_uploader("Upload MP4 Video", type=["mp4"])
#     if uploaded is None:
#         return
    
#     file_name = uploaded.name
#     file_hash = _sha256(raw)

#     # clean display name (no extension)
#     clean_name = os.path.splitext(file_name)[0]

#     # store BOTH safe + display versions
#     st.session_state["video_name"] = clean_name
#     st.session_state["video_name_display"] = file_name
#     st.session_state["video_hash"] = file_hash

#     raw = uploaded.read()
#     file_hash = _sha256(raw)

#     # Only write to disk when the file is new
#     os.makedirs(UPLOAD_DIR, exist_ok=True)
#     dest = os.path.join(UPLOAD_DIR, f"{file_hash}.mp4")

#     if not os.path.exists(dest):
#         with open(dest, "wb") as fh:
#             fh.write(raw)

#     # Only reset playback state when a *different* video is loaded
#     if st.session_state.get("_loaded_hash") != file_hash:
#         st.session_state["_loaded_hash"] = file_hash
#         st.session_state["cap_path"] = dest
#         st.session_state["frame_index"] = 0
#         st.session_state["playing"] = False

#         cap = cv2.VideoCapture(dest)
#         st.session_state["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         st.session_state["video_fps"] = cap.get(cv2.CAP_PROP_FPS) or 30
#         cap.release()