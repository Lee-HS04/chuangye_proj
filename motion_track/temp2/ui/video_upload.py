"""
ui/video_upload.py
------------------
Handles MP4 uploads with content-based deduplication.

Videos are stored under  uploads/<sha256_hex>.mp4
so the same file is never written twice across reruns.
"""

import hashlib
import os
import streamlit as st

UPLOAD_DIR = "uploads"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def handle_video_upload() -> None:
    """
    Render the file-uploader widget and persist the video to disk exactly once.

    Sets / updates session_state keys:
        cap_path        – absolute path to the saved .mp4
        frame_index     – reset to 0 on a *new* file
        playing         – reset to False on a new file
        total_frames    – frame count of the video
        video_fps       – frame-rate of the video
    """
    import cv2

    uploaded = st.sidebar.file_uploader("Upload MP4 Video", type=["mp4"])
    if uploaded is None:
        return

    raw = uploaded.read()
    file_hash = _sha256(raw)

    # Only write to disk when the file is new
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(UPLOAD_DIR, f"{file_hash}.mp4")

    if not os.path.exists(dest):
        with open(dest, "wb") as fh:
            fh.write(raw)

    # Only reset playback state when a *different* video is loaded
    if st.session_state.get("_loaded_hash") != file_hash:
        st.session_state["_loaded_hash"] = file_hash
        st.session_state["cap_path"] = dest
        st.session_state["frame_index"] = 0
        st.session_state["playing"] = False

        cap = cv2.VideoCapture(dest)
        st.session_state["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.session_state["video_fps"] = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.release()