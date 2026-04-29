"""
ui/video_upload.py
------------------
Handles multiple MP4 uploads with a processing queue.
Videos are stored under uploads/<sha256_hex>.mp4 (content-based dedup).
Queue is maintained in session_state["upload_queue"] as ordered list of dicts.
"""
import cv2
import hashlib
import os
import streamlit as st

UPLOAD_DIR = "uploads"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def handle_video_upload() -> None:
    """
    Renders a multi-file uploader and populates the processing queue.
    Each unique video is enqueued exactly once.

    Queue entry schema:
    {
        "hash":        str,        # sha256 of raw bytes
        "cap_path":    str,        # path on disk
        "name":        str,        # clean name (no extension)
        "display":     str,        # original filename
        "total_frames":int,
        "fps":         float,
        "status":      str,        # "queued" | "processing" | "done" | "error"
        "output_path": str | None, # set when done
    }
    """
    uploaded_files = st.sidebar.file_uploader(
        "Upload MP4 Videos",
        type=["mp4"],
        accept_multiple_files=True,
        key="multi_uploader",
    )

    if not uploaded_files:
        return

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Ensure queue exists
    if "upload_queue" not in st.session_state:
        st.session_state["upload_queue"] = []

    # Track hashes already in queue to avoid duplicates
    existing_hashes = {entry["hash"] for entry in st.session_state["upload_queue"]}

    for uploaded in uploaded_files:
        raw = uploaded.read()
        file_hash = _sha256(raw)

        if file_hash in existing_hashes:
            continue  # already queued, skip

        dest = os.path.join(UPLOAD_DIR, f"{file_hash}.mp4")
        if not os.path.exists(dest):
            with open(dest, "wb") as f:
                f.write(raw)

        cap = cv2.VideoCapture(dest)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        clean_name = os.path.splitext(uploaded.name)[0]

        entry = {
            "hash":         file_hash,
            "cap_path":     dest,
            "name":         clean_name,
            "display":      uploaded.name,
            "total_frames": total_frames,
            "fps":          fps,
            "status":       "queued",
            "output_path":  None,
        }

        st.session_state["upload_queue"].append(entry)
        existing_hashes.add(file_hash)