# main.py
import os
import cv2
import csv
import streamlit as st

from config import DEFAULTS
from ui.video_upload import handle_video_upload
from ui.frame_display import process_frame
from core.counters import (
    R2PScorer, SLSDetector, SwayTracker,
    extract_features, CMJTracker, calculate_fppa,
)
from body_tracking import process_video_gvhmr, project_3d_to_2d, smpl_to_coco17, get_yolo26_keypoints
from remote_ssh_pipeline import process_video_on_remote
from posture_analysis import load_rules

# ────────────── Page Config ──────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="R2P Ready-to-Play Guard")
st.title("R2P Ready-to-Play Guard")

# ────────────── Session Defaults ─────────────────────────────────────────────
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ────────────── Persistent Queue & Results ───────────────────────────────────
if "upload_queue" not in st.session_state:
    st.session_state["upload_queue"] = []

if "completed_videos" not in st.session_state:
    st.session_state["completed_videos"] = []     # list of output_path strings

if "selected_video" not in st.session_state:
    st.session_state["selected_video"] = None     # path currently shown in player

# ────────────── Load Rules ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RULES_PATH = os.path.join(BASE_DIR, "assets", "rules.txt")
rules_all = load_rules(RULES_PATH)
default_rules = list(rules_all.keys())[:3]
selected_rules = default_rules   # fixed – no sidebar rule picker in batch mode

# ────────────── Sidebar: upload only ─────────────────────────────────────────
st.sidebar.title("📂 Upload Videos")
st.sidebar.caption(
    "Upload one or more MP4 files. "
    "They will be processed automatically in queue order."
)
handle_video_upload()   # populates st.session_state["upload_queue"]


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def draw_skeleton(frame, keypoints_2d):
    if not keypoints_2d:
        return frame
    edges = [
        (0,1),(0,2),(1,3),(2,4),
        (5,7),(7,9),(6,8),(8,10),
        (5,6),(5,11),(6,12),(11,12),
        (11,13),(13,15),(12,14),(14,16),
    ]
    for pt in keypoints_2d:
        if pt is not None:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
    for p1, p2 in edges:
        kp1, kp2 = keypoints_2d[p1], keypoints_2d[p2]
        if kp1 is not None and kp2 is not None:
            cv2.line(
                frame,
                (int(kp1[0]), int(kp1[1])),
                (int(kp2[0]), int(kp2[1])),
                (0, 0, 255), 2,
            )
    return frame


def _make_trackers():
    """Return a fresh set of per-video trackers."""
    return {
        "sway":    SwayTracker(fps=60),
        "sls":     SLSDetector(),
        "cmj":     CMJTracker(fps=60),
        "r2p":     R2PScorer(),
        "floor_y": 500,
    }

def _process_one_video(entry: dict, progress_text, frame_placeholder) -> str | None:
    cap_path    = entry["cap_path"]
    video_name  = entry["name"]
    total_frames = entry["total_frames"]
    fps          = entry["fps"]

    output_dir = os.path.join("outputs", video_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "annotated.mp4")

    # ── GVHMR ─────────────────────────────────────────────
    progress_text.info(f"⏳ **{video_name}** — running 3D pose estimation (SSH GPU)…")
    f_mm = 24

    try:
        gvhmr_results = process_video_on_remote(cap_path, f_mm=f_mm)
    except Exception as exc:
        progress_text.error(f"❌ GVHMR failed for **{video_name}**: {exc}")
        return None

    if gvhmr_results is None:
        progress_text.error(f"❌ GVHMR returned no data for **{video_name}**.")
        return None

    # ── Video open ────────────────────────────────────────
    cap = cv2.VideoCapture(cap_path)

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── Writer (robust codec fallback) ────────────────────
    writer = None
    for codec in ("avc1", "H264", "XVID", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))
        if writer.isOpened():
            break
        writer.release()
        writer = None

    if writer is None:
        progress_text.error(f"❌ Could not open VideoWriter for **{video_name}**.")
        cap.release()
        return None

    # ── Trackers ──────────────────────────────────────────
    trackers = _make_trackers()
    cmj_counter   = trackers["cmj"]
    sls_counter   = trackers["sls"]
    sway_tracker  = trackers["sway"]
    r2p_scorer    = trackers["r2p"]
    floor_y       = trackers["floor_y"]

    num_gvhmr_frames = len(gvhmr_results["joints_3d_global"])

    progress_bar = st.progress(0, text=f"Processing {video_name}…")

    # ✅ FIX: sequential read instead of cap.set
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # ← guarantees last valid frame is INCLUDED

        # ── Resize for display ─────────────────────────
        if frame.shape[1] > 800:
            scale = 800 / frame.shape[1]
            disp  = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        else:
            scale = 1.0
            disp  = frame.copy()

        # ── Keypoints ────────────────────────────────
        idx_b = min(idx, num_gvhmr_frames - 1)

        smpl_global = gvhmr_results["joints_3d_global"][idx_b]
        smpl_incam  = gvhmr_results["joints_3d_incam"][idx_b]

        K = (
            gvhmr_results["K_fullimg"][idx_b]
            if gvhmr_results["K_fullimg"].ndim == 3
            else gvhmr_results["K_fullimg"]
        )

        keypoints_3d = smpl_to_coco17(smpl_global)
        joints_2d    = project_3d_to_2d(smpl_incam, K)
        keypoints_2d = smpl_to_coco17(joints_2d)

        if scale != 1.0:
            keypoints_2d = [
                (int(pt[0] * scale), int(pt[1] * scale)) if pt is not None else None
                for pt in keypoints_2d
            ]

        disp = draw_skeleton(disp, keypoints_2d)

        # ── process_frame ─────────────────────────────
        st.session_state["frame_index"]  = idx
        st.session_state["sway_tracker"] = sway_tracker

        if f"bfy_{video_name}" not in st.session_state:
            try:
                bfy = float((keypoints_3d[15][1] + keypoints_3d[16][1]) / 2)
            except:
                bfy = 0.0
            st.session_state[f"bfy_{video_name}"] = bfy

        st.session_state["baseline_feet_y"] = st.session_state[f"bfy_{video_name}"]

        annotated_rgb = process_frame(
            disp,
            keypoints_2d,
            keypoints_3d,
            selected_rules,
            rules_all,
            floor_y,
            cmj_counter,
            sls_counter,
            sway_tracker,
            r2p_scorer,
        )

        # Preview
        if idx % 10 == 0:
            frame_placeholder.image(
                annotated_rgb,
                caption=f"{video_name} — frame {idx}",
                use_container_width=True,
            )

        # ── WRITE FRAME (guaranteed every valid frame) ──
        frame_bgr = cv2.cvtColor(
            cv2.resize(annotated_rgb, (orig_w, orig_h)),
            cv2.COLOR_RGB2BGR,
        )
        writer.write(frame_bgr)

        # Progress
        if total_frames > 0:
            progress_bar.progress(
                min((idx + 1) / total_frames, 1.0),
                text=f"Processing {video_name} — {idx+1}/{total_frames}",
            )

        idx += 1

    cap.release()
    writer.release()
    progress_bar.empty()

    # ── Save metrics ─────────────────────────────
    metrics_path = os.path.join("outputs", "metrics.csv")
    file_exists  = os.path.isfile(metrics_path)

    with open(metrics_path, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["video", "cv", "one_minus_cv"])
        w.writerow([
            video_name,
            sway_tracker.get_cv(),
            sway_tracker.get_one_minus_cv(),
        ])

    return output_path



# def _process_one_video(entry: dict, progress_text, frame_placeholder) -> str | None:
#     """
#     Process a single queue entry end-to-end.
#     Writes annotated frames to outputs/<name>/annotated.mp4.
#     Returns output path on success, None on failure.
#     """
#     cap_path    = entry["cap_path"]
#     video_name  = entry["name"]
#     total_frames = entry["total_frames"]
#     fps          = entry["fps"]

#     output_dir = os.path.join("outputs", video_name)
#     os.makedirs(output_dir, exist_ok=True)
#     output_path = os.path.join(output_dir, "annotated.mp4")

#     # ── 3-D tracking via remote GVHMR ────────────────────────────────────────
#     progress_text.info(f"⏳ **{video_name}** — running 3D pose estimation (SSH GPU)…")
#     f_mm = 24   # default focal length; adjust if needed

#     try:
#         gvhmr_results = process_video_on_remote(cap_path, f_mm=f_mm)
#     except Exception as exc:
#         progress_text.error(f"❌ GVHMR failed for **{video_name}**: {exc}")
#         return None

#     if gvhmr_results is None:
#         progress_text.error(f"❌ GVHMR returned no data for **{video_name}**.")
#         return None

#     # ── Open video & writer ───────────────────────────────────────────────────
#     cap = cv2.VideoCapture(cap_path)
#     orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Try codecs in order; mp4v always works on Windows (no OpenH264 needed)
#     writer = None
#     for codec in ("avc1", "H264", "XVID", "mp4v"):
#         fourcc = cv2.VideoWriter_fourcc(*codec)
#         writer = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))
#         if writer.isOpened():
#             break
#         writer.release()
#         writer = None
#     if writer is None:
#         progress_text.error(f"❌ Could not open VideoWriter for **{video_name}**.")
#         cap.release()
#         return None

#     trackers = _make_trackers()
#     cmj_counter   = trackers["cmj"]
#     sls_counter   = trackers["sls"]
#     sway_tracker  = trackers["sway"]
#     r2p_scorer    = trackers["r2p"]
#     floor_y       = trackers["floor_y"]

#     num_gvhmr_frames = len(gvhmr_results["joints_3d_global"])

#     progress_bar = st.progress(0, text=f"Processing {video_name}…")

#     for idx in range(total_frames):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # ── Display-scale frame ───────────────────────────────────────────
#         if frame.shape[1] > 800:
#             scale = 800 / frame.shape[1]
#             disp  = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
#         else:
#             scale = 1.0
#             disp  = frame.copy()

#         # ── Keypoints ────────────────────────────────────────────────────
#         idx_b = min(idx, num_gvhmr_frames - 1)
#         smpl_global = gvhmr_results["joints_3d_global"][idx_b]
#         smpl_incam  = gvhmr_results["joints_3d_incam"][idx_b]

#         K = (
#             gvhmr_results["K_fullimg"][idx_b]
#             if gvhmr_results["K_fullimg"].ndim == 3
#             else gvhmr_results["K_fullimg"]
#         )

#         keypoints_3d = smpl_to_coco17(smpl_global)
#         joints_2d    = project_3d_to_2d(smpl_incam, K)
#         keypoints_2d = smpl_to_coco17(joints_2d)

#         if scale != 1.0:
#             keypoints_2d = [
#                 (int(pt[0] * scale), int(pt[1] * scale)) if pt is not None else None
#                 for pt in keypoints_2d
#             ]

#         disp = draw_skeleton(disp, keypoints_2d)

#         # ── process_frame (annotations + metrics) ────────────────────────
#         # Temporarily shim session_state so process_frame works standalone
#         st.session_state["frame_index"]   = idx
#         st.session_state["sway_tracker"]  = sway_tracker
#         if f"bfy_{video_name}" not in st.session_state:
#             try:
#                 bfy = float((keypoints_3d[15][1] + keypoints_3d[16][1]) / 2)
#             except (TypeError, IndexError):
#                 bfy = 0.0
#             st.session_state[f"bfy_{video_name}"] = bfy
#         st.session_state["baseline_feet_y"] = st.session_state[f"bfy_{video_name}"]

#         annotated_rgb = process_frame(
#             disp,
#             keypoints_2d,
#             keypoints_3d,
#             selected_rules,
#             rules_all,
#             floor_y,
#             cmj_counter,
#             sls_counter,
#             sway_tracker,
#             r2p_scorer,
#         )

#         # Preview every 10th frame while processing
#         if idx % 10 == 0:
#             frame_placeholder.image(annotated_rgb, caption=f"{video_name} — frame {idx}/{total_frames}", use_container_width=True)

#         # Write to output at original resolution
#         frame_bgr = cv2.cvtColor(
#             cv2.resize(annotated_rgb, (orig_w, orig_h)),
#             cv2.COLOR_RGB2BGR,
#         )
#         writer.write(frame_bgr)

#         # Update progress bar
#         progress_bar.progress(
#             (idx + 1) / total_frames,
#             text=f"Processing {video_name} — {idx+1}/{total_frames} frames",
#         )

#     cap.release()
#     writer.release()
#     progress_bar.empty()

#     # ── Save sway metrics ─────────────────────────────────────────────────────
#     metrics_path = os.path.join("outputs", "metrics.csv")
#     file_exists  = os.path.isfile(metrics_path)
#     with open(metrics_path, "a", newline="") as f:
#         w = csv.writer(f)
#         if not file_exists:
#             w.writerow(["video", "cv", "one_minus_cv"])
#         w.writerow([
#             video_name,
#             sway_tracker.get_cv(),
#             sway_tracker.get_one_minus_cv(),
#         ])

#     return output_path


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

left_col, right_col = st.columns([1, 2])

# ── Left: queue status + video gallery ───────────────────────────────────────
with left_col:
    st.subheader("📋 Queue")

    queue = st.session_state["upload_queue"]

    if not queue:
        st.info("Upload videos in the sidebar to begin.")
    else:
        for i, entry in enumerate(queue):
            status_icon = {
                "queued":     "🟡",
                "processing": "🔵",
                "done":       "✅",
                "error":      "❌",
            }.get(entry["status"], "❓")

            st.markdown(
                f"{status_icon} **{entry['display']}** "
                f"— {entry['total_frames']} frames @ {entry['fps']:.0f} fps"
            )

    # ── Completed videos gallery ──────────────────────────────────────────────
    completed = st.session_state["completed_videos"]
    if completed:
        st.markdown("---")
        st.subheader("🎬 Processed Videos")
        st.caption("Click a video to load it in the player →")

        for path in completed:
            label = os.path.basename(os.path.dirname(path))   # folder name = video name
            is_selected = st.session_state["selected_video"] == path
            btn_type = "primary" if is_selected else "secondary"

            if st.button(
                f"▶ {label}",
                key=f"gallery_{path}",
                type=btn_type,
                use_container_width=True,
            ):
                st.session_state["selected_video"] = path
                st.rerun()


# ── Right: processing feedback + playback ────────────────────────────────────
with right_col:

    # ── Processing area ───────────────────────────────────────────────────────
    progress_text    = st.empty()
    frame_preview    = st.empty()

    # Find the next queued video and process it
    next_entry = next(
        (e for e in st.session_state["upload_queue"] if e["status"] == "queued"),
        None,
    )

    if next_entry is not None:
        next_entry["status"] = "processing"

        output_path = _process_one_video(next_entry, progress_text, frame_preview)

        if output_path:
            next_entry["status"]      = "done"
            next_entry["output_path"] = output_path
            st.session_state["completed_videos"].append(output_path)
            progress_text.success(f"✅ **{next_entry['name']}** — processing complete.")
        else:
            next_entry["status"] = "error"

        # Rerun so the next queued video starts immediately
        st.rerun()

    # ── Playback ──────────────────────────────────────────────────────────────
    selected_path = st.session_state.get("selected_video")

    if selected_path and os.path.exists(selected_path):
        st.subheader(f"▶ {os.path.basename(os.path.dirname(selected_path))}")
        with open(selected_path, "rb") as f:
            st.video(f.read())

    elif not next_entry and not st.session_state["completed_videos"]:
        st.info("Upload videos in the sidebar. Processing will start automatically.")