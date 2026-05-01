import cv2
import os
import numpy as np
import threading

from core.counters import (
    R2PScorer,
    RepCounter,
    SwayTracker,
    SLSDetector,
    extract_features,
    CMJTracker
)


_FALLBACK_STATE = {
    "cv_logged": False,
    "cv_saved": False,
    "video_results": [],
    "baseline_feet_y": None,
    "frame_index": 0,
    "cv_debug_frame_index": 0,
}


def _has_streamlit_context():
    # Backend/worker threads must never try to resolve Streamlit runtime context.
    if threading.current_thread() is not threading.main_thread():
        return False

    try:
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
        except ImportError:
            try:
                from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
            except ImportError:
                get_script_run_ctx = None

        if get_script_run_ctx is None:
            return False

        return get_script_run_ctx() is not None
    except Exception:
        return False


def _runtime_state():
    if _has_streamlit_context():
        try:
            import streamlit as st
            return st.session_state
        except Exception:
            return _FALLBACK_STATE
    return _FALLBACK_STATE


def set_runtime_state_values(**kwargs):
    # Allows non-Streamlit callers (engine/api) to configure debug/runtime keys.
    _FALLBACK_STATE.update(kwargs)


def process_frame(
    frame,
    keypoints,
    keypoints_3d,
    selected_rules,
    rules_all,
    floor_y,
    cmj_counter,
    sls_counter,
    sway_tracker,
    r2p_scorer
):

    display_frame = frame.copy()

    state = _runtime_state()

    if "cv_logged" not in state:
        state["cv_logged"] = False


    if keypoints is None or keypoints_3d is None:
        if not state["cv_saved"]:
            sway_tracker = state.get("sway_tracker", sway_tracker)

            result = {
                "cv": sway_tracker.get_cv(),
                "one_minus_cv": sway_tracker.get_one_minus_cv()
            }

            state["video_results"].append(result)
            state["cv_saved"] = True


        cv2.line(display_frame, (0, floor_y),
                 (display_frame.shape[1], floor_y),
                 (255, 255, 0), 2)

        return cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

    # ==========================================================
    # BUILD TRACKER OUTPUT DICTIONARY
    # ==========================================================
    tracker_output = {
        "joints_3d_global": keypoints_3d
    }

    # ==========================================================
    # BASELINE FEET HEIGHT (ONLY SET ONCE)
    # ==========================================================
    if state.get("baseline_feet_y") is None:
        state["baseline_feet_y"] = (
            keypoints_3d[15][1] + keypoints_3d[16][1]
        ) / 2

    baseline_feet_y = state["baseline_feet_y"]

    # ==========================================================
    # FRAME INDEX
    # ==========================================================
    if "frame_index" not in state:
        state["frame_index"] = 0

    if "cv_debug_frame_index" not in state:
        state["cv_debug_frame_index"] = 0

    frame_index = state["frame_index"]
    cv_debug_frame_index = state["cv_debug_frame_index"]

    # ==========================================================
    # EXTRACT FEATURES
    # ==========================================================
    features = extract_features(
        tracker_output,
        baseline_feet_y
    )

    # ==========================================================
    # UPDATE TRACKERS
    # ==========================================================
    sway_tracker = state.get("sway_tracker", sway_tracker)
    sway_tracker.update(
        features["mid_hip"],
        features["mid_shoulder"]
    )

    sway_velocity = sway_tracker.get_sway_velocity()
    sway_cv = sway_tracker.get_cv()

    # Optional diagnostics to compare Streamlit and engine CV pipelines.
    debug_enabled = bool(state.get("cv_debug_enabled", False)) or os.getenv("CV_DEBUG", "0") == "1"
    if debug_enabled:
        pipeline = state.get("cv_debug_pipeline", "unknown")
        max_frames = int(os.getenv("CV_DEBUG_MAX_FRAMES", "20"))
        printed = int(state.get("cv_debug_count", 0))
        if printed < max_frames:
            print(
                f"[CVDEBUG][{pipeline}] frame={cv_debug_frame_index} "
                f"mid_hip={features['mid_hip']} "
                f"mid_shoulder={features['mid_shoulder']} "
                f"sway_vel={sway_velocity:.6f} cv={sway_cv:.6f}"
            )
            state["cv_debug_count"] = printed + 1
    state["cv_debug_frame_index"] = cv_debug_frame_index + 1

    hip = keypoints_3d[11]
    knee = keypoints_3d[13]
    ankle = keypoints_3d[15]
    sls_counter.update(hip,knee,ankle)

    left_ankle = keypoints_3d[15]
    right_ankle = keypoints_3d[16]
    ankle_y = (left_ankle[1] + right_ankle[1]) / 2
    cmj_counter.update(ankle_y)



    # SLS + CMJ metrics (continuous)
    fppa = sls_counter.get_fppa()

    rsi = cmj_counter.get_rsi()
    # flight_time = cmj_counter.get_refined_flight_time()
    # contraction_time = cmj_counter.get_contraction_time()

    # ==========================================================
    # DISPLAY VARIABLES
    # ==========================================================
    y = 30
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ==========================================================
    # CMJ METRICS
    # ==========================================================
    cv2.putText(display_frame, f"RSI: {rsi:.2f}", (10, y),
                font, 0.7, (255, 255, 255), 2)
    y += 25

    # cv2.putText(display_frame, f"Flight: {flight_time*1000:.0f} ms", (10, y),
    #             font, 0.7, (255, 255, 255), 2)
    # y += 25

    # cv2.putText(display_frame, f"CT: {contraction_time*1000:.0f} ms", (10, y),
    #             font, 0.7, (255, 255, 255), 2)
    # y += 30

    # ==========================================================
    # SLS METRIC
    # ==========================================================
    cv2.putText(display_frame, f"FPPA: {fppa:.1f} deg", (10, y),
                font, 0.7, (255, 255, 255), 2)
    y += 30

    # ==========================================================
    # SWAY METRICS
    # ==========================================================
    cv2.putText(display_frame, f"Sway Vel: {sway_velocity:.3f}", (10, y),
                font, 0.7, (255, 255, 255), 2)
    y += 25

    cv2.putText(display_frame, f"CV: {sway_cv:.1f}%", (10, y),
                font, 0.7, (255, 255, 255), 2)
    y += 30

    # # ==========================================================
    # # COACH OUTPUT (UNCHANGED)
    # # ==========================================================
    # total_score, traffic_light = r2p_scorer.compute(
    #     cv=sway_cv,
    #     fppa=fppa,
    #     delta_rsi=rsi
    # )

    # if traffic_light == "GREEN":
    #     coach_msg = "Great Form"
    #     color = (0, 255, 0)

    # elif traffic_light == "YELLOW":
    #     coach_msg = "Adjust Form"
    #     color = (0, 255, 255)

    # else:
    #     coach_msg = "Fix Form"
    #     color = (0, 0, 255)

    # cv2.putText(display_frame, coach_msg, (10, y),
    #             font, 0.9, color, 2)

    # y += 35

    # # ==========================================================
    # # FLOOR LINE
    # # ==========================================================
    # cv2.line(display_frame,
    #         (0, floor_y),
    #         (display_frame.shape[1], floor_y),
    #         (255, 255, 0), 2)

    # ==========================================================
    # FRAME INDEX
    # ==========================================================
    state["frame_index"] = frame_index + 1

    return cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)



    # # ==========================================================
    # # UPDATE TRACKERS
    # # ==========================================================
    # sway_tracker = st.session_state["sway_tracker"]
    # sway_tracker.update(features["mid_hip"])

    # sway_velocity = sway_tracker.get_sway_velocity()
    # sway_cv = sway_tracker.get_cv()

    # fppa = sls_counter.get_fppa()

    # rsi = cmj_counter.get_rsi()
    # flight_time = cmj_counter.get_refined_flight_time()
    # contraction_time = cmj_counter.get_contraction_time()

    # # delta_rsi = cmj_counter.update(
    # #     features["jump_feet"],
    # #     frame_index
    # # )

    # sls_reps = sls_counter.update(features)

    # # ==========================================================
    # # DISPLAY VARIABLES
    # # ==========================================================
    # y = 30

    # # ==========================================================
    # # CMJ SCORE
    # # ==========================================================
    # if delta_rsi is not None:

    #     rep_score_pct = max(0, 100 - delta_rsi)

    #     color = (
    #         (0, 255, 0) if rep_score_pct > 80 else
    #         (0, 255, 255) if rep_score_pct > 60 else
    #         (0, 0, 255)
    #     )

    #     cv2.putText(
    #         display_frame,
    #         f"CMJ Score: {rep_score_pct:.0f}%",
    #         (10, y),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         0.8,
    #         color,
    #         2
    #     )

    #     y += 30

    # # ==========================================================
    # # SLS REPS
    # # ==========================================================
    # cv2.putText(
    #     display_frame,
    #     f"SLS Reps: {sls_reps}",
    #     (10, y),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.8,
    #     (255, 255, 0),
    #     2
    # )

    # y += 30

    # # ==========================================================
    # # SWAY METRICS
    # # ==========================================================
    # cv2.putText(
    #     display_frame,
    #     f"Sway Vel: {sway_velocity:.2f}",
    #     (10, y),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.7,
    #     (255, 255, 255),
    #     2
    # )

    # y += 25

    # cv2.putText(
    #     display_frame,
    #     f"CV: {sway_cv:.1f}%",
    #     (10, y),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.7,
    #     (255, 255, 255),
    #     2
    # )

    # y += 25

    # # ==========================================================
    # # GLOBAL SCORING
    # # ==========================================================
    # total_score, traffic_light = r2p_scorer.compute(
    #     cv=sway_cv,
    #     fppa=features["sls_fppa"],
    #     delta_rsi=delta_rsi
    # )

    # if traffic_light == "GREEN":
    #     coach_msg = "Great Form"
    #     color = (0, 255, 0)

    # elif traffic_light == "YELLOW":
    #     coach_msg = "Adjust Form"
    #     color = (0, 255, 255)

    # else:
    #     coach_msg = "Fix Form"
    #     color = (0, 0, 255)

    # cv2.putText(
    #     display_frame,
    #     coach_msg,
    #     (10, y),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.9,
    #     color,
    #     2
    # )

    # y += 35

    # # ==========================================================
    # # FLOOR LINE
    # # ==========================================================
    # cv2.line(
    #     display_frame,
    #     (0, floor_y),
    #     (display_frame.shape[1], floor_y),
    #     (255, 255, 0),
    #     2
    # )

    # # ==========================================================
    # # INCREMENT FRAME
    # # ==========================================================
    # st.session_state["frame_index"] += 1

    # return cv2.cvtColor(
    #     display_frame,
    #     cv2.COLOR_BGR2RGB
    # )




