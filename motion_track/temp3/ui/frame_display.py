# # ui/frame_display.py
# import cv2
# import numpy as np
# import streamlit as st
# from collections import deque
# from core.posture import extract_features

# # ────────────── Process Frame ──────────────
# def process_frame(frame, keypoints, keypoints_3d, selected_rules, rules_all, floor_y,
#                   cmj_counter, sls_counter, sway_tracker):

#     display_frame = frame.copy()

#     # Early exit if no keypoints
#     if keypoints is None or keypoints_3d is None:
#         cv2.line(display_frame, (0, floor_y), (display_frame.shape[1], floor_y), (255, 255, 0), 2)
#         return cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

#     # Extract features from 3D keypoints
#     features = extract_features(keypoints_3d)

#     # ────────────── Helper ──────────────
#     def valid_pt(arr):
#         if arr is None: return False
#         if hasattr(arr, 'size'): return arr.size > 0
#         return len(arr) > 0

#     # Jump feet height
#     if len(keypoints) > 16 and valid_pt(keypoints[15]) and valid_pt(keypoints[16]):
#         features["jump_feet"] = floor_y - min(keypoints[15][1], keypoints[16][1])

#     # Mid-hip (balance)
#     mid_hip = None
#     if len(keypoints) > 12 and valid_pt(keypoints[11]) and valid_pt(keypoints[12]):
#         left_hip, right_hip = keypoints[11], keypoints[12]
#         mid_hip = ((left_hip[0]+right_hip[0])/2, (left_hip[1]+right_hip[1])/2)

#         # Smoothing
#         alpha = 0.7
#         if "prev_mid_hip" in st.session_state:
#             prev = st.session_state["prev_mid_hip"]
#             mid_hip = (alpha*mid_hip[0]+(1-alpha)*prev[0],
#                        alpha*mid_hip[1]+(1-alpha)*prev[1])
#         st.session_state["prev_mid_hip"] = mid_hip

#     if mid_hip is None and "prev_mid_hip" in st.session_state:
#         mid_hip = st.session_state["prev_mid_hip"]

#     features["mid_hip"] = mid_hip

#     # Update sway tracker
#     if mid_hip is not None:
#         sway_tracker.update(mid_hip)

#     features["sway_velocity"] = sway_tracker.get_sway_velocity()
#     features["sway_cv"] = sway_tracker.get_cv()

#     # ────────────── Live Coaching Feedback ──────────────
#     y = 30
#     quality_score = 100
#     for rule_name in selected_rules:
#         val = features.get(rule_name)
#         rule = rules_all.get(rule_name)
#         if val is None or rule is None:
#             continue

#         min_v, max_v = rule
#         if val < min_v:
#             msg = f"{rule_name}: Too Low"
#             color = (0, 0, 255)
#             quality_score -= 10
#         elif val > max_v:
#             msg = f"{rule_name}: Too High"
#             color = (0, 165, 255)
#             quality_score -= 10
#         else:
#             msg = f"{rule_name}: Good"
#             color = (0, 255, 0)

#         cv2.putText(display_frame, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
#         y += 25

#     # ────────────── Rep Counting ──────────────
#     rep_score = cmj_counter.update(features)
#     cv2.putText(display_frame, f"Reps: {cmj_counter.reps}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
#     y += 30

#     if rep_score is not None:
#         msg = f"Last Rep: {rep_score}%"
#         if rep_score > 80:
#             color = (0, 255, 0)
#         elif rep_score > 60:
#             color = (0, 255, 255)
#         else:
#             color = (0, 0, 255)
#         cv2.putText(display_frame, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
#         y += 30

#     # ────────────── Fatigue Display ──────────────
#     if "fatigue_hist" not in sway_tracker.__dict__:
#         sway_tracker.fatigue_hist = deque(maxlen=30)

#     sway_val = features.get("sway_velocity")
#     if sway_val is not None:
#         sway_tracker.fatigue_hist.append(sway_val)
#         avg = np.mean(sway_tracker.fatigue_hist)

#         if avg < 5:
#             status, color = "Fresh", (0, 255, 0)
#         elif avg < 10:
#             status, color = "Slight Fatigue", (0, 255, 255)
#         else:
#             status, color = "Fatigued", (0, 0, 255)

#         cv2.putText(display_frame, f"Fatigue: {status}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
#         y += 25

#     # ────────────── Display Sway Metrics ──────────────
#     y_feedback = 150
#     sway_velocity = features.get("sway_velocity")
#     cv = features.get("sway_cv")

#     if sway_velocity is not None:
#         cv2.putText(display_frame, f"Sway Vel: {sway_velocity:.2f}", (10, y_feedback),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#         y_feedback += 25

#     if cv is not None:
#         if cv < 10:
#             status = "Stable"
#             color = (0, 255, 0)
#         elif cv < 20:
#             status = "Slight Fatigue"
#             color = (0, 255, 255)
#         else:
#             status = "Fatigued"
#             color = (0, 0, 255)
#         cv2.putText(display_frame, f"CV: {cv:.1f}%", (10, y_feedback),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
#         y_feedback += 25
#         cv2.putText(display_frame, status, (10, y_feedback),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
#     else:
#         cv2.putText(display_frame, "Balance: Detecting...", (10, y_feedback),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

#     # ────────────── Global Coach Message ──────────────
#     if quality_score > 80:
#         coach_msg = "Great Form"
#         color = (0, 255, 0)
#     elif quality_score > 60:
#         coach_msg = "Adjust Form"
#         color = (0, 255, 255)
#     else:
#         coach_msg = "Fix Form"
#         color = (0, 0, 255)

#     cv2.putText(display_frame, coach_msg, (10, y + 35),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#     # ────────────── Floor Line ──────────────
#     cv2.line(display_frame, (0, floor_y), (display_frame.shape[1], floor_y), (255, 255, 0), 2)

#     return cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)







# ui/frame_display.py
import cv2
import numpy as np
import streamlit as st
from collections import deque

from core.counters import R2PScorer, RepCounter, SwayTracker, extract_features, CMJTracker, calculate_fppa

# ────────────── Process Frame ──────────────
def process_frame(frame, keypoints, keypoints_3d, selected_rules, rules_all, floor_y,
                  cmj_counter, sls_counter, sway_tracker, r2p_scorer):

    display_frame = frame.copy()

    if keypoints is None or keypoints_3d is None:
        cv2.line(display_frame, (0, floor_y), (display_frame.shape[1], floor_y), (255, 255, 0), 2)
        return cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

    # ────────────── Extract Features ──────────────
    features = {}

    # Mid-hip position (for sway/balance)
    mid_hip = None
    if len(keypoints) > 12 and keypoints[11] is not None and keypoints[12] is not None:
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        mid_hip = (
            (left_hip[0] + right_hip[0]) / 2,
            (left_hip[1] + right_hip[1]) / 2
        )
        # Smooth with previous frame
        alpha = 0.7
        if "prev_mid_hip" in st.session_state:
            prev = st.session_state["prev_mid_hip"]
            mid_hip = (
                alpha * mid_hip[0] + (1 - alpha) * prev[0],
                alpha * mid_hip[1] + (1 - alpha) * prev[1]
            )
        st.session_state["prev_mid_hip"] = mid_hip
    elif "prev_mid_hip" in st.session_state:
        mid_hip = st.session_state["prev_mid_hip"]

    features["mid_hip"] = mid_hip

    # Jump height (CMJ)
    if len(keypoints) > 16 and keypoints[15] is not None and keypoints[16] is not None:
        features["jump_feet"] = floor_y - min(keypoints[15][1], keypoints[16][1])
    else:
        features["jump_feet"] = None

    # ────────────── Update trackers ──────────────
    if mid_hip is not None:
        sway_tracker.update(mid_hip)

    sway_velocity = sway_tracker.get_sway_velocity()
    sway_cv = sway_tracker.get_cv()

    # Corrected RepCounter update calls
    delta_rsi = cmj_counter.update(features) 
    sls_reps = sls_counter.update(features)                     # features dictionary

    # ────────────── LIVE FEEDBACK ──────────────
    y = 30
    quality_score = 100

    for rule_name in selected_rules:
        val = features.get(rule_name)
        rule = rules_all.get(rule_name)

        if val is None or rule is None:
            continue

        min_v, max_v = rule

        if val < min_v:
            msg = f"{rule_name}: Too Low"
            color = (0, 0, 255)
            quality_score -= 10
        elif val > max_v:
            msg = f"{rule_name}: Too High"
            color = (0, 165, 255)
            quality_score -= 10
        else:
            msg = f"{rule_name}: Good"
            color = (0, 255, 0)

        cv2.putText(display_frame, msg, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 25

    # ────────────── REP SCORES ──────────────
    if delta_rsi is not None:
        # Convert RSI delta to percentage for display
        rep_score_pct = max(0, 100 - delta_rsi)
        msg = f"CMJ Score: {rep_score_pct:.0f}%"
        color = (0, 255, 0) if rep_score_pct > 80 else (0, 255, 255) if rep_score_pct > 60 else (0, 0, 255)
        cv2.putText(display_frame, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y += 30

    # SLS reps
    cv2.putText(display_frame, f"SLS Reps: {sls_reps}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    y += 30

    # ────────────── BALANCE / SWAY ──────────────
    if sway_velocity is not None:
        cv2.putText(display_frame, f"Sway Vel: {sway_velocity:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 25

    if sway_cv is not None:
        if sway_cv < 10:
            status = "Stable"
            color = (0, 255, 0)
        elif sway_cv < 20:
            status = "Slight Fatigue"
            color = (0, 255, 255)
        else:
            status = "Fatigued"
            color = (0, 0, 255)
        cv2.putText(display_frame, f"CV: {sway_cv:.1f}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 25
        cv2.putText(display_frame, status, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y += 25
    else:
        cv2.putText(display_frame, "Balance: Detecting...", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        y += 25

    # ────────────── GLOBAL COACH MESSAGE ──────────────
    total_score, traffic_light = r2p_scorer.compute(cv=sway_cv, fppa=None, delta_rsi=delta_rsi)

    if traffic_light == "GREEN":
        coach_msg, color = "Great Form", (0, 255, 0)
    elif traffic_light == "YELLOW":
        coach_msg, color = "Adjust Form", (0, 255, 255)
    else:
        coach_msg, color = "Fix Form", (0, 0, 255)

    cv2.putText(display_frame, coach_msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    y += 35

    # ────────────── FLOOR LINE ──────────────
    cv2.line(display_frame, (0, floor_y), (display_frame.shape[1], floor_y), (255, 255, 0), 2)

    return cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)