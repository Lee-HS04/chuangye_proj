import cv2
import numpy as np
import streamlit as st
from collections import deque

from core.posture import extract_features


# ────────────── Smart Rep Counter ──────────────
class SmartRepCounter:
    def __init__(self, feature_name, min_val, max_val):
        self.feature = feature_name
        self.min_val = min_val
        self.max_val = max_val
        self.state = "TOP"
        self.reps = 0
        self.current_min = float("inf")

    def update(self, features):
        val = features.get(self.feature)
        if val is None:
            return None

        self.current_min = min(self.current_min, val)

        # State machine
        if self.state == "TOP" and val < self.max_val:
            self.state = "DOWN"

        elif self.state == "DOWN" and val <= self.min_val:
            self.state = "BOTTOM"

        elif self.state == "BOTTOM" and val > self.min_val:
            self.state = "UP"

        elif self.state == "UP" and val >= self.max_val:
            self.state = "TOP"
            self.reps += 1

            # Score based on depth
            score = max(0, min(100, int(100 * (self.max_val - self.current_min) / (self.max_val - self.min_val))))
            self.current_min = float("inf")
            return score

        return None


# ────────────── Process Frame ──────────────
def process_frame(frame, keypoints, selected_rules, rules_all, floor_y,
                  cmj_counter, sls_counter, sway_tracker):

    display_frame = frame.copy()

    if keypoints is None:
        cv2.line(display_frame, (0, floor_y), (display_frame.shape[1], floor_y), (255, 255, 0), 2)
        return cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

    features = extract_features(keypoints)

    # ────────────── Derived Features ──────────────
    if len(keypoints) > 16 and keypoints[15] and keypoints[16]:
        features["jump_feet"] = floor_y - min(keypoints[15][1], keypoints[16][1])

    # if len(keypoints) > 12 and keypoints[11] and keypoints[12]:
    #     mid_hip = ((keypoints[11][0] + keypoints[12][0]) / 2,
    #                (keypoints[11][1] + keypoints[12][1]) / 2)
    #     features["mid_hip"] = (mid_hip[0], floor_y - mid_hip[1])
    #     sway_tracker.update(features["mid_hip"])
    #     features["sway_velocity"] = sway_tracker.get_sway_velocity()

    # ────────────── Mid-Hip (Balance) ──────────────
    mid_hip = None
    if len(keypoints) > 12 and keypoints[11] is not None and keypoints[12] is not None:
        left_hip = keypoints[11]
        right_hip = keypoints[12]

        # Support both (x, y) and (x, y, conf) formats
        if len(left_hip) >= 2 and len(right_hip) >= 2:
            mid_hip = (
                (left_hip[0] + right_hip[0]) / 2,
                (left_hip[1] + right_hip[1]) / 2
            )

            # 🔥 Smoothing with previous frame
            alpha = 0.7 #0.7
            if "prev_mid_hip" in st.session_state:
                prev = st.session_state["prev_mid_hip"]
                mid_hip = (
                    alpha * mid_hip[0] + (1 - alpha) * prev[0],
                    alpha * mid_hip[1] + (1 - alpha) * prev[1]
                )

            # Save for next frame
            st.session_state["prev_mid_hip"] = mid_hip

    # Fallback to previous mid-hip if current is None
    if mid_hip is None and "prev_mid_hip" in st.session_state:
        mid_hip = st.session_state["prev_mid_hip"]

    features["mid_hip"] = mid_hip

    # Update sway tracker every frame
    if mid_hip is not None:
        sway_tracker.update(mid_hip)

    # Compute sway velocity & CV
    features["sway_velocity"] = sway_tracker.get_sway_velocity()
    features["sway_cv"] = sway_tracker.get_cv()

    # # ────────────── Display Balance Feedback ──────────────
    # y_feedback = 30
    # sway_velocity = features.get("sway_velocity")
    # cv = features.get("sway_cv")

    # if sway_velocity is not None:
    #     cv2.putText(display_frame, f"Sway Vel: {sway_velocity:.2f}", (10, y_feedback),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    #     y_feedback += 25

    # if cv is not None:
    #     if cv < 10:
    #         status = "Stable"
    #         color = (0, 255, 0)
    #     elif cv < 20:
    #         status = "Slight Fatigue"
    #         color = (0, 255, 255)
    #     else:
    #         status = "Fatigued"
    #         color = (0, 0, 255)
    #     cv2.putText(display_frame, f"CV: {cv:.1f}%", (10, y_feedback),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    #     y_feedback += 25
    #     cv2.putText(display_frame, status, (10, y_feedback),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    # else:
    #     cv2.putText(display_frame, "Balance: Detecting...", (10, y_feedback),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)


    # ────────────── LIVE COACHING ──────────────
    y = 30
    feedback_msgs = []
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

        feedback_msgs.append(msg)

        cv2.putText(display_frame, msg, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 25

    # ────────────── REP COUNTING ──────────────
    rep_score = cmj_counter.update(features)

    # Total reps (always visible)
    cv2.putText(display_frame, f"Reps: {cmj_counter.reps}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    y += 30

    # Show last rep score (only when completed)
    if rep_score is not None:
        msg = f"Last Rep: {rep_score}%"

        # Color based on quality
        if rep_score > 80:
            color = (0, 255, 0)
        elif rep_score > 60:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

        cv2.putText(display_frame, msg, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y += 30

    # ────────────── FATIGUE (LIKE SWAY) ──────────────
    if "fatigue_hist" not in sway_tracker.__dict__:
        sway_tracker.fatigue_hist = deque(maxlen=30)

    sway_val = features.get("sway_velocity")

    if sway_val is not None:
        sway_tracker.fatigue_hist.append(sway_val)

        avg = np.mean(sway_tracker.fatigue_hist)

        if avg < 5:
            status, color = "Fresh", (0, 255, 0)
        elif avg < 10:
            status, color = "Slight Fatigue", (0, 255, 255)
        else:
            status, color = "Fatigued", (0, 0, 255)

        cv2.putText(display_frame, f"Fatigue: {status}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 25

# # ────────────── Update Sway Tracker ──────────────
#     if features.get("mid_hip") is not None:
#         sway_tracker.update(features["mid_hip"])
#         sway_velocity = sway_tracker.get_sway_velocity()
#         cv = sway_tracker.get_cv()

#         # Display on the RIGHT side
#         x_pos = display_frame.shape[1] - 300

#         if sway_velocity is not None:
#             cv2.putText(display_frame, f"Sway Vel: {sway_velocity:.2f}", (x_pos, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#         if cv is not None:
#             # Scientific thresholds
#             if cv < 10:
#                 status = "Stable"
#                 color = (0, 255, 0)
#             elif cv < 20:
#                 status = "Slight Fatigue"
#                 color = (0, 255, 255)
#             else:
#                 status = "Fatigued"
#                 color = (0, 0, 255)

            
#             print("Frame index:", st.session_state.get("frame_index"))
#             print("Mid-Hip:", features.get("mid_hip"))
#             print("Sway Vel:", sway_tracker.get_sway_velocity())
#             print("CV:", sway_tracker.get_cv())

#             cv2.putText(display_frame, f"CV: {cv:.1f}%", (x_pos, 80),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
#             cv2.putText(display_frame, status, (x_pos, 110),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


    # ────────────── Display Balance Feedback ──────────────
    y_feedback = 150
    sway_velocity = features.get("sway_velocity")
    cv = features.get("sway_cv")

    if sway_velocity is not None:
        cv2.putText(display_frame, f"Sway Vel: {sway_velocity:.2f}", (10, y_feedback),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_feedback += 25

    if cv is not None:
        if cv < 10:
            status = "Stable"
            color = (0, 255, 0)
        elif cv < 20:
            status = "Slight Fatigue"
            color = (0, 255, 255)
        else:
            status = "Fatigued"
            color = (0, 0, 255)
        cv2.putText(display_frame, f"CV: {cv:.1f}%", (10, y_feedback),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_feedback += 25
        cv2.putText(display_frame, status, (10, y_feedback),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        cv2.putText(display_frame, "Balance: Detecting...", (10, y_feedback),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
    # ────────────── GLOBAL COACH MESSAGE ──────────────
    if quality_score > 80:
        coach_msg = "Great Form "
        color = (0, 255, 0)
    elif quality_score > 60:
        coach_msg = "Adjust Form "
        color = (0, 255, 255)
    else:
        coach_msg = "Fix Form "
        color = (0, 0, 255)


    cv2.putText(display_frame, coach_msg, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    y += 35

    # ────────────── Floor Line ──────────────
    cv2.line(display_frame, (0, floor_y), (display_frame.shape[1], floor_y), (255, 255, 0), 2)

    return cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)



# # ui/frame_display.py
# import cv2
# import numpy as np
# from core.posture import extract_features, evaluate_multiple_rules, draw_feedback

# def process_frame(frame, keypoints, selected_rules, rules_all, floor_y,
#                   cmj_counter, sls_counter, sway_tracker):
#     display_frame = frame.copy()

#     # If keypoints are None
#     if keypoints is None:
#         cv2.line(display_frame, (0, floor_y), (display_frame.shape[1], floor_y), (255, 255, 0), 2)
#         return cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

#     features = extract_features(keypoints)

#     # Jump feet (CMJ)
#     if len(keypoints) > 16 and keypoints[15] is not None and keypoints[16] is not None:
#         features["jump_feet"] = floor_y - min(keypoints[15][1], keypoints[16][1])
#     else:
#         features["jump_feet"] = None

#     # Mid hip (Balance)
#     if len(keypoints) > 12 and keypoints[11] is not None and keypoints[12] is not None:
#         mid_hip = ((keypoints[11][0] + keypoints[12][0]) / 2,
#                    (keypoints[11][1] + keypoints[12][1]) / 2)
#         features["mid_hip"] = (mid_hip[0], floor_y - mid_hip[1])
#     else:
#         features["mid_hip"] = None

#     # Update sway tracker
#     if features.get("mid_hip") is not None:
#         sway_tracker.update(features["mid_hip"])
#         features["sway_velocity"] = sway_tracker.get_sway_velocity()
#     else:
#         features["sway_velocity"] = None

#     # Evaluate rules
#     results = evaluate_multiple_rules(features, rules_all, selected_rules)
#     all_failed = set()
#     y = 30
#     for name, result in results.items():
#         score = result["score"]
#         failed = result["failed"]
#         all_failed.update(failed)
#         cv2.putText(display_frame, f"{name}: {score}%", (10, y),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#         y += 25
#         if failed:
#             cv2.putText(display_frame, "Fail: " + ", ".join(failed), (10, y),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#             y += 25

#     draw_feedback(display_frame, keypoints, all_failed)

#     # Update counters
#     reps = 0
#     reps += cmj_counter.update(features)
#     reps += sls_counter.update(features)
#     cv2.putText(display_frame, f"Reps: {reps}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
#     y += 30

#     # Sway display
#     sway_velocity = features.get("sway_velocity")
#     if sway_velocity is not None:
#         baseline = 5
#         change = (sway_velocity - baseline) / baseline * 100
#         if change < 10:
#             ftxt, fcolor = "Stable", (0, 255, 0)
#         elif change < 20:
#             ftxt, fcolor = "Slight Fatigue", (0, 255, 255)
#         else:
#             ftxt, fcolor = "Fatigued", (0, 0, 255)
#         cv2.putText(display_frame, f"Sway: {sway_velocity:.2f}", (10, y),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, fcolor, 2)
#         y += 25
#         cv2.putText(display_frame, ftxt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fcolor, 2)

#     # Floor line
#     cv2.line(display_frame, (0, floor_y), (display_frame.shape[1], floor_y), (255, 255, 0), 2)

#     return cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)






# # ui/frame_display.py
# import cv2
# import numpy as np
# from posture_analysis import extract_features, evaluate_exercise, draw_feedback


# # ── Colour helpers ────────────────────────────────────────────────────────────

# def _score_color(score: int):
#     """BGR: green ≥75, cyan ≥50, red <50."""
#     if score >= 75:
#         return (50, 200, 50)
#     elif score >= 50:
#         return (50, 200, 200)
#     else:
#         return (50, 50, 220)


# def _draw_progress_bar(frame, x, y, w, h, score, bg=(60, 60, 60)):
#     """Draw a filled progress bar directly onto frame (in-place, BGR)."""
#     cv2.rectangle(frame, (x, y), (x + w, y + h), bg, -1)
#     filled = int(w * score / 100)
#     if filled > 0:
#         cv2.rectangle(frame, (x, y), (x + filled, y + h), _score_color(score), -1)


# def _draw_score_card(frame, exercise_name, agg_score, rule_results,
#                      sway_text, sway_color, reps):
#     """
#     Draw a semi-transparent score card in the top-left corner (in-place, BGR).

#     Layout:
#     ┌──────────────────────────────┐
#     │  EXERCISE NAME        87 %   │  ← aggregate, colour-coded
#     │  ══════════════════════      │  ← full-width agg bar
#     │  ├ Rule name   ████░░  92 ✔  │
#     │  └ Rule name   ████░░  74 ✘  │  ← per-rule rows
#     │  ─────────────────────────   │
#     │  Sway 3.2  Stable   Reps: 4  │
#     └──────────────────────────────┘
#     """
#     FONT   = cv2.FONT_HERSHEY_SIMPLEX
#     PAD    = 10
#     LH     = 28          # line height
#     BAR_W  = 60
#     BAR_H  = 8
#     CARD_W = 290

#     n_rules  = len(rule_results)
#     has_foot = bool(sway_text) or reps > 0
#     card_h   = PAD + LH + LH + n_rules * LH + (LH if has_foot else 0) + PAD

#     # Semi-transparent dark background
#     overlay = frame.copy()
#     cv2.rectangle(overlay, (0, 0), (CARD_W, card_h), (15, 15, 15), -1)
#     cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

#     y = PAD + 18

#     # ── Header: exercise name + aggregate score ───────────────────────────────
#     agg_col  = _score_color(agg_score)
#     ex_label = exercise_name.upper()[:24]
#     cv2.putText(frame, ex_label, (PAD, y), FONT, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
#     score_str = f"{agg_score}%"
#     (sw, _), _ = cv2.getTextSize(score_str, FONT, 0.65, 2)
#     cv2.putText(frame, score_str, (CARD_W - PAD - sw, y), FONT, 0.65, agg_col, 2, cv2.LINE_AA)

#     # Aggregate bar (full width under header)
#     y += 6
#     _draw_progress_bar(frame, PAD, y, CARD_W - 2 * PAD, BAR_H - 2, agg_score)

#     # Divider
#     y += BAR_H + 4
#     cv2.line(frame, (PAD, y), (CARD_W - PAD, y), (70, 70, 70), 1)
#     y += 8

#     # ── Per-rule rows ─────────────────────────────────────────────────────────
#     items = list(rule_results.items())
#     for i, (name, res) in enumerate(items):
#         score  = res["score"]
#         value  = res.get("value")
#         failed = bool(res["failed"])
#         col    = _score_color(score)

#         prefix = "└" if i == len(items) - 1 else "├"
#         short  = name[:18]
#         cv2.putText(frame, f"{prefix} {short}", (PAD, y),
#                     FONT, 0.42, (185, 185, 185), 1, cv2.LINE_AA)

#         # Mini bar
#         bx = PAD + 160
#         by = y - BAR_H + 1
#         _draw_progress_bar(frame, bx, by, BAR_W, BAR_H, score)

#         # Numeric score
#         sc_str = f"{score}"
#         (scw, _), _ = cv2.getTextSize(sc_str, FONT, 0.40, 1)
#         cv2.putText(frame, sc_str, (bx + BAR_W + 4, y),
#                     FONT, 0.40, col, 1, cv2.LINE_AA)

#         # OK / !! marker
#         mark = "OK" if not failed else "!!"
#         mcol = (50, 210, 50) if not failed else (50, 50, 220)
#         (mw, _), _ = cv2.getTextSize(mark, FONT, 0.38, 1)
#         cv2.putText(frame, mark, (CARD_W - PAD - mw, y),
#                     FONT, 0.38, mcol, 1, cv2.LINE_AA)

#         # Value hint (small grey)
#         if value is not None:
#             vstr = f"{value:.1f}"
#             cv2.putText(frame, vstr, (PAD + 120, y),
#                         FONT, 0.36, (120, 120, 120), 1, cv2.LINE_AA)

#         y += LH

#     # ── Footer: sway + reps ───────────────────────────────────────────────────
#     if has_foot:
#         cv2.line(frame, (PAD, y - 4), (CARD_W - PAD, y - 4), (60, 60, 60), 1)
#         footer = ""
#         if sway_text:
#             footer += sway_text
#         if reps > 0:
#             footer += f"   Reps: {reps}"
#         cv2.putText(frame, footer.strip(), (PAD, y + 10),
#                     FONT, 0.40, sway_color, 1, cv2.LINE_AA)


# # ── Public entry point ────────────────────────────────────────────────────────

# def process_frame(frame, keypoints, selected_rules, rules_all, floor_y,
#                   cmj_counter, sls_counter, sway_tracker,
#                   exercise_name="Exercise"):
#     display_frame = frame.copy()

#     # Floor line
#     cv2.line(display_frame, (0, floor_y),
#              (display_frame.shape[1], floor_y), (255, 255, 0), 2)

#     if keypoints is None:
#         return cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

#     features = extract_features(keypoints)

#     # Jump height (CMJ)
#     if len(keypoints) > 16 and keypoints[15] is not None and keypoints[16] is not None:
#         features["jump_feet"] = floor_y - min(keypoints[15][1], keypoints[16][1])
#     else:
#         features["jump_feet"] = None

#     # Mid-hip (balance / sway)
#     if len(keypoints) > 12 and keypoints[11] is not None and keypoints[12] is not None:
#         mid_hip = (
#             (keypoints[11][0] + keypoints[12][0]) / 2,
#             (keypoints[11][1] + keypoints[12][1]) / 2,
#         )
#         features["mid_hip"] = (mid_hip[0], floor_y - mid_hip[1])
#     else:
#         features["mid_hip"] = None

#     # Sway velocity
#     if features.get("mid_hip") is not None:
#         sway_tracker.update(features["mid_hip"])
#         features["sway_velocity"] = sway_tracker.get_sway_velocity()
#     else:
#         features["sway_velocity"] = None

#     # Rep counters
#     reps  = cmj_counter.update(features)
#     reps += sls_counter.update(features)

#     # Evaluate selected rules as one exercise
#     ex = evaluate_exercise(features, rules_all, selected_rules)
#     agg_score    = ex["score"]
#     rule_results = ex["rules"]
#     all_failed   = set(ex["failed"])

#     # Joint highlight overlay
#     draw_feedback(display_frame, keypoints, all_failed)

#     # Sway label
#     sway_text, sway_color = "", (200, 200, 200)
#     sv = features.get("sway_velocity")
#     if sv is not None:
#         change = (sv - 5) / 5 * 100
#         if change < 10:
#             sway_text, sway_color = f"Sway {sv:.1f}  Stable",         (50, 210, 50)
#         elif change < 20:
#             sway_text, sway_color = f"Sway {sv:.1f}  Slight fatigue", (50, 210, 210)
#         else:
#             sway_text, sway_color = f"Sway {sv:.1f}  Fatigued",       (50, 50, 220)

#     # Score card
#     _draw_score_card(
#         display_frame,
#         exercise_name,
#         agg_score,
#         rule_results,
#         sway_text,
#         sway_color,
#         reps,
#     )

#     return cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)