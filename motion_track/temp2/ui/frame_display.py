# ui/frame_display.py
import cv2
import numpy as np
from core.posture import extract_features, evaluate_multiple_rules, draw_feedback

def process_frame(frame, keypoints, selected_rules, rules_all, floor_y,
                  cmj_counter, sls_counter, sway_tracker):
    display_frame = frame.copy()

    # If keypoints are None
    if keypoints is None:
        cv2.line(display_frame, (0, floor_y), (display_frame.shape[1], floor_y), (255, 255, 0), 2)
        return cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

    features = extract_features(keypoints)

    # Jump feet (CMJ)
    if len(keypoints) > 16 and keypoints[15] is not None and keypoints[16] is not None:
        features["jump_feet"] = floor_y - min(keypoints[15][1], keypoints[16][1])
    else:
        features["jump_feet"] = None

    # Mid hip (Balance)
    if len(keypoints) > 12 and keypoints[11] is not None and keypoints[12] is not None:
        mid_hip = ((keypoints[11][0] + keypoints[12][0]) / 2,
                   (keypoints[11][1] + keypoints[12][1]) / 2)
        features["mid_hip"] = (mid_hip[0], floor_y - mid_hip[1])
    else:
        features["mid_hip"] = None

    # Update sway tracker
    if features.get("mid_hip") is not None:
        sway_tracker.update(features["mid_hip"])
        features["sway_velocity"] = sway_tracker.get_sway_velocity()
    else:
        features["sway_velocity"] = None

    # Evaluate rules
    results = evaluate_multiple_rules(features, rules_all, selected_rules)
    all_failed = set()
    y = 30
    for name, result in results.items():
        score = result["score"]
        failed = result["failed"]
        all_failed.update(failed)
        cv2.putText(display_frame, f"{name}: {score}%", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 25
        if failed:
            cv2.putText(display_frame, "Fail: " + ", ".join(failed), (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y += 25

    draw_feedback(display_frame, keypoints, all_failed)

    # Update counters
    reps = 0
    reps += cmj_counter.update(features)
    reps += sls_counter.update(features)
    cv2.putText(display_frame, f"Reps: {reps}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    y += 30

    # Sway display
    sway_velocity = features.get("sway_velocity")
    if sway_velocity is not None:
        baseline = 5
        change = (sway_velocity - baseline) / baseline * 100
        if change < 10:
            ftxt, fcolor = "Stable", (0, 255, 0)
        elif change < 20:
            ftxt, fcolor = "Slight Fatigue", (0, 255, 255)
        else:
            ftxt, fcolor = "Fatigued", (0, 0, 255)
        cv2.putText(display_frame, f"Sway: {sway_velocity:.2f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, fcolor, 2)
        y += 25
        cv2.putText(display_frame, ftxt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fcolor, 2)

    # Floor line
    cv2.line(display_frame, (0, floor_y), (display_frame.shape[1], floor_y), (255, 255, 0), 2)

    return cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

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