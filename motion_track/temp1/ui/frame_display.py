import cv2
import numpy as np
from body_tracking import get_keypoints
from core.posture import process_keypoints

def render_frame(frame, floor_y, rules_all, selected_rules, counter, sway_tracker):
    keypoints, annotated = get_keypoints(frame)
    display_frame = annotated.copy() if annotated is not None else frame.copy()

    all_failed = set()
    reps = 0

    if keypoints:
        features, results, all_failed, reps = process_keypoints(
            keypoints, floor_y, rules_all, selected_rules, counter, sway_tracker
        )

        # Draw rule results
        y = 30
        for name, result in results.items():
            score  = result.get("score", 0)
            failed = result.get("failed", [])
            cv2.putText(display_frame, f"{name}: {score}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
            y += 25
            if failed:
                cv2.putText(display_frame, "Fail: " + ", ".join(failed),
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
                y += 25

        # Draw feedback and reps
        from posture_analysis import draw_feedback
        draw_feedback(display_frame, keypoints, all_failed)
        cv2.putText(display_frame, f"Reps: {reps}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        # Sway info
        sway_velocity = features.get("sway_velocity")
        if sway_velocity is not None:
            baseline = 5
            change = (sway_velocity - baseline) / baseline * 100
            if change < 10:
                ftxt, fcolor = "Stable", (0,255,0)
            elif change < 20:
                ftxt, fcolor = "Slight Fatigue", (0,255,255)
            else:
                ftxt, fcolor = "Fatigued", (0,0,255)
            y += 25
            cv2.putText(display_frame, f"Sway: {sway_velocity:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX,0.7,fcolor,2)
            y += 25
            cv2.putText(display_frame, ftxt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,0.7,fcolor,2)

    # Draw floor line
    cv2.line(display_frame, (0,floor_y), (display_frame.shape[1], floor_y), (255,255,0), 2)
    return cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)