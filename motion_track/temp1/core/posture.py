from posture_analysis import extract_features, evaluate_multiple_rules

def process_keypoints(keypoints, floor_y, rules_all, selected_rules, counter, sway_tracker):
    features = extract_features(keypoints)

    # Jump height
    if len(keypoints) > 16 and keypoints[15] and keypoints[16]:
        features["jump_feet"] = floor_y - min(keypoints[15][1], keypoints[16][1])
    else:
        features["jump_feet"] = None

    # Mid hip
    if len(keypoints) > 12 and keypoints[11] and keypoints[12]:
        mid_hip = ((keypoints[11][0]+keypoints[12][0])/2,
                   (keypoints[11][1]+keypoints[12][1])/2)
        features["mid_hip"] = (mid_hip[0], floor_y - mid_hip[1])
    else:
        features["mid_hip"] = None

    # Sway
    sway_tracker.update(features.get("mid_hip"))
    features["sway_velocity"] = sway_tracker.get_sway_velocity()

    # Evaluate rules
    results = evaluate_multiple_rules(features, rules_all, selected_rules)
    all_failed = set()
    for r in results.values():
        all_failed.update(r.get("failed", []))

    # Update reps
    reps = counter.update(features)

    return features, results, all_failed, reps