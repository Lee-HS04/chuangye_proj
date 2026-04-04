# core/posture.py
import cv2
from core.utils import calculate_angle

def extract_features(keypoints):
    features = {}
    if keypoints is None:
        return features
        
    def valid_pt(arr):
        # A keypoint is valid if it isn't None and isn't an empty array/list
        if arr is None: return False
        if hasattr(arr, 'size'): return arr.size > 0
        return len(arr) > 0

    # FPPA example
    if all(valid_pt(keypoints[i]) for i in [11,13,15]):
        features["FPPA_left"] = calculate_angle(keypoints[13], keypoints[11], keypoints[15])
    if all(valid_pt(keypoints[i]) for i in [12,14,16]):
        features["FPPA_right"] = calculate_angle(keypoints[14], keypoints[12], keypoints[16])
    if valid_pt(keypoints[11]) and valid_pt(keypoints[12]):
        mid_hip = ((keypoints[11][0]+keypoints[12][0])/2, (keypoints[11][1]+keypoints[12][1])/2)
        features["mid_hip"] = mid_hip
    return features

def evaluate_multiple_rules(features, rules_all, selected_rules):
    results = {}
    for rule_name in selected_rules:
        rule = rules_all.get(rule_name,{})
        score = 100
        failed = []
        for feat, limits in rule.items():
            val = features.get(feat)
            if val is None:
                continue
            if val<limits["min"] or val>limits["max"]:
                failed.append(feat)
                score -= 50/len(rule)
        results[rule_name] = {"score": max(0,int(score)),"failed":failed}
    return results

def draw_feedback(frame, keypoints, failed_features):
    for feat in failed_features:
        cv2.putText(frame, f"Check {feat}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)