import math
import cv2

# ==============================
# Utility Functions
# ==============================

def calculate_angle(a, b, c):
    try:
        ax, ay = a
        bx, by = b
        cx, cy = c

        ab = (ax - bx, ay - by)
        cb = (cx - bx, cy - by)

        dot = ab[0]*cb[0] + ab[1]*cb[1]
        mag_ab = math.sqrt(ab[0]**2 + ab[1]**2)
        mag_cb = math.sqrt(cb[0]**2 + cb[1]**2)

        if mag_ab * mag_cb == 0:
            return None

        angle = math.degrees(math.acos(dot / (mag_ab * mag_cb)))
        return angle
    except:
        return None


def get_distance(p1, p2):
    if p1 is None or p2 is None:
        return None
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


# ==============================
# Feature Extraction
# ==============================

def extract_features(keypoints):

    def kp(i):
        return keypoints[i] if i < len(keypoints) else None

    features = {}

    features["left_knee"] = calculate_angle(kp(11), kp(13), kp(15))
    features["right_knee"] = calculate_angle(kp(12), kp(14), kp(16))

    features["left_arm_vertical"] = calculate_angle(
        kp(7), kp(5), (kp(5)[0], kp(5)[1] - 100)
    ) if kp(5) else None

    features["right_arm_vertical"] = calculate_angle(
        kp(8), kp(6), (kp(6)[0], kp(6)[1] - 100)
    ) if kp(6) else None

    if kp(5) and kp(11):
        dx = kp(5)[0] - kp(11)[0]
        dy = kp(5)[1] - kp(11)[1]
        features["body_tilt"] = math.degrees(math.atan2(dy, dx))
    else:
        features["body_tilt"] = None

    features["jump_feet"] = get_distance(kp(15), kp(16))

    return features


# ==============================
# Load Rules (Robust)
# ==============================

def load_rules(file_path):
    rules = {}
    current_rule = None

    with open(file_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            if ":" not in line:
                current_rule = line
                rules[current_rule] = {}
                continue

            try:
                part, values = line.split(":", 1)
                part = part.strip()
                values = values.strip().replace("\r", "")

                if not values or "-" not in values:
                    continue

                min_val, max_val = values.split("-", 1)

                rules[current_rule][part] = (
                    float(min_val.strip()),
                    float(max_val.strip())
                )

            except:
                print(f"Skipping bad line: {line}")

    return rules


# ==============================
# Rule Evaluation
# ==============================

def evaluate_rule(features, rule):
    score = 0
    total = len(rule)
    failed = []

    for part, (min_val, max_val) in rule.items():
        val = features.get(part)

        if val is None or not (min_val <= val <= max_val):
            failed.append(part)
        else:
            score += 1

    if total == 0:
        return 0, failed

    return int((score / total) * 100), failed


def evaluate_multiple_rules(features, rules, selected_rules):
    results = {}

    for name in selected_rules:
        if name in rules:
            score, failed = evaluate_rule(features, rules[name])
            results[name] = {
                "score": score,
                "failed": failed
            }

    return results


# ==============================
# Joint Highlighting
# ==============================

JOINT_MAP = {
    "left_knee": [13],
    "right_knee": [14],
    "left_arm_vertical": [5, 7],
    "right_arm_vertical": [6, 8],
    "body_tilt": [5, 11],
    "jump_feet": [15, 16]
}


def draw_feedback(frame, keypoints, failed_parts):
    for part, indices in JOINT_MAP.items():
        color = (0, 255, 0)  # green

        if part in failed_parts:
            color = (0, 0, 255)  # red

        for i in indices:
            if i < len(keypoints) and keypoints[i] is not None:
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                cv2.circle(frame, (x, y), 8, color, -1)


# ==============================
# Rep Counter
# ==============================

class RepCounter:
    """
    Counts exercise repetitions based on joint angles.
    Example: Proper squat counting using left_knee angle.
    """
    def __init__(self, exercise, feature, min_angle=70, max_angle=110):
        self.exercise = exercise
        self.feature = feature
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.reps = 0
        self.state = "up"  # "up" or "down"

    def update(self, features):
        angle = features.get(self.feature)

        if angle is None:
            return self.reps

        if self.state == "up" and angle < self.min_angle:
            self.state = "down"
        elif self.state == "down" and angle > self.max_angle:
            self.state = "up"
            self.reps += 1

        return self.reps


# ==============================
# Sway Tracker
# ==============================

class SwayTracker:
    """
    Tracks mid-hip position over time to estimate sway velocity.
    Useful for balance assessment.
    """
    def __init__(self, max_history=10):
        self.max_history = max_history
        self.history = []  # stores y-coordinate differences
        self.last_pos = None

    def update(self, mid_hip):
        """
        mid_hip: tuple (x, y) or None
        """
        if mid_hip is None:
            return None

        x, y = mid_hip
        if self.last_pos is not None:
            dy = y - self.last_pos[1]
            self.history.append(dy)
            if len(self.history) > self.max_history:
                self.history.pop(0)

        self.last_pos = (x, y)
        return self.get_sway_velocity()

    def get_sway_velocity(self):
        if not self.history:
            return 0.0
        # simple average absolute difference per frame
        return sum(abs(dy) for dy in self.history) / len(self.history)