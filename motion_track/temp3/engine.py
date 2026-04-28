import os
import cv2
import time
import csv
from posture_analysis import load_rules
from core.counters import R2PScorer, RepCounter, SwayTracker, CMJTracker
from body_tracking import project_3d_to_2d, smpl_to_coco17, get_yolo26_keypoints
from remote_ssh_pipeline import process_video_on_remote
from ui.frame_display import process_frame
from ui.sidebar import load_rule_groups

def draw_skeleton(frame, keypoints_2d):
    if not keypoints_2d: return frame
    edges = [(0,1),(0,2),(1,3),(2,4),(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    for pt in keypoints_2d:
        if pt: cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
    for p1, p2 in edges:
        kp1, kp2 = keypoints_2d[p1], keypoints_2d[p2]
        if kp1 and kp2: cv2.line(frame, (int(kp1[0]), int(kp1[1])), (int(kp2[0]), int(kp2[1])), (0, 0, 255), 2)
    return frame

def run_analysis(video_path, task_id, exercise_name="Balance", f_mm=24):
    print(f"Starting background analysis for {exercise_name} (Task {task_id})...")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    rule_groups = load_rule_groups(os.path.join(BASE_DIR, "assets", "rules.txt"))
    rules_all = load_rules(os.path.join(BASE_DIR, "assets", "rules.txt"))
    selected_rules = rule_groups.get(exercise_name, [])

    cmj_counter = CMJTracker(fps=30)
    sls_counter = RepCounter(exercise="SLS", feature="sls_fppa", min_angle=5, max_angle=15)
    balance_tracker = SwayTracker(fps=30)
    r2p_scorer = R2PScorer()

    cap = cv2.VideoCapture(video_path)
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    # WebM from browser often reports 1000 FPS causing hyper-accelerated output.
    if raw_fps > 60 or raw_fps < 5 or str(raw_fps) == "nan":
        fps = 30
    else:
        fps = int(raw_fps)
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup Video Writer with WebM/vp80 codec
    out_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{task_id}_annotated.webm")
    fourcc = cv2.VideoWriter_fourcc(*'vp80') 
    
    print(f"Video Writer Params: File={out_path}, FPS={fps}, Width={width}, Height={height}")
    
    use_yolo26 = exercise_name in ["SLS", "Proper Squat"] #add the name of the exercise for which you want YOLO26 to be used
    gvhmr_results = None
    
    if not use_yolo26:
        gvhmr_results = process_video_on_remote(video_path, f_mm=f_mm)

    idx = 0
    # Read first frame to ensure correct video dimensions for VideoWriter
    ret, first_frame = cap.read()
    if not ret:
        print("ERROR: Video is completely empty or cannot be read by OpenCV.")
        return out_path

    # Extract ACTUAL dimensions from the first frame
    frame_h, frame_w = first_frame.shape[:2]
    if width == 0 or height == 0 or width != frame_w or height != frame_h:
        print(f"WARNING: Fixing dimensions from ({width}x{height}) to ({frame_w}x{frame_h})")
        width, height = frame_w, frame_h
    if fps == 0:
        fps = 30

    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Process the first frame we already read
    frame = first_frame
    while ret:
        annotated = frame.copy()
        keypoints_3d = None
        keypoints_2d = None

        if use_yolo26:
            keypoints_2d_raw = get_yolo26_keypoints(frame)
            keypoints_3d = [(pt[0], pt[1], 0.0) if pt else None for pt in keypoints_2d_raw]
            keypoints_2d = [(int(pt[0]), int(pt[1])) if pt else None for pt in keypoints_2d_raw]
            annotated = draw_skeleton(annotated, keypoints_2d)
        elif gvhmr_results:
            num_frames = len(gvhmr_results["joints_3d_global"])
            idx_bounded = min(idx, num_frames - 1)
            smpl_global = gvhmr_results["joints_3d_global"][idx_bounded]
            smpl_incam = gvhmr_results["joints_3d_incam"][idx_bounded]
            K = gvhmr_results["K_fullimg"][idx_bounded] if gvhmr_results["K_fullimg"].ndim == 3 else gvhmr_results["K_fullimg"]
            keypoints_3d = smpl_to_coco17(smpl_global)
            joints_2d = project_3d_to_2d(smpl_incam, K)
            keypoints_2d = smpl_to_coco17(joints_2d)
            keypoints_2d = [(int(pt[0]), int(pt[1])) if pt else None for pt in keypoints_2d]
            annotated = draw_skeleton(annotated, keypoints_2d)

        # Process logic
        processed_frame = process_frame(
            annotated, keypoints_2d, keypoints_3d, selected_rules, rules_all,
            500, cmj_counter, sls_counter, balance_tracker, r2p_scorer
        )
        
        # Write frame to final video
        out.write(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
        idx += 1
        ret, frame = cap.read()

    cap.release()
    out.release()
    
    # Save results similarly to main.py logic (merged partner functionality)
    metrics_path = os.path.join(out_dir, "metrics.csv")
    cv_val = balance_tracker.get_cv()
    one_minus_cv = balance_tracker.get_one_minus_cv()
    file_exists = os.path.isfile(metrics_path)
    with open(metrics_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["video", "cv", "one_minus_cv"])
        writer.writerow([f"{task_id}_annotated.webm", cv_val, one_minus_cv])

    print(f"Task {task_id} complete! Saved {cv_val:.2f}% CV to {out_path}")
    return out_path