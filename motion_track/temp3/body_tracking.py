# body_tracking.py
import cv2
import math
import numpy as np
import torch
import sys
import os
import subprocess

# Auto-add GVHMR to path so we can use its utilities
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
GVHMR_DIR = os.path.join(PROJECT_ROOT, "GVHMR")
if GVHMR_DIR not in sys.path:
    sys.path.append(GVHMR_DIR)

# -----------------------
# Setup processing
# -----------------------
def process_video_gvhmr(video_path, output_dir="temp_gvhmr_output"):
    """
    Runs GVHMR inference on the full video and returns 3D joints and camera parameters
    for each frame.
    """
    print("Starting GVHMR batch processing. This may take some time...")
    os.makedirs(output_dir, exist_ok=True)
    
    demo_script = os.path.join(GVHMR_DIR, "tools", "demo", "demo.py")
    
    # We call the demo pipeline.
    cmd = [
        sys.executable, demo_script,
        "--video", video_path,
        "--output_root", output_dir,
    ]
    
    # Run the demo script to produce the output
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("GVHMR processing failed.")
        return None
        
    # GVHMR demo.py generates `hmr4d_results.pt` inside a subfolder based on video name
    video_base = os.path.splitext(os.path.basename(video_path))[0]
    result_pt = os.path.join(output_dir, video_base, "hmr4d_results.pt")
    
    if not os.path.exists(result_pt):
        print(f"Expected results not found at: {result_pt}")
        return None
        
    print("Loading 3D parameters from:", result_pt)
    pred = torch.load(result_pt, map_location="cpu")
    
    smpl_params_global = pred.get("smpl_params_global", {})
    smpl_params_incam = pred.get("smpl_params_incam", {})
    K_fullimg = pred.get("K_fullimg", None)  # shape (F, 3, 3)
    
    # Use FK to get the 3D joint locations
    try:
        from hmr4d.model.gvhmr.utils.endecoder import endecoder
        joints_3d_global = endecoder.fk_v2(**smpl_params_global)
        joints_3d_incam = endecoder.fk_v2(**smpl_params_incam)
    except ImportError:
        print("Failed to import endecoder from GVHMR.")
        return None
    
    if isinstance(joints_3d_global, torch.Tensor):
        joints_3d_global = joints_3d_global.numpy()
    if isinstance(joints_3d_incam, torch.Tensor):
        joints_3d_incam = joints_3d_incam.numpy()
    if isinstance(K_fullimg, torch.Tensor):
        K_fullimg = K_fullimg.numpy()
        
    # Return dict ready for frame-by-frame consumption
    return {
        "joints_3d_global": joints_3d_global,
        "joints_3d_incam": joints_3d_incam,
        "K_fullimg": K_fullimg
    }

def smpl_to_coco17(joints_smpl):
    """
    Map SMPL 22 joints to COCO 17 joints format to main backwards compatibility.
    joints_smpl: list or array of shape (22, D)
    Returns: list of 17 mapped joints
    """
    coco_17 = [None] * 17
    coco_17[0] = joints_smpl[15] # Nose ~ Head
    coco_17[1] = joints_smpl[15]
    coco_17[2] = joints_smpl[15]
    coco_17[3] = joints_smpl[15]
    coco_17[4] = joints_smpl[15]
    coco_17[5] = joints_smpl[16] # L_Shoulder
    coco_17[6] = joints_smpl[17] # R_Shoulder
    coco_17[7] = joints_smpl[18] # L_Elbow
    coco_17[8] = joints_smpl[19] # R_Elbow
    coco_17[9] = joints_smpl[20] # L_Wrist
    coco_17[10] = joints_smpl[21] # R_Wrist
    coco_17[11] = joints_smpl[1] # L_Hip
    coco_17[12] = joints_smpl[2] # R_Hip
    coco_17[13] = joints_smpl[4] # L_Knee
    coco_17[14] = joints_smpl[5] # R_Knee
    coco_17[15] = joints_smpl[7] # L_Ankle
    coco_17[16] = joints_smpl[8] # R_Ankle
    return coco_17

def project_3d_to_2d(joints_3d_incam, K_fullimg):
    """
    Project one frame's 3D in-camera joints to 2D image coordinates using camera intrinsics.
    joints_3d_incam: (22, 3)
    K_fullimg: (3, 3)
    Returns: (22, 2) list of (x,y)
    """
    projected = []
    for joint in joints_3d_incam:
        z = joint[2] if abs(joint[2]) > 1e-6 else 1e-6
        x2d = np.dot(K_fullimg, joint)
        x = x2d[0] / z
        y = x2d[1] / z
        projected.append((int(x), int(y)))
    return projected
