import paramiko
import scp
import os
import torch
import numpy as np

REMOTE_IP = "101.6.162.37"
REMOTE_PORT = 62222
USERNAME = "ai"
PASSWORD = "CS26S02"  # Note: normally should be stored securely, but this is user-provided

def get_ssh_client():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(REMOTE_IP, port=REMOTE_PORT, username=USERNAME, password=PASSWORD)
    return ssh

def process_video_on_remote(video_path, output_dir="temp_gvhmr_output", f_mm=None):
    os.makedirs(output_dir, exist_ok=True)
    video_filename = os.path.basename(video_path)
    video_base = os.path.splitext(video_filename)[0]
    
    remote_video_dir = "/home/ai/GVHMR/uploads"
    remote_video_path = f"{remote_video_dir}/{video_filename}"
    remote_output_dir = f"/home/ai/GVHMR/outputs"
    remote_result_pt = f"{remote_output_dir}/{video_base}/hmr4d_results.pt"
    
    print("Connecting to remote server...")
    try:
        ssh = get_ssh_client()
    except Exception as e:
        print(f"Failed to connect: {e}")
        return None

    print(f"Uploading video {video_filename} to remote...")
    try:
        ssh.exec_command(f"mkdir -p {remote_video_dir} {remote_output_dir}")
        ssh.exec_command(f"rm -f '{remote_video_path}'")
        with scp.SCPClient(ssh.get_transport()) as scp_client:
            scp_client.put(video_path, remote_video_path)
    except Exception as e:
        print(f"Failed to copy video: {e}")
        ssh.close()
        return None
        
    print("Running GVHMR on remote GPU (this will take a while)...")
    
    # We create a tiny python script remotely to do the decoding there, 
    # since the server already has the SMPL weight files installed!
    remote_decoder_script = f"""
import torch, sys
from hmr4d.model.gvhmr.utils.endecoder import EnDecoder
pt_path = '{remote_result_pt}'
print('Decoding 3D joints on server...')
data = torch.load(pt_path, map_location='cpu')
decoder = EnDecoder()

# add batch dimension (B=1)
smpl_global = {{k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in data['smpl_params_global'].items()}}
smpl_incam = {{k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in data['smpl_params_incam'].items()}}

# Decode and squeeze back the batch dimension
data['joints_3d_global_decoded'] = decoder.fk_v2(**smpl_global).squeeze(0).cpu()
data['joints_3d_incam_decoded'] = decoder.fk_v2(**smpl_incam).squeeze(0).cpu()
torch.save(data, pt_path)
print('Done decoding.')
"""

    remote_decoder_path = f"{remote_video_dir}/decode_{video_base}.py"
    
    try:
        # Write the temporary decoding script to the server
        with scp.SCPClient(ssh.get_transport()) as scp_client:
            # We temporarily write the script to our local disk and push it
            with open("temp_decoder.py", "w") as f:
                f.write(remote_decoder_script)
            scp_client.put("temp_decoder.py", remote_decoder_path)
            os.remove("temp_decoder.py")
    except Exception as e:
        print(f"Failed to copy decoder script: {e}")
        ssh.close()
        return None

    # Command uses bash -lc and explicit source to load conda, activates it, runs demo, and runs decoder!
    
    f_mm_arg = f"--f_mm {f_mm}" if f_mm is not None else ""
    
    cmd = (
        f"bash -lc \"source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; "
        f"conda activate gvhmr && "
        f"cd /home/ai/GVHMR && "
        f"CUDA_VISIBLE_DEVICES=0 python tools/demo/demo.py --video '{remote_video_path}' --output_root '{remote_output_dir}' -s {f_mm_arg} && "
        f"CUDA_VISIBLE_DEVICES=0 python {remote_decoder_path} && "
        f"rm -f {remote_decoder_path}\""
    )
    stdin, stdout, stderr = ssh.exec_command(cmd)
    
    # Wait for the command to finish and print output
    exit_status = stdout.channel.recv_exit_status() 
    if exit_status != 0:
        print("Remote processing failed:")
        print(stderr.read().decode())
        ssh.close()
        return None
        
    print("Processing complete. Downloading 3D parameters...")
    local_result_pt = os.path.join(output_dir, f"{video_base}_hmr4d_results.pt")
    try:
        with scp.SCPClient(ssh.get_transport()) as scp_client:
            scp_client.get(remote_result_pt, local_result_pt)
    except Exception as e:
        print(f"Failed to download results: {e}")
        ssh.close()
        return None
        
    ssh.close()
    
    print("Loading 3D parameters locally...")
    if not os.path.exists(local_result_pt):
        print(f"Failed to find downloaded result file: {local_result_pt}")
        return None
        
    pred = torch.load(local_result_pt, map_location="cpu")
    
    K_fullimg = pred.get("K_fullimg", None)
    joints_3d_global = pred.get("joints_3d_global_decoded", None)
    joints_3d_incam = pred.get("joints_3d_incam_decoded", None)
    
    if joints_3d_global is None or joints_3d_incam is None:
        print("Error: Extracted .pt file did not have decoded joints appended!")
        return None
    
    if isinstance(joints_3d_global, torch.Tensor):
        joints_3d_global = joints_3d_global.numpy()
    if isinstance(joints_3d_incam, torch.Tensor):
        joints_3d_incam = joints_3d_incam.numpy()
    if isinstance(K_fullimg, torch.Tensor):
        K_fullimg = K_fullimg.numpy()
        
    return {
        "joints_3d_global": joints_3d_global,
        "joints_3d_incam": joints_3d_incam,
        "K_fullimg": K_fullimg
    }