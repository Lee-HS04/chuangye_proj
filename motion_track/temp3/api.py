from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import uuid
import shutil
import base64
import cv2
import numpy as np

from engine import run_analysis
from body_tracking import get_yolo26_keypoints
from core.state_machine import StateMachineFSM

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount outputs so the frontend can stream the generated videos
app.mount("/videos", StaticFiles(directory=OUTPUT_DIR), name="videos")

# In-memory mock database tracking video status
jobs = {}

def analyze_in_background(task_id: str, file_path: str, exercise_name: str):
    try:
        jobs[task_id] = {"status": "processing"}
        out_path = run_analysis(file_path, task_id, exercise_name)
        jobs[task_id] = {
            "status": "completed", 
            # Changed to .webm for native browser playback compatibility
            "result_video": f"/videos/{task_id}_annotated.webm" 
        }
    except Exception as e:
        jobs[task_id] = {"status": "failed", "error": str(e)}

@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...), 
    exercise_name: str = Form("Balance")
):
    task_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{task_id}_{video.filename}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
        
    jobs[task_id] = {"status": "queued"}
    background_tasks.add_task(analyze_in_background, task_id, file_path, exercise_name)
    
    return {"task_id": task_id, "message": "Video uploaded successfully. Processing in background."}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    if task_id in jobs:
        return jobs[task_id]
    return {"status": "not_found"}

@app.websocket("/ws/form-check/{exercise}")
async def websocket_form_check(websocket: WebSocket, exercise: str):
    await websocket.accept()
    
    fsm = StateMachineFSM(exercise)
    
    try:
        while True:
            # Receive base64 encoded frame
            data = await websocket.receive_text()
            
            # Decode Base64 to cv2 image
            try:
                encoded_data = data.split(',')[1] if ',' in data else data
                nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                    
                # Run lightweight YOLO detection
                keypoints_2d_raw = get_yolo26_keypoints(frame)
                keypoints = [(pt[0], pt[1]) if pt else None for pt in keypoints_2d_raw]

                status, message, current_state = fsm.process_frame(keypoints)
                
                await websocket.send_json({
                    "status": status,
                    "state": current_state,
                    "hint": message
                })
                
            except Exception as cv_e:
                import traceback
                print("CV Decoding/Processing Error:")
                traceback.print_exc()
                continue
                
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)