from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from app.config import settings
from app.utils import save_upload_file
from tasks.video_task import process_video_task
from tasks.celery_app import celery_app
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_video(
    video: UploadFile = File(...),
    prompt: str = Form(...),
    reference: UploadFile = File(None)
):
    # Validate file type
    if not video.content_type.startswith("video/"):
        raise HTTPException(400, "Invalid video file")
    
    video_path = await save_upload_file(video, "videos")
    ref_path = None
    if reference:
        ref_path = await save_upload_file(reference, "references")
    
    # Start Celery task
    task = process_video_task.delay(video_path, prompt, ref_path)
    
    return {"task_id": task.id, "status": "queued"}

@app.get("/status/{task_id}")
def get_status(task_id: str):
    result = celery_app.AsyncResult(task_id)
    
    if result.failed():
        return {"status": "FAILURE", "error": str(result.info)}
    
    if result.ready():
        output_url = result.result.get("output_url")
        return {"status": "SUCCESS", "result_url": output_url}
    
    # PENDING or STARTED
    progress = result.info.get("progress", 0) if result.info else 0
    return {"status": result.state, "progress": progress}

@app.get("/output/{filename}")
def serve_output(filename: str):
    filepath = os.path.join(settings.OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(404, "Output not found")
    return FileResponse(filepath, media_type="video/mp4")