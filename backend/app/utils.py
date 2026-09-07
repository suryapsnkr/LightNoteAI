import os
import uuid
import aiofiles
from fastapi import UploadFile
from app.config import settings

async def save_upload_file(file: UploadFile, subdir: str) -> str:
    os.makedirs(os.path.join(settings.UPLOAD_DIR, subdir), exist_ok=True)
    ext = file.filename.split(".")[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    filepath = os.path.join(settings.UPLOAD_DIR, subdir, filename)
    
    async with aiofiles.open(filepath, "wb") as out:
        content = await file.read()
        await out.write(content)
    
    return filepath