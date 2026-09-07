import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    UPLOAD_DIR = "storage/uploads"
    OUTPUT_DIR = "storage/outputs"
    MAX_VIDEO_SIZE = 50 * 1024 * 1024  # 50 MB

settings = Settings()