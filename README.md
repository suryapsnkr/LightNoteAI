# AI Video Editor

## Architecture
- Frontend: React (or HTML+JS)
- Backend: FastAPI + Celery + Redis
- AI Models:
  - Prompt parsing: GPT‑4 (OpenAI)
  - Object detection: GroundingDINO
  - Segmentation: SAM
  - Inpainting: LaMa
  - Replacement: Stable Diffusion Inpainting
- Video processing: OpenCV, ffmpeg

## Why These Choices
- **GroundingDINO + SAM** provide strong zero‑shot object detection and segmentation with minimal fine‑tuning.
- **Stable Diffusion Inpainting** offers plausible replacements that blend well.
- **Celery** decouples the UI from heavy processing, improving user experience.

## Setup Instructions
1. Clone repo.
2. Install dependencies: `pip install -r requirements.txt`
3. Set up Redis (for Celery).
4. Start FastAPI: `uvicorn main:app --reload`
5. Start Celery worker: `celery -A tasks worker --loglevel=info`
6. Open `frontend/index.html`.

## Known Limitations
- Processing is slow; frame‑by‑frame generation is costly.
- Object tracking can fail if the object moves quickly or changes perspective.
- Replacement may not perfectly match lighting/pose.
- Only supports one operation per video.


┌─────────────┐      ┌─────────────┐      ┌─────────────────┐
│   Frontend  │ ───▶ │   Backend   │ ───▶ │  Task Queue     │
│  (React/    │      │  (FastAPI)  │      │  (Celery/Redis) │
│   HTML+JS)  │ ◀─── │             │ ◀─── │                 │
└─────────────┘      └─────────────┘      └────────┬────────┘
                                                    │
                                                    ▼
                                         ┌─────────────────────┐
                                         │  AI/Video Pipeline  │
                                         │  - Prompt parsing   │
                                         │  - Object detection │
                                         │  - Segmentation     │
                                         │  - Tracking         │
                                         │  - Removal          │
                                         │  - Replacement      │
                                         │  - Video assembly   │
                                         └─────────────────────┘