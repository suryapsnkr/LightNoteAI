from pydantic import BaseModel
from typing import Optional

class ParsePromptResponse(BaseModel):
    operation: str       # "replace", "remove", "change_text"
    target: str          # "Coca-Cola bottle"
    replacement: Optional[str] = None  # "Pepsi"

class TaskStatus(BaseModel):
    task_id: str
    status: str          # PENDING, STARTED, SUCCESS, FAILURE
    progress: int = 0
    result_url: Optional[str] = None
    error: Optional[str] = None