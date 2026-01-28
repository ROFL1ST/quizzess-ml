from pydantic import BaseModel
from typing import List, Optional

# --- Legacy: Adaptive Quiz ---
class UserHistoryItem(BaseModel):
    question_id: int
    is_correct: bool
    difficulty: float

class RecommendationRequest(BaseModel):
    user_id: int
    quiz_id: int
    history: List[UserHistoryItem]
    prior_ability: Optional[float] = 0.5 

# --- New: Essay Grading ---
class EssayGradingRequest(BaseModel):
    question_text: str
    teacher_key: str
    student_answer: str
