from pydantic import BaseModel
from typing import List, Optional

class UserHistoryItem(BaseModel):
    question_id: int
    is_correct: bool
    difficulty: float # 0.0 to 1.0

class RecommendationRequest(BaseModel):
    user_id: int
    quiz_id: int
    history: List[UserHistoryItem]
    prior_ability: Optional[float] = 0.5 # Default to medium if not provided

class RecommendationResponse(BaseModel):
    target_difficulty: float
    predicted_ability: float
    message: str
    mode: str # "heuristic" or "ml_logistic"
