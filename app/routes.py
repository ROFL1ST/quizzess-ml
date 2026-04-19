from fastapi import APIRouter, Security, HTTPException, status
from fastapi.security import APIKeyHeader
from app.models.schemas import EssayGradingRequest, RecommendationRequest
from app.services.grading import grading_service
from app.services.recommender import recommender_service
from app.core.config import settings

router = APIRouter()

api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=True)

def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == settings.API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )

@router.get("/")
def health_check():
    return {
        "status": "running", 
        "version": "2.2.0", 
        "sbert_loaded": grading_service.sbert_model is not None
    }

# ---------------------------------------------------------
# FEATURE 1: AUTOMATED ESSAY GRADING (RESEARCH CORE)
# ---------------------------------------------------------
@router.post("/grade-essay", dependencies=[Security(get_api_key)])
def grade_essay(req: EssayGradingRequest):
    return grading_service.grade(req.teacher_key, req.student_answer)

# ---------------------------------------------------------
# FEATURE 2: ADAPTIVE QUIZ RECOMMENDATION (LEGACY)
# ---------------------------------------------------------
@router.post("/recommend", dependencies=[Security(get_api_key)])
def recommend_question(data: RecommendationRequest):
    return recommender_service.recommend(data)
