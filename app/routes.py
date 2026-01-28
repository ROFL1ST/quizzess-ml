from fastapi import APIRouter
from app.models.schemas import EssayGradingRequest, RecommendationRequest
from app.services.grading import grading_service
from app.services.recommender import recommender_service

router = APIRouter()

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
@router.post("/grade-essay")
def grade_essay(req: EssayGradingRequest):
    return grading_service.grade(req.teacher_key, req.student_answer)

# ---------------------------------------------------------
# FEATURE 2: ADAPTIVE QUIZ RECOMMENDATION (LEGACY)
# ---------------------------------------------------------
@router.post("/recommend")
def recommend_question(data: RecommendationRequest):
    return recommender_service.recommend(data)
