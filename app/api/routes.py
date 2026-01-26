from fastapi import APIRouter
from app.models import RecommendationRequest, RecommendationResponse
from app.services.recommendation import predict_next_difficulty

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok", "service": "ml-service"}

@router.post("/recommend", response_model=RecommendationResponse)
def recommend_difficulty(request: RecommendationRequest):
    # Anchor / Prior
    anchor_ability = request.prior_ability if request.prior_ability is not None else 0.5
    
    return predict_next_difficulty(request.history, anchor_ability)
