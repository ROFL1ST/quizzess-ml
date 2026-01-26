from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression

app = FastAPI(title="Adaptive Quiz ML Service", version="2.0.0")

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

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "ml-service"}

@app.post("/recommend", response_model=RecommendationResponse)
def recommend_difficulty(request: RecommendationRequest):
    # Anchor / Prior
    anchor_ability = request.prior_ability if request.prior_ability is not None else 0.5
    
    # 1. COLD START / LOW DATA: Use Weighted Random Walk (Heuristic)
    if not request.history or len(request.history) < 5:
        target = calculate_heuristic(request.history, anchor_ability)
        return {
            "target_difficulty": round(target, 2),
            "predicted_ability": round(target, 2), # In heuristic, ability ~ current diff
            "message": "Collecting data... (Heuristic Mode)",
            "mode": "heuristic"
        }

    # 2. SUFFICIENT DATA: Use Scikit-Learn Logistic Regression
    try:
        # Prepare Data
        X = np.array([[item.difficulty] for item in request.history])
        y = np.array([1 if item.is_correct else 0 for item in request.history])
        
        # Check if all one class (all correct or all wrong), regression will fail/warn
        if len(np.unique(y)) < 2:
            target = calculate_heuristic(request.history, anchor_ability)
            return {
                "target_difficulty": round(target, 2),
                "predicted_ability": round(target, 2),
                "message": f"Data not diverse enough for ML (All {'Correct' if y[0] == 1 else 'Wrong'}). Using Heuristic.",
                "mode": "heuristic_fallback"
            }

        # Train Model
        # Predict Probability of Correctness (P(1)) given Difficulty (X)
        model = LogisticRegression()
        model.fit(X, y)
        
        # We want to find Difficulty D where P(Correct|D) ~= 0.70 (Optimal Challenge Zone)
        # Logistic Function: P = 1 / (1 + exp(-(m*x + c)))
        # ln(P / (1-P)) = m*x + c
        # x = (ln(P / (1-P)) - c) / m
        
        # Target Probability
        P_target = 0.70
        
        # Coefficients
        m = model.coef_[0][0]
        c = model.intercept_[0]
        
        if abs(m) < 0.001: # Avoid division by zero if flat slope
            target_diff = calculate_heuristic(request.history, anchor_ability)
        else:
            # Inverse Logit
            target_diff = (np.log(P_target / (1 - P_target)) - c) / m

        # Clamp result 0.0 - 1.0
        target_diff = max(0.1, min(1.0, target_diff))
        
        # Predict current ability (Diff where P=0.5)
        current_ability = -c / m if abs(m) > 0.001 else anchor_ability
        current_ability = max(0.0, min(1.0, current_ability))

        return {
            "target_difficulty": round(target_diff, 2),
            "predicted_ability": round(current_ability, 2),
            "message": "Optimized by Logistic Regression ML",
            "mode": "ml_logistic"
        }

    except Exception as e:
        print(f"ML Error: {e}")
        # Fallback
        target = calculate_heuristic(request.history, anchor_ability)
        return {
            "target_difficulty": round(target, 2),
            "predicted_ability": round(target, 2),
            "message": "ML failed, falling back to heuristic",
            "mode": "fallback_error"
        }

def calculate_heuristic(history: List[UserHistoryItem], anchor: float) -> float:
    if not history:
        return anchor
        
    # Simple Weighted Average with Pull
    current = anchor
    for item in history:
        step = 0.05
        if item.is_correct:
            # If current diff was easy and we passed, jump up
             current += step
        else:
             current -= step
    
    return max(0.1, min(1.0, current))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002)
