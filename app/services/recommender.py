from sklearn.linear_model import LogisticRegression
from app.models.schemas import RecommendationRequest

class RecommenderService:
    def __init__(self):
        # In production, load a trained model pickle here
        self.model = LogisticRegression()

    def recommend(self, data: RecommendationRequest) -> dict:
        # 1. Calculate Current Ability (Theta)
        # Simple logic: Ratio of correct answers + Weight of difficulty
        total_score = 0
        total_items = 0
        
        if not data.history:
            current_ability = data.prior_ability if data.prior_ability else 0.5
        else:
            for item in data.history:
                total_items += 1
                if item.is_correct:
                    total_score += (1.0 + item.difficulty) 
                else:
                    total_score -= (0.5 * (1.0 - item.difficulty))
            
            raw_score = total_score / (total_items * 1.5) if total_items > 0 else 0.5
            current_ability = 0.5 + (raw_score * 0.5) 
            current_ability = max(0.1, min(0.9, current_ability))

        # 2. Determine Next Difficulty Target
        target_difficulty = current_ability + 0.05 

        return {
            "user_id": data.user_id,
            "current_ability": round(current_ability, 3),
            "next_difficulty_target": round(target_difficulty, 3),
            "version": "v2-logic"
        }

recommender_service = RecommenderService()
