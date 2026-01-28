import logging
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class GradingService:
    def __init__(self):
        self.sbert_model = None
        self._load_model()

    def _load_model(self):
        try:
            logger.info("⏳ Loading SBERT Model...")
            self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ SBERT Model Loaded Successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to load SBERT: {e}")
            self.sbert_model = None

    def grade(self, teacher_key: str, student_answer: str) -> dict:
        key = teacher_key.strip()
        ans = student_answer.strip()

        if not key or not ans:
            return {
                "score_final": 0.0,
                "feedback": "Empty answer provided",
                "debug": {"tfidf": 0, "sbert": 0}
            }

        # Algorithm A: TF-IDF
        score_tfidf = 0.0
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([key, ans])
            cosine_sim_tfidf = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            score_tfidf = float(cosine_sim_tfidf) * 100.0
        except Exception as e:
            logger.error(f"TF-IDF Error: {e}")

        # Algorithm B: SBERT
        score_sbert = 0.0
        if self.sbert_model:
            try:
                embeddings = self.sbert_model.encode([key, ans], convert_to_tensor=True)
                cosine_sim_sbert = util.cos_sim(embeddings[0], embeddings[1])
                score_sbert = float(cosine_sim_sbert.item()) * 100.0
            except Exception as e:
                logger.error(f"SBERT Error: {e}")

        # Decision Logic
        final_score = score_sbert
        
        feedback = "Jawaban cukup baik."
        if final_score >= 90:
            feedback = "Sempurna! Pemahaman sangat mendalam."
        elif final_score >= 75:
            feedback = "Bagus, namun ada sedikit perbedaan istilah."
        elif final_score >= 50:
            feedback = "Kurang tepat, perbaiki lagi konsepnya."
        else:
            feedback = "Jawaban tidak relevan dengan kunci jawaban."

        return {
            "score_final": round(final_score, 2),
            "feedback": feedback,
            "debug": {
                "score_tfidf": round(score_tfidf, 2),
                "score_sbert": round(score_sbert, 2),
                "method_used": "SBERT-all-MiniLM-L6-v2"
            }
        }

grading_service = GradingService()
