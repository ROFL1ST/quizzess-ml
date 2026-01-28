import logging
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class GradingService:
    def __init__(self):
        self.sbert_model = None
        self.tokenizer = None
        self.bert_model = None
        self._load_model()

    def _load_model(self):
        # Load SBERT
        try:
            logger.info("⏳ Loading SBERT Model...")
            self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ SBERT Model Loaded Successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to load SBERT: {e}")
            self.sbert_model = None
        
        # Load IndoBERT
        try:
            logger.info("⏳ Loading IndoBERT Model...")
            model_name = "indobenchmark/indobert-base-p1"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModel.from_pretrained(model_name)
            logger.info("✅ IndoBERT Model Loaded Successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to load IndoBERT: {e}")
            self.tokenizer = None
            self.bert_model = None

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

        # Algorithm C: IndoBERT (Comparison)
        score_indobert = 0.0
        if self.bert_model and self.tokenizer:
            try:
                # Tokenize
                inputs = self.tokenizer([key, ans], padding=True, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                
                # Mean Pooling
                token_embeddings = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']
                
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
                
                # Cosine Sim
                cosine_sim_bert = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0][0]
                score_indobert = float(cosine_sim_bert) * 100.0
            except Exception as e:
                logger.error(f"IndoBERT Error: {e}")

        # Decision Logic (Still using SBERT as primary for now, or weighted?)
        # User asked for comparison, so we keep SBERT as primary determinant for feedback 
        # but we can return indobert score in debug.
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
                "score_indobert": round(score_indobert, 2),
                "method_used": "SBERT + IndoBERT(Debug)"
            }
        }

grading_service = GradingService()
