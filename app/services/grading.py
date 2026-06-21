import json
import logging
import re
import torch

from app.core.config import settings
from app.services.model_loader import get_model, get_tokenizer

logger = logging.getLogger(__name__)


ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""

INSTRUCTION = "Kamu adalah sistem penilaian esai otomatis. Nilai jawaban siswa berdasarkan kunci jawaban yang diberikan. Berikan skor 0-100 dan reasoning dalam Bahasa Indonesia."


class GradingService:
    def __init__(self):
        # Lazy model loading via model_loader
        self.tokenizer = None
        self.model = None
        self.sbert_model = None  # kept for health check compatibility

    def _ensure_model(self):
        if self.model is None or self.tokenizer is None:
            logger.info("Loading LLaMA grading model lazily via model_loader...")
            self.model = get_model()
            self.tokenizer = get_tokenizer()
            logger.info("LLaMA grading model loaded and cached.")

    def _parse_response(self, raw: str) -> dict:
        if "```" in raw:
            raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)
        result = json.loads(raw)
        return {
            "score": float(result.get("score", 0)),
            "reasoning": result.get("reasoning", "-")
        }

    def grade(self, teacher_key: str, student_answer: str) -> dict:
        key = teacher_key.strip()
        ans = student_answer.strip()

        if not key or not ans:
            return {"score_final": 0.0, "reasoning": "Jawaban atau kunci jawaban kosong."}

        # Ensure model and tokenizer are loaded lazily
        try:
            self._ensure_model()
        except Exception as e:
            logger.error(f"Failed to load model lazily: {e}")
            return {"score_final": 0.0, "reasoning": "Model belum tersedia. Periksa konfigurasi server."}

        input_text = f"Kunci Jawaban: {key}\nJawaban Siswa: {ans}"
        prompt = ALPACA_PROMPT.format(INSTRUCTION, input_text)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=settings.MAX_NEW_TOKENS,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            input_len = inputs["input_ids"].shape[1]
            generated = self.tokenizer.decode(
                outputs[0][input_len:], skip_special_tokens=True
            )
            result = self._parse_response(generated)
            return {
                "score_final": round(result["score"], 2),
                "reasoning": result["reasoning"]
            }
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {"score_final": 0.0, "reasoning": f"Error saat inferensi: {e}"}


grading_service = GradingService()
