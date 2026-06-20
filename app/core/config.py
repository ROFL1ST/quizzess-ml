import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "Adaptive Quiz ML Service"
    VERSION: str = "3.1.0"
    API_V1_STR: str = "/api/v1"

    # Base LLaMA model (from Meta via HuggingFace)
    LLAMA_BASE_MODEL_ID: str = os.getenv("LLAMA_BASE_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")

    # LoRA adapter (fine-tuned)
    LLAMA_MODEL_ID: str = os.getenv("LLAMA_MODEL_ID", "rofl1st/llama-3.1-8b-essay-grader")

    # HuggingFace token (required for gated base model)
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")

    # Inference config
    MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "256"))

    # Security
    API_KEY: str = os.getenv("API_KEY", "default-secure-key")

settings = Settings()
