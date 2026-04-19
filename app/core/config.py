import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "Adaptive Quiz ML Service"
    VERSION: str = "2.2.0"
    API_V1_STR: str = "/api/v1"
    
    # Model Paths (Optional, for caching)
    SBERT_MODEL_NAME: str = 'all-MiniLM-L6-v2'

    # Security
    API_KEY: str = os.getenv("API_KEY", "default-secure-key")

settings = Settings()
