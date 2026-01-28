from fastapi import FastAPI
from app.routes import router as api_router
from app.core.config import settings
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

# Include Routes
app.include_router(api_router)
