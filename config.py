import os
from pathlib import Path
from dotenv import load_dotenv

# Explicitly load .env from the project folder and override empty OS variables
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)


class Config:
    """Centralized configuration for FinSight AI."""

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
    MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash").strip()
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "30000"))
    DATABASE_PATH = os.getenv("DATABASE_PATH", "finsight.db")
    MAX_UPLOAD_SIZE_MB = 10