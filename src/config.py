import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    API_KEY = os.getenv("GEMINI_API_KEY")

    # Models
    LLM_MODEL = "gemma-3-27b-it"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # RAG Settings
    CHUNK_SIZE = 700
    RETRIEVAL_K = 5  # Number of chunks to retrieve

    # Clinical Keywords (Keep content only if it contains these)
    KEYWORDS = [
        "mmhg", "140/90", "130/80", "≥", "<=", ">=", ">",
        "initiat", "start", "pharmacological", "drug", "therapy",
        "threshold", "stage", "grade", "contraindication", "dose"
    ]

    # Safety Guardrails
    UNSAFE_PATTERNS = [
        r"\bbest\b", r"\bsafest\b", r"\bmost effective\b",
        r"\bpreferred drug\b", r"\bwhich drug\b"
    ]

    if not API_KEY:
        raise ValueError("GEMINI_API_KEY is missing from .env file")