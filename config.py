import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# API Keys and Endpoints
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Collection settings
DEFAULT_COLLECTION = os.getenv("DEFAULT_COLLECTION", "qa_collection2")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "512"))

#OPENAI MODEL
LLM_MODEL = "gpt-4o-2024-08-06"

# Embedding model settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Search settings
DEFAULT_SEARCH_LIMIT = int(os.getenv("DEFAULT_SEARCH_LIMIT", "5"))
HYBRID_SEARCH_WEIGHT = float(os.getenv("HYBRID_SEARCH_WEIGHT", "0.3"))

# Document processing settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# UI Settings
APP_TITLE = os.getenv("APP_TITLE", "RFP Q&A System")
APP_SUBTITLE = os.getenv("APP_SUBTITLE", "Ask questions about your RFP documents")

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),  # Console handler
        logging.FileHandler("rfp_qa.log")  # File handler
    ]
)