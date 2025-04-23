# core/config.py
import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import logging 
from typing import List

# Load variables from .env file located in the project root directory
# Ensure load_dotenv() is called before Settings() instance is created
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
else:
    # Fallback if running from a different working directory structure, though less ideal
    load_dotenv()

class Settings(BaseSettings):
    """Loads configuration settings from environment variables and .env file."""

    GEMINI_API_KEY: str = "YOUR_GEMINI_API_KEY_HERE" # Default if not set

    # --- Supabase Configuration ---
    SUPABASE_URL: str | None = None
    SUPABASE_KEY: str | None = None # SERVICE_ROLE or ANON key
    SUPABASE_SERVICE_KEY: str | None = None # SERVICE_ROLE key for admin operations

    # --- Service URLs (defaults assume running locally or via docker-compose) ---
    API_GATEWAY_URL: str = "http://localhost:8000" # Added Gateway URL for UI service
    PAPER_FETCHER_URL: str = "http://localhost:8001"
    DOC_PROCESSOR_URL: str = "http://localhost:8002"
    ANALYSIS_ENGINE_URL: str = "http://localhost:8003"
    VECTOR_SERVICE_URL: str = "http://localhost:8004"
    KG_SERVICE_URL: str = "http://localhost:8005" # Keep if separate
    # USER_PROFILE_URL: str = "http://localhost:8006" # Keep if separate
    UI_SERVICE_URL: str = "http://localhost:7860" # Optional: URL for UI service itself

    # --- Storage Configuration ---
    # Bucket name in Supabase Storage
    DOC_STORAGE_BUCKET: str = "irn-documents" # Default if not set

    # --- Model Names & Embedding Dimension ---
    GEMINI_PRO_MODEL: str = "gemini-1.5-pro-latest"
    GEMINI_FLASH_MODEL: str = "gemini-1.5-flash-latest"
    GEMINI_EMBEDDING_MODEL: str = "models/text-embedding-004"
    EMBEDDING_DIM: int = 768 # Default to 768 if not set
    PARSER_CHUNK_SIZE: int = int(os.getenv("PARSER_CHUNK_SIZE", 1500)) # Target characters per chunk
    PARSER_CHUNK_OVERLAP: int = int(os.getenv("PARSER_CHUNK_OVERLAP", 150)) # Overlap between chunks
    # Separators for recursive text splitting (order matters: tries first ones first)
    PARSER_SEPARATORS: List[str] = ["\n\n", "\n", ". ", ", ", " ", ""] # Common separators
    
    
     # --- Vector Search Configuration --- ### <--- ADD THIS SECTION ###
    # Default threshold for cosine similarity (1 - cosine_distance). Higher = more similar.
    # Adjust based on embedding model and desired relevance. 0.7-0.8 is common.
    SEARCH_MATCH_THRESHOLD: float = float(os.getenv("SEARCH_MATCH_THRESHOLD", 0.7))

    # Load settings from .env file if present
    # BaseSettings automatically reads environment variables.
    # The `env_file` setting tells it to also load from a .env file.
    class Config:
        # If using load_dotenv() explicitly above, this might be redundant, but safe to keep
        # Relative path assumes BaseSettings is instantiated where .env is discoverable
        # Explicit loading via load_dotenv(dotenv_path=...) is more robust
        env_file = '.env'
        env_file_encoding = 'utf-8'
        extra = 'ignore' # Ignore extra fields not defined in the model

# Instantiate settings once for import
settings = Settings()

# Basic logging setup (configure once)
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Get root logger for the application
logger = logging.getLogger("IRN_Core")

# Suppress excessive logging from libraries if needed
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("supabase").setLevel(logging.WARNING)
# Adjust google.generativeai logging if it becomes too verbose
# logging.getLogger("google.generativeai").setLevel(logging.WARNING)

logger.info(f"Core Settings loaded. Log Level: {log_level_str}")

# --- Configuration Validation Checks ---
if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
     logger.warning("Supabase URL or Key not configured in environment/.env. Database operations will fail.")
if not settings.GEMINI_API_KEY or settings.GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
     logger.warning("GEMINI_API_KEY not configured in environment/.env. Gemini operations will fail.")
if not settings.DOC_STORAGE_BUCKET:
     logger.warning("DOC_STORAGE_BUCKET is not set in environment/.env, using default 'irn-documents'. Ensure this bucket exists in Supabase Storage.")
else:
      logger.info(f"Using Supabase Storage Bucket: {settings.DOC_STORAGE_BUCKET}")

# Ensure embedding dimension matches expectations if needed elsewhere directly
try:
    assert settings.EMBEDDING_DIM > 0
    logger.info(f"Using Embedding Dimension: {settings.EMBEDDING_DIM}")
except (AssertionError, ValueError):
    logger.error(f"Invalid EMBEDDING_DIM configured: {settings.EMBEDDING_DIM}. Must be a positive integer.")
    # Potentially raise an error here if it's critical for startup
    # raise ValueError("Invalid EMBEDDING_DIM configured.")
    
    
logger.info("Core Settings loaded.")
# ... (warning checks) ...
logger.info(f"Document Processor Config: Chunk Size={settings.PARSER_CHUNK_SIZE}, Overlap={settings.PARSER_CHUNK_OVERLAP}")