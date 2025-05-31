# core/config.py
import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import logging
from typing import List

# Load variables from .env file located in the project root directory
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
else:
    load_dotenv() # Fallback

class Settings(BaseSettings):
    """Loads configuration settings from environment variables and .env file."""

    GEMINI_API_KEY: str = "YOUR_GEMINI_API_KEY_HERE"

    # --- Supabase Configuration ---
    SUPABASE_URL: str | None = None
    SUPABASE_KEY: str | None = None # ANON key usually
    SUPABASE_SERVICE_KEY: str | None = None # SERVICE_ROLE key

    # --- Neo4j Configuration --- ### <--- ADD THIS ---
    NEO4J_URI: str = "neo4j://localhost:7687" # Default local URI
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "your_neo4j_password" # Change this!

    # --- Service URLs ---
    API_GATEWAY_URL: str = "http://localhost:8000"
    PAPER_FETCHER_URL: str = "http://localhost:8001"
    DOC_PROCESSOR_URL: str = "http://localhost:8002"
    ANALYSIS_ENGINE_URL: str = "http://localhost:8003"
    VECTOR_SERVICE_URL: str = "http://localhost:8004"
    KG_SERVICE_URL: str = os.getenv("KG_SERVICE_URL", "http://localhost:8005") # New KG service URL
    UI_SERVICE_URL: str = "http://localhost:7860"

    # --- Storage Configuration ---
    DOC_STORAGE_BUCKET: str = "irn-documents"

    # --- Model Names & Embedding Dimension ---
    GEMINI_PRO_MODEL: str = "gemini-1.5-pro-latest"
    GEMINI_FLASH_MODEL: str = "gemini-1.5-flash-latest"
    GEMINI_EMBEDDING_MODEL: str = "models/text-embedding-004"
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", 768))

    # --- Document Processing Config ---
    PARSER_CHUNK_SIZE: int = int(os.getenv("PARSER_CHUNK_SIZE", 1500))
    PARSER_CHUNK_OVERLAP: int = int(os.getenv("PARSER_CHUNK_OVERLAP", 150))
    PARSER_SEPARATORS: List[str] = ["\n\n", "\n", ". ", ", ", " ", ""]

    # --- Vector Search Configuration ---
    SEARCH_MATCH_THRESHOLD: float = float(os.getenv("SEARCH_MATCH_THRESHOLD", 0.7))

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        extra = 'ignore'

# Instantiate settings once for import
settings = Settings()

# --- Logging Setup (Keep as is) ---
# ... (existing logging setup) ...
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper(); log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("IRN_Core")
logging.getLogger("httpx").setLevel(logging.WARNING); logging.getLogger("supabase").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING); logging.getLogger("neo4j").setLevel(logging.WARNING) # Add neo4j logger suppression

# --- Configuration Validation Checks (Add Neo4j Check) ---
logger.info(f"Core Settings loaded. Log Level: {log_level_str}")
if not settings.SUPABASE_URL or not settings.SUPABASE_KEY: logger.warning("Supabase URL/Key missing.")
if not settings.SUPABASE_SERVICE_KEY: logger.warning("Supabase Service Key missing.")
if not settings.GEMINI_API_KEY or settings.GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE": logger.warning("GEMINI_API_KEY missing.")
if not settings.DOC_STORAGE_BUCKET: logger.warning("DOC_STORAGE_BUCKET missing, using default.")
else: logger.info(f"Using Supabase Storage Bucket: {settings.DOC_STORAGE_BUCKET}")
# --- Add Neo4j Check ---
if not settings.NEO4J_URI or not settings.NEO4J_USER or not settings.NEO4J_PASSWORD or settings.NEO4J_PASSWORD == "your_neo4j_password":
    logger.warning("Neo4j connection details (URI, USER, PASSWORD) not fully configured. KG Service may fail.")
else:
     logger.info(f"Neo4j Configured: URI={settings.NEO4J_URI}, User={settings.NEO4J_USER}")

try: assert settings.EMBEDDING_DIM > 0; logger.info(f"Using Embedding Dimension: {settings.EMBEDDING_DIM}")
except (AssertionError, ValueError): logger.error(f"Invalid EMBEDDING_DIM: {settings.EMBEDDING_DIM}.")
logger.info(f"Vector Search Config: Match Threshold={settings.SEARCH_MATCH_THRESHOLD}")
logger.info(f"Document Processor Config: Chunk Size={settings.PARSER_CHUNK_SIZE}, Overlap={settings.PARSER_CHUNK_OVERLAP}")