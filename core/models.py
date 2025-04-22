# core/models.py
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Dict, Any
import uuid
import datetime

from core.config import logger

# --- Utility Functions ---

def generate_chunk_id(paper_id: str, index: int) -> str:
    """Generates a reproducible, unique ID for a chunk within a paper."""
    # Format index with leading zeros for consistent sorting if needed
    return f"{paper_id}_chunk_{index:04d}"

# --- Core Data Models ---

class PaperMetadata(BaseModel):
    """Represents the metadata associated with an academic paper or document."""
    # Using default_factory for fields that should be auto-generated if not provided
    id: str = Field(..., description="Unique identifier for the paper (e.g., arXiv ID, DOI, generated ID for uploads)")
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    abstract: Optional[str] = None
    publication_date: Optional[str] = None # Consider using datetime.date or str formatted consistently (YYYY-MM-DD)
    source: Optional[str] = Field(None, description="Origin of the paper (e.g., arXiv, Semantic Scholar, PubMed, Upload)")
    url: Optional[str] = Field(None, description="Primary URL for the paper (e.g., arXiv page, DOI link)")
    keywords: List[str] = Field(default_factory=list)
    processing_status: str = Field(default="pending", description="Status: pending, processing, processed, processed_with_errors, failed")
    # Database timestamps often added automatically, but can be included for clarity
    created_at: Optional[datetime.datetime] = None
    updated_at: Optional[datetime.datetime] = None

    class Config:
        # Example configuration if needed, e.g., for ORM mode
        from_attributes = True # Renamed from orm_mode in Pydantic v2

class DocumentChunk(BaseModel):
    """Represents a processed chunk of text from a document."""
    chunk_id: str = Field(..., description="Unique ID for this chunk (e.g., paper_id_chunk_0001)")
    paper_id: str = Field(..., description="ID of the paper this chunk belongs to")
    text: str = Field(..., description="Text content of the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata (e.g., section, page, source info)")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the text")

    class Config:
        from_attributes = True


# --- Service Request/Response Models ---

# Paper Fetcher Service
class FetchRequest(BaseModel):
    """Request to fetch papers from external sources."""
    query: str
    max_results: int = Field(default=10, gt=0, le=50) # Add validation
    sources: List[str] = ["arxiv", "semantic_scholar"] # Default sources

# Document Processor Service
class ProcessRequest(BaseModel):
    """Request to process a specific document."""
    paper_id: str
    # Option 1: Document is already available via URL
    source_url: Optional[str] = None
    # Option 2: Document is in Supabase Storage
    bucket_name: Optional[str] = None
    object_name: Optional[str] = None # Path within the bucket (e.g., uploads/paper_id/file.pdf)

    # Validator to ensure at least one source location is provided
    @model_validator(mode='before')
    @classmethod
    def check_source_provided(cls, values):
        if not values.get('source_url') and not values.get('object_name'):
            # Check bucket_name only if object_name is provided
             raise ValueError("Either 'source_url' or 'object_name' (with optional 'bucket_name') must be provided for processing.")
        return values

# Vector Service
class EmbedRequest(BaseModel):
    """Request to embed a list of document chunks."""
    # Pass list of chunk dicts that can be validated into DocumentChunk
    chunks: List[Dict[str, Any]]

class EmbedResponse(BaseModel):
    """Response confirming which chunks were processed for embedding."""
    # Return list of chunk dicts that were processed (or just IDs/status)
    processed_chunk_ids: List[str]
    failed_chunk_ids: List[str] = Field(default_factory=list)

class SearchQuery(BaseModel):
    """Request for semantic search."""
    query_text: str
    top_k: int = Field(default=5, gt=0, le=100)
    # Filters applied during vector search (specific structure depends on backend implementation)
    # Example: filter by paper_id {"paper_id": "arxiv:xxxx"}
    filters: Optional[Dict[str, Any]] = None

class SearchResultItem(BaseModel):
    """Represents a single item returned from a search."""
    paper_id: str
    chunk_id: str
    score: float = Field(description="Similarity score (higher is better, typically 0-1 for cosine similarity)")
    text: str = Field(description="The text content of the matched chunk (often retrieved from metadata)")
    metadata: Dict[str, Any] = Field(description="Metadata associated with the chunk")

class SearchResponse(BaseModel):
    """Response containing a list of search results."""
    results: List[SearchResultItem]

# Analysis Engine Service
class AnalysisRequest(BaseModel):
    """Request to perform analysis on documents."""
    query: Optional[str] = Field(None, description="User's specific question for QA or context")
    paper_ids: Optional[List[str]] = Field(None, description="List of paper IDs to focus analysis on")
    analysis_type: str = Field(default="summary", description="Type of analysis (e.g., 'summary', 'compare_methods', 'qa', 'gap_analysis')")
    detail_level: str = Field(default="medium", description="Desired detail (e.g., 'short', 'medium', 'detailed')")

    @model_validator(mode='before')
    @classmethod
    def check_input_provided(cls, values):
        if not values.get('paper_ids') and not values.get('query'):
            raise ValueError("Either 'paper_ids' or a 'query' must be provided for analysis.")
        return values

class AnalysisResult(BaseModel):
    """Response containing the generated analysis."""
    result_text: str = Field(description="The main text body of the generated analysis")
    # List of chunk_ids or potentially paper_ids cited in the analysis
    cited_sources: List[str] = Field(default_factory=list, description="List of source chunk IDs cited in the result text")
    analysis_type: str = Field(description="The type of analysis that was performed")


# --- API Gateway Request/Response Models (Simplified mirroring for clarity) ---

class GatewayFetchRequest(BaseModel):
    """Simplified fetch request for the Gateway."""
    query: str
    max_results: int = Field(default=10, gt=0, le=50)

class GatewayProcessRequest(BaseModel):
    """Simplified process request for the Gateway, including storage info."""
    paper_id: str
    # Allow specifying source via gateway
    bucket_name: Optional[str] = None
    object_name: Optional[str] = None
    source_url: Optional[str] = None

    # Validator to ensure at least one source is hinted, although backend service does final check
    @model_validator(mode='before')
    @classmethod
    def check_source_hint(cls, values):
         if not values.get('source_url') and not values.get('object_name'):
             # This isn't strictly necessary if the backend validates, but good practice
             logger.warning("Gateway received process request without source_url or object_name hint.")
             pass # Allow backend to handle missing source definitively
         return values


class GatewaySearchRequest(BaseModel):
    """Simplified search request for the Gateway."""
    query: str
    top_k: int = Field(default=5, gt=0, le=100)
    # Filters could be added here if needed by the UI/client directly
    # filters: Optional[Dict[str, Any]] = None

class GatewayAnalysisRequest(BaseModel):
    """Simplified analysis request for the Gateway."""
    query: Optional[str] = None
    paper_ids: Optional[List[str]] = None
    analysis_type: str = "summary"
    # detail_level could be added here if needed from client

class GatewayResponse(BaseModel):
    """Standard response wrapper for the API Gateway."""
    status: str = Field(description="'success' or 'error'")
    data: Any | None = Field(default=None, description="The primary data payload (depends on the endpoint)")
    message: Optional[str] = Field(default=None, description="Optional status message or error details")