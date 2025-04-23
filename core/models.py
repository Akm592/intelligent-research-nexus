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
    id: str = Field(..., description="Unique identifier for the paper (e.g., arXiv ID, DOI, generated ID for uploads)")
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    abstract: Optional[str] = None
    publication_date: Optional[str] = None # Consider using datetime.date or str formatted consistently (YYYY-MM-DD)
    source: Optional[str] = Field(None, description="Origin of the paper (e.g., arXiv, Semantic Scholar, PubMed, Upload)")
    url: Optional[str] = Field(None, description="Primary URL for the paper (e.g., abstract page, DOI link)")
    # --- ADDED FIELD for direct PDF download ---
    pdf_url: Optional[str] = Field(None, description="Direct URL to the PDF file for downloading")
    # ------------------------------------------
    keywords: List[str] = Field(default_factory=list)
    processing_status: str = Field(default="pending", description="Status: pending, processing, processed, processed_with_errors, failed")
    status_message: Optional[str] = Field(None, description="Optional message providing details about the processing status") # Added based on previous fixes
    created_at: Optional[datetime.datetime] = None
    updated_at: Optional[datetime.datetime] = None

    class Config:
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
    """Request to process a specific document. Sent from Gateway to Doc Processor."""
    paper_id: str
    # Option 1: Document URL provided explicitly by the caller (preferred if known)
    source_url: Optional[str] = None
    # Option 2: Document is in Supabase Storage (from upload)
    bucket_name: Optional[str] = None
    object_name: Optional[str] = None # Path within the bucket

    # --- REMOVED VALIDATOR ---
    # The Document Processor's get_document_content function MUST now handle
    # the logic of finding the source if source_url or object_name are None
    # by looking up the paper_id in the database.
    pass

# Vector Service
class EmbedRequest(BaseModel):
    """Request to embed a list of document chunks."""
    chunks: List[Dict[str, Any]] # Expect list of dicts that can validate to DocumentChunk

class EmbedResponse(BaseModel):
    """Response confirming which chunks were processed for embedding."""
    processed_chunk_ids: List[str]
    failed_chunk_ids: List[str] = Field(default_factory=list)

class SearchQuery(BaseModel):
    """Request for semantic search."""
    query_text: str
    top_k: int = Field(default=5, gt=0, le=100)
    filters: Optional[Dict[str, Any]] = None # Example: {"paper_id": "arxiv:xxxx"}

class SearchResultItem(BaseModel):
    """Represents a single item returned from a search."""
    paper_id: str
    chunk_id: str
    score: float = Field(description="Similarity score")
    text: str = Field(description="The text content of the matched chunk")
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
    cited_sources: List[str] = Field(default_factory=list, description="List of source chunk IDs cited in the result text")
    analysis_type: str = Field(description="The type of analysis that was performed")


# --- API Gateway Request/Response Models (Representing Client Input) ---

class GatewayFetchRequest(BaseModel):
    """Simplified fetch request for the Gateway."""
    query: str
    max_results: int = Field(default=10, gt=0, le=50)

class GatewayProcessRequest(BaseModel):
    """Simplified process request for the Gateway, including storage info."""
    paper_id: str
    # Allow specifying source via gateway (UI might know PDF URL now)
    bucket_name: Optional[str] = None
    object_name: Optional[str] = None
    source_url: Optional[str] = None # UI should provide PDF URL here if known

class GatewaySearchRequest(BaseModel):
    """Simplified search request for the Gateway."""
    query: str
    top_k: int = Field(default=5, gt=0, le=100)
    # filters: Optional[Dict[str, Any]] = None

class GatewayAnalysisRequest(BaseModel):
    """Simplified analysis request for the Gateway."""
    query: Optional[str] = None
    paper_ids: Optional[List[str]] = None
    analysis_type: str = "summary"
    # detail_level: Optional[str] = None

class GatewayResponse(BaseModel):
    """Standard response wrapper for the API Gateway."""
    status: str = Field(description="'success' or 'error'")
    data: Any | None = Field(default=None, description="The primary data payload (depends on the endpoint)")
    message: Optional[str] = Field(default=None, description="Optional status message or error details")