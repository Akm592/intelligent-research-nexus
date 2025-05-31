# services/kg_service/app/models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# --- Input Models for Nodes ---

class KgPaperInput(BaseModel):
    """Data needed to create/update a Paper node."""
    paper_id: str = Field(..., description="Unique ID (e.g., arxiv:xxx, upload:xxx)")
    title: Optional[str] = None
    publication_date: Optional[str] = None # Store as string for simplicity, Neo4j can handle date/time types too
    source: Optional[str] = None
    url: Optional[str] = None # Abstract URL or main landing page URL
    pdf_url: Optional[str] = None # Direct PDF URL

    class Config:
   
        schema_extra = {
            "example": {
                "paper_id": "arxiv:2305.12345v1",
                "title": "Example Paper Title",
                "publication_date": "2023-05-20",
                "source": "arxiv",
                "url": "http://arxiv.org/abs/2305.12345v1",
                "pdf_url": "http://arxiv.org/pdf/2305.12345v1"
            }
        }

class KgAuthorInput(BaseModel):
    """Data needed to create/update an Author node."""
    name: str = Field(..., description="Author name (used as key - case-sensitive merge by default)")
    # Add other fields like ORCID later if needed

    class Config:
        schema_extra = {
            "example": {
                "name": "Jane Doe"
            }
        }

# --- Input Models for Relationships ---

class AuthoredByRequest(BaseModel):
    """Data needed to create an AUTHORED_BY relationship."""
    paper_id: str
    author_name: str

    class Config:
        schema_extra = {
            "example": {
                "paper_id": "arxiv:2305.12345v1",
                "author_name": "Yanan Jian"
            }
        }

# --- Standard Response Model ---

class KgResponse(BaseModel):
    """Standard response from KG service."""
    status: str = Field(description="'success' or 'error'")
    message: str
    details: Optional[Any] = None # For extra info or error details

    class Config:
        schema_extra = {
            "example_success": {
                "status": "success",
                "message": "Operation completed successfully."
            },
            "example_error": {
                "status": "error",
                "message": "Operation failed.",
                "details": "Specific error message here."
            }
        }