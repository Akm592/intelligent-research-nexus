# services/vector_service/app/main.py

from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError # For catching Pydantic errors
from core.models import EmbedRequest, EmbedResponse, SearchQuery, SearchResponse, DocumentChunk
from core.config import settings, logger as core_logger # Use core logger
from core.gemini_client import gemini_client # Use the shared client
from . import vector_ops # Import local vector_ops module
import logging
import asyncio
from typing import List, Dict, Any, Optional

from core.supabase_client import get_supabase_client

# Use a child logger specific to this module
logger = core_logger.getChild("VectorService").getChild("Main")

# --- FastAPI App Setup ---

# Optional: Add lifespan events if needed for this service (e.g., DB connection pool)
# For Supabase client managed via core/supabase_client.py, explicit lifespan might not be needed here.

app = FastAPI(
    title="Vector Database Service (Supabase/pgvector)",
    description="Handles embedding generation via Gemini and storage/search in Supabase.",
    version="1.0.0"
)

# --- API Endpoints ---

@app.post("/embed", response_model=EmbedResponse)
async def embed_and_store_chunks(request: EmbedRequest):
    """
    Receives chunk data, generates embeddings using Gemini, and upserts
    the chunks with embeddings into the Supabase database via vector_ops.
    """
    logger.info(f"Received /embed request with {len(request.chunks)} chunk dictionaries.")
    
    if not request.chunks:
        logger.warning("Embed request received with zero chunks.")
        return EmbedResponse(processed_chunk_ids=[], failed_chunk_ids=[])
    
    # --- Input Validation ---
    chunks_to_process: List[DocumentChunk] = []
    all_incoming_ids = [c.get('chunk_id', f'unknown_{i}') for i, c in enumerate(request.chunks)] # Get all IDs for potential failure reporting
    
    try:
        # Validate incoming dictionaries into DocumentChunk objects
        chunks_to_process = [DocumentChunk(**chunk_dict) for chunk_dict in request.chunks]
        # Log paper ID for context if available
        paper_id_context = chunks_to_process[0].paper_id if chunks_to_process else "N/A"
        job_prefix = f"[{paper_id_context}]"
        logger.info(f"{job_prefix} Successfully validated {len(chunks_to_process)} input chunks.")
    except ValidationError as e:
        logger.error(f"Failed to validate input chunk data: {e}", exc_info=False) # Don't need full trace for validation error
        # Return 400 Bad Request if input format is wrong
        raise HTTPException(status_code=400, detail=f"Invalid chunk data format: {e}")
    except Exception as e: # Catch other potential errors during instantiation
        logger.error(f"Unexpected error processing input chunks: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Error processing input chunk data.")
        
    if not chunks_to_process: # Should not happen if validation passed, but double-check
        logger.error(f"{job_prefix} Validation passed but resulted in zero chunks to process.")
        return EmbedResponse(processed_chunk_ids=[], failed_chunk_ids=all_incoming_ids)
    
    # --- Step 1: Generate Embeddings ---
    logger.info(f"{job_prefix} Requesting embedding generation for {len(chunks_to_process)} chunks...")
    
    embedded_chunks: Optional[List[DocumentChunk]] = None
    usage_metadata: Dict[str, Any] = {}
    
    try:
        # Call the shared Gemini client
        embedded_chunks, usage_metadata = await gemini_client.generate_embeddings(chunks_to_process)
    except Exception as e:
        logger.error(f"{job_prefix} Unexpected error calling gemini_client.generate_embeddings: {e}", exc_info=True)
        # Treat unexpected client error as internal server error
        raise HTTPException(status_code=500, detail="Internal error during embedding generation.")
    
    if embedded_chunks is None:
        # Gemini client explicitly returned None, indicating failure
        error_msg = usage_metadata.get('error', 'Unknown embedding generation error')
        logger.error(f"{job_prefix} Embedding generation failed. Error: {error_msg}")
        # Return 500 if embedding itself failed critically
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {error_msg}")
    
    logger.info(f"{job_prefix} Embedding generation completed. Usage: {usage_metadata}. Result count: {len(embedded_chunks)}")
    
    # --- Step 2: Upsert Embeddings into Database ---
    
    # Separate successfully embedded chunks from those skipped by Gemini (e.g., empty text)
    chunks_to_upsert = [c for c in embedded_chunks if c.embedding is not None]
    skipped_chunk_ids = [c.chunk_id for c in embedded_chunks if c.embedding is None and c.chunk_id] # Get IDs of skipped chunks
    
    logger.info(f"{job_prefix} Attempting to upsert {len(chunks_to_upsert)} chunks with embeddings.")
    
    processed_ids = []
    failed_ids = list(skipped_chunk_ids) # Start failed list with skipped chunks
    
    if chunks_to_upsert:
        try:
            # Call the local vector_ops function to handle DB interaction
            upserted_ids = await vector_ops.upsert_embeddings(chunks_to_upsert)
            # upsert_embeddings returns the list of IDs it *attempted* to upsert successfully
            processed_ids.extend(upserted_ids)
            logger.info(f"{job_prefix} Database upsert request successful for {len(upserted_ids)} vectors.")
        except Exception as db_error:
            # Catch errors raised by upsert_embeddings (RuntimeError or others)
            logger.error(f"{job_prefix} Database upsert operation failed: {db_error}", exc_info=True)
            # If DB fails, mark all chunks we *tried* to upsert as failed
            failed_ids.extend([c.chunk_id for c in chunks_to_upsert if c.chunk_id])
            # Return 500 Internal Server Error as the service couldn't complete storage
            raise HTTPException(status_code=500, detail=f"Failed to store embeddings in database: {db_error}")
    else:
        logger.warning(f"{job_prefix} No chunks had embeddings generated, nothing to upsert.")
    
    # --- Step 3: Return Response ---
    logger.info(f"{job_prefix} Embed request processed. Final counts -> Processed: {len(processed_ids)}, Failed/Skipped: {len(failed_ids)}")
    return EmbedResponse(processed_chunk_ids=processed_ids, failed_chunk_ids=failed_ids)

@app.post("/search", response_model=SearchResponse)
async def search_vectors(query: SearchQuery):
    """
    Receives a search query, generates a query embedding using Gemini,
    and performs a similarity search in the Supabase database via vector_ops.
    """
    logger.info(f"Received /search request. Top_k={query.top_k}, Filters={query.filters}, Query='{query.query_text[:50]}...'")
    
    # --- Step 1: Generate Query Embedding ---
    logger.info(f"Requesting query embedding generation for: '{query.query_text[:50]}...'")
    
    query_embedding: Optional[List[float]] = None
    usage_metadata: Dict[str, Any] = {}
    
    try:
        query_embedding, usage_metadata = await gemini_client.generate_query_embedding(query.query_text)
    except Exception as e:
        logger.error(f"Unexpected error calling gemini_client.generate_query_embedding: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error during query embedding generation.")
    
    if query_embedding is None:
        error_msg = usage_metadata.get('error', 'Unknown query embedding error')
        logger.error(f"Failed to generate query embedding: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Failed to generate query embedding: {error_msg}")
    
    logger.info(f"Query embedding generated successfully. Usage: {usage_metadata}")
    
    # --- Step 2: Perform Vector Search ---
    logger.info(f"Performing vector search in database...")
    
    try:
        # Call the local vector_ops function
        search_results = await vector_ops.search_similar(
            query_text=query.query_text, # Pass original text for logging/context
            query_embedding=query_embedding, # Pass the generated embedding
            top_k=query.top_k,
            filters=query.filters
        )
        
        logger.info(f"Vector search completed. Found {len(search_results)} results.")
        # Wrap results in the response model
        return SearchResponse(results=search_results)
    except ValueError as ve: # Catch specific configuration errors like RPC not found
        logger.error(f"Search configuration or RPC error: {ve}", exc_info=False)
        # Return 400 Bad Request for configuration-related issues
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as search_error:
        # Catch errors raised by search_similar (RuntimeError or others)
        logger.error(f"Vector search operation failed: {search_error}", exc_info=True)
        # Return 500 Internal Server Error if the search itself fails
        raise HTTPException(status_code=500, detail=f"Vector search failed: {search_error}")

# --- Health Check Endpoint ---

@app.get("/health")
async def health_check():
    # Basic health check
    # Could add checks here for Gemini client configuration or Supabase connectivity
    logger.debug("Health check endpoint called")
    try:
        await get_supabase_client() # Quick check if client can init
        db_status = "connected"
    except Exception:
        db_status = "error"
    gemini_status = "configured" if gemini_client.configured else "not_configured"
    return {"status": "ok", "dependencies": {"database": db_status, "gemini": gemini_status}}
    return {"status": "ok", "message": "Vector Service is running"}
