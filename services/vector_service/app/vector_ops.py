# services/vector_service/app/vector_ops.py

import asyncio
from core.models import DocumentChunk, SearchResultItem
from core.config import settings, logger as core_logger # Use core logger, settings
from core.supabase_client import get_supabase_client, CHUNKS_TABLE # Use chunks table name from client module
from core.gemini_client import gemini_client # Needed if generating fallback query embedding here
from typing import List, Dict, Any, Optional

# Use a child logger specific to this module
logger = core_logger.getChild("VectorService").getChild("VectorOps")

# Column and function names matching your SQL schema
VECTOR_COLUMN = "embedding"
SEARCH_FUNCTION = "match_document_chunks" # Your RPC function name

async def upsert_embeddings(chunks: List[DocumentChunk]) -> List[str]:
    """
    Upserts chunks with embeddings into the Supabase 'chunks' table.
    ... Raises exceptions on failure to be caught by the endpoint handler.
    Returns a list of chunk_ids that were successfully sent in the upsert request.
    """
    if not chunks:
        logger.info("No chunks provided for upsert.")
        return []

    # Assume all chunks are for the same paper for logging context if possible
    paper_id_context = chunks[0].paper_id if chunks else "N/A"
    job_prefix = f"[{paper_id_context}]"
    logger.info(f"{job_prefix} Preparing to upsert {len(chunks)} vectors into table '{CHUNKS_TABLE}'.")

    supabase = await get_supabase_client() # Get client instance
    vectors_to_upsert = []
    ids_in_request = [] # Keep track of IDs we are attempting to upsert

    for chunk in chunks:
        # Ensure chunk has necessary data before attempting upsert
        if chunk.embedding and chunk.chunk_id and chunk.paper_id:
            # Prepare data dictionary matching exact column names in 'chunks' table
            chunk_data = {
                "chunk_id": chunk.chunk_id,
                "paper_id": chunk.paper_id,
                "text": chunk.text, # Store full text
                VECTOR_COLUMN: chunk.embedding, # The vector embedding list
                "metadata": chunk.metadata, # The JSONB metadata
                # 'created_at' is handled by DB default
            }
            vectors_to_upsert.append(chunk_data)
            ids_in_request.append(chunk.chunk_id)
        elif not chunk.embedding:
            logger.warning(f"{job_prefix} Chunk '{chunk.chunk_id or 'MISSING_ID'}' skipped (no embedding found).")
        else: # Missing chunk_id or paper_id
            logger.warning(f"{job_prefix} Chunk skipped (missing chunk_id or paper_id). Text: {chunk.text[:50]}...")

    if not vectors_to_upsert:
        logger.warning(f"{job_prefix} No valid chunks with embeddings found to upsert.")
        return [] # Return empty list as nothing was attempted

    try:
        logger.debug(f"{job_prefix} Attempting Supabase upsert for {len(vectors_to_upsert)} vectors.")
        # Upsert into the table using chunk_id as the conflict target (primary key)
        response = await supabase.table(CHUNKS_TABLE)\
            .upsert(vectors_to_upsert, on_conflict='chunk_id')\
            .execute()
        # Log success based on absence of errors
        logger.info(f"{job_prefix} Successfully sent upsert request for {len(vectors_to_upsert)} vectors to Supabase.")
        # Return the list of IDs that were included in the successful upsert request
        return ids_in_request
    except Exception as e:
        logger.error(f"{job_prefix} Database error upserting vectors: {e}", exc_info=True)
        raise RuntimeError(f"Database error during upsert: {e}") from e

async def search_similar(query_text: str, top_k: int, filters: Optional[Dict[str, Any]] = None, query_embedding: Optional[List[float]] = None) -> List[SearchResultItem]:
    """
    Searches for similar vectors using the 'match_document_chunks' Supabase RPC function.
    Raises exceptions on failure.
    """
    logger.info(f"Performing vector search. Top_k={top_k}, Filters={filters}, Query='{query_text[:50]}...'")

    if query_embedding is None:
        # This path should ideally not be hit if main.py generates the embedding first
        logger.error("CRITICAL: search_similar called without providing a query_embedding.")
        raise ValueError("Query embedding must be provided for search.")

    supabase = await get_supabase_client()

    # Prepare arguments for the RPC function, matching SQL definition *exactly*
    rpc_params = {
        'query_embedding': query_embedding, # Type: vector(768)
        'match_threshold': settings.SEARCH_MATCH_THRESHOLD, # Make threshold configurable
        'match_count': top_k, # Type: int
    }

    # Handle optional filter parameter 'filter_paper_id' (Type: text)
    # Ensure filter key and value type match the RPC function definition
    if filters and 'paper_id' in filters:
        filter_val = filters['paper_id']
        if isinstance(filter_val, str) and filter_val:
            rpc_params['filter_paper_id'] = filter_val # Parameter name matches SQL function
            logger.info(f"Applying filter: paper_id = {filter_val}")
        else:
            logger.warning(f"Ignoring invalid filter value type for paper_id: {type(filter_val)}. Expected string.")
            # Let the RPC function use its default (NULL) if filter is invalid
    else:
        # Explicitly set to None if not provided or invalid, matching the SQL default NULL
        rpc_params['filter_paper_id'] = None

    # Log parameters being sent (excluding sensitive embedding)
    log_params = {k:v for k,v in rpc_params.items() if k != 'query_embedding'}
    logger.debug(f"Calling Supabase RPC function '{SEARCH_FUNCTION}' with params: {log_params}")

    try:
        # Execute the RPC function
        response = await supabase.rpc(SEARCH_FUNCTION, rpc_params).execute()

        # response.data should contain a list of dictionaries matching the RETURNS TABLE structure
        if response.data:
            logger.info(f"Supabase RPC search returned {len(response.data)} results.")
            search_results = []
            for item in response.data:
                # Map the returned columns ('id', 'paper_id', 'content', 'metadata', 'similarity')
                # to the SearchResultItem Pydantic model.
                try:
                    result_item = SearchResultItem(
                        chunk_id=item.get("id"), # 'id' from RPC is chunk_id
                        paper_id=item.get("paper_id"),
                        score=item.get("similarity", 0.0), # 'similarity' from RPC
                        text=item.get("content", ""), # 'content' from RPC is chunk text
                        metadata=item.get("metadata", {}) # 'metadata' from RPC is chunk metadata (JSONB)
                    )
                    search_results.append(result_item)
                except Exception as parse_error: # Catch potential Pydantic validation errors
                    logger.warning(f"Failed to parse search result item: {item}. Error: {parse_error}", exc_info=False)
            return search_results
        else:
            logger.info(f"Supabase RPC search returned no results.")
            return [] # Return empty list if no matches found
    except Exception as e:
        # Check for common "does not exist" error
        error_message = str(e)
        if "function" in error_message.lower() and SEARCH_FUNCTION in error_message and "does not exist" in error_message.lower():
            logger.critical(f"!!! RPC function '{SEARCH_FUNCTION}' not found in Supabase. Ensure the SQL script was run correctly. !!!")
            # Raise a ValueError for configuration issues
            raise ValueError(f"Configuration error: RPC function '{SEARCH_FUNCTION}' not found in Supabase.") from e
        else:
            logger.error(f"Database error during RPC search: {e}", exc_info=True)
            raise RuntimeError(f"Database error during search: {e}") from e
