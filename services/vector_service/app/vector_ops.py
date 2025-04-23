# services/vector_service/app/vector_ops.py

import asyncio # <-- Import asyncio
from core.models import DocumentChunk, SearchResultItem
from core.config import settings, logger as core_logger # Use core logger, settings
# Use client utility, table names, and ensure client init uses run_in_executor
from core.supabase_client import get_supabase_client, CHUNKS_TABLE
# Gemini client might not be needed here if main.py handles all embedding generation
# from core.gemini_client import gemini_client
from typing import List, Dict, Any, Optional
from supabase import PostgrestAPIError # Import specific error
from postgrest import APIResponse

# Use a child logger specific to this module
logger = core_logger.getChild("VectorService").getChild("VectorOps")

# Column and function names matching your SQL schema
VECTOR_COLUMN = "embedding"
SEARCH_FUNCTION = "match_document_chunks" # Your RPC function name

async def upsert_embeddings(chunks: List[DocumentChunk]) -> List[str]:
    """
    Upserts chunks with embeddings into the Supabase 'chunks' table using asyncio.to_thread.
    Raises exceptions on failure to be caught by the endpoint handler.
    Returns a list of chunk_ids that were successfully included in the upsert request data.
    """
    if not chunks:
        logger.info("No chunks provided for upsert.")
        return []

    # Assume all chunks are for the same paper for logging context if possible
    paper_id_context = chunks[0].paper_id if chunks else "N/A"
    job_prefix = f"[{paper_id_context}]"
    logger.info(f"{job_prefix} Preparing to upsert {len(chunks)} vectors into table '{CHUNKS_TABLE}'.")

    supabase = await get_supabase_client() # Get async-initialized client instance
    vectors_to_upsert = []
    ids_in_request = [] # Keep track of IDs we are attempting to upsert

    for chunk in chunks:
        # Ensure chunk has necessary data before attempting upsert
        if chunk.embedding and chunk.chunk_id and chunk.paper_id and chunk.text is not None: # Check text too
            # Prepare data dictionary matching exact column names in 'chunks' table
            chunk_data = {
                "chunk_id": chunk.chunk_id,
                "paper_id": chunk.paper_id,
                "text": chunk.text, # Store full text
                VECTOR_COLUMN: chunk.embedding, # The vector embedding list
                "metadata": chunk.metadata or {}, # Ensure metadata is at least an empty dict
                # 'created_at' is handled by DB default
            }
            vectors_to_upsert.append(chunk_data)
            ids_in_request.append(chunk.chunk_id)
        elif not chunk.embedding:
            logger.warning(f"{job_prefix} Chunk '{chunk.chunk_id or 'MISSING_ID'}' skipped (no embedding found).")
        else: # Missing chunk_id, paper_id, or text
            missing_fields = [
                f for f in ['chunk_id', 'paper_id', 'text'] if getattr(chunk, f, None) is None
            ]
            logger.warning(f"{job_prefix} Chunk skipped (missing fields: {missing_fields}). Chunk ID: {chunk.chunk_id}")

    if not vectors_to_upsert:
        logger.warning(f"{job_prefix} No valid chunks with embeddings found to upsert.")
        return [] # Return empty list as nothing was attempted

    try:
        logger.debug(f"{job_prefix} Attempting Supabase upsert for {len(vectors_to_upsert)} vectors.")

        # --- Define the synchronous DB call ---
        def db_upsert_call():
            # Upsert into the table using chunk_id as the conflict target (primary key)
            return supabase.table(CHUNKS_TABLE)\
                .upsert(vectors_to_upsert, on_conflict='chunk_id')\
                .execute()

        # --- Run the synchronous call in a thread ---
        response: APIResponse = await asyncio.to_thread(db_upsert_call)

        # Log success based on absence of errors (Supabase client raises errors)
        # The actual response.data might be empty for upsert, which is okay.
        logger.info(f"{job_prefix} Successfully sent upsert request for {len(vectors_to_upsert)} vectors to Supabase.")
        # Return the list of IDs that were included in the successful upsert request batch
        return ids_in_request

    except PostgrestAPIError as e:
        # Log specific Supabase errors
        logger.error(f"{job_prefix} Database API error upserting vectors: {e.message} (Code: {e.code}, Details: {e.details})", exc_info=False)
        # Raise a runtime error to be caught by the main endpoint handler
        raise RuntimeError(f"Database API error during upsert: {e.message}") from e
    except Exception as e:
        logger.error(f"{job_prefix} Unexpected error during database upsert: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected database error during upsert: {e}") from e


async def search_similar(query_text: str, top_k: int, filters: Optional[Dict[str, Any]] = None, query_embedding: Optional[List[float]] = None) -> List[SearchResultItem]:
    """
    Searches for similar vectors using the 'match_document_chunks' Supabase RPC function,
    using asyncio.to_thread.
    Raises exceptions on failure.
    """
    logger.info(f"Performing vector search. Top_k={top_k}, Filters={filters}, Query='{query_text[:50]}...'")

    if query_embedding is None:
        logger.error("CRITICAL: search_similar called without providing a query_embedding.")
        raise ValueError("Query embedding must be provided for search.")

    supabase = await get_supabase_client() # Get async-initialized client

    # Prepare arguments for the RPC function, matching SQL definition *exactly*
    rpc_params = {
        'query_embedding': query_embedding,
        'match_threshold': settings.SEARCH_MATCH_THRESHOLD,
        'match_count': top_k,
    }

    # Handle optional filter parameter 'filter_paper_id'
    if filters and 'paper_id' in filters:
        filter_val = filters['paper_id']
        if isinstance(filter_val, str) and filter_val:
            rpc_params['filter_paper_id'] = filter_val
            logger.info(f"Applying filter: paper_id = {filter_val}")
        else:
            logger.warning(f"Ignoring invalid filter value type for paper_id: {type(filter_val)}. Expected non-empty string.")
            rpc_params['filter_paper_id'] = None # Explicitly set to None if invalid
    else:
        rpc_params['filter_paper_id'] = None # Default to None if filter not provided

    # Log parameters being sent (excluding sensitive embedding)
    log_params = {k: v for k, v in rpc_params.items() if k != 'query_embedding'}
    logger.debug(f"Calling Supabase RPC function '{SEARCH_FUNCTION}' with params: {log_params}")

    try:
        # --- Define the synchronous DB RPC call ---
        def db_rpc_call():
            return supabase.rpc(SEARCH_FUNCTION, rpc_params).execute()

        # --- Run the synchronous call in a thread ---
        response: APIResponse = await asyncio.to_thread(db_rpc_call)

        search_results = []
        if response.data:
            logger.info(f"Supabase RPC search returned {len(response.data)} results.")
            for item in response.data:
                try:
                    # Map RPC return columns to SearchResultItem fields
                    result_item = SearchResultItem(
                        chunk_id=item.get("id"), # 'id' from RPC is chunk_id
                        paper_id=item.get("paper_id"),
                        score=item.get("similarity", 0.0), # 'similarity' from RPC
                        text=item.get("content", ""), # 'content' from RPC is chunk text
                        metadata=item.get("metadata", {}) # 'metadata' from RPC
                    )
                    search_results.append(result_item)
                except Exception as parse_error:
                    logger.warning(f"Failed to parse search result item: {item}. Error: {parse_error}", exc_info=False)
            return search_results
        else:
            logger.info(f"Supabase RPC search returned no results.")
            return [] # Return empty list

    except PostgrestAPIError as e:
         # Check for common "does not exist" error specifically for the RPC function
        if "relation" in e.message and f"public.{SEARCH_FUNCTION}" in e.message and "does not exist" in e.message:
             logger.critical(f"!!! RPC function '{SEARCH_FUNCTION}' not found in Supabase. Ensure the SQL script defining it was run correctly. !!!")
             raise ValueError(f"Configuration error: RPC function '{SEARCH_FUNCTION}' not found in Supabase.") from e
        else:
             logger.error(f"Database API error during RPC search: {e.message} (Code: {e.code}, Details: {e.details})", exc_info=False)
             raise RuntimeError(f"Database API error during search: {e.message}") from e
    except Exception as e:
        logger.error(f"Unexpected error during RPC search: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected database error during search: {e}") from e