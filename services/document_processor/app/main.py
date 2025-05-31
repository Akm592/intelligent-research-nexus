# services/document_processor/app/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from core.models import ProcessRequest, DocumentChunk, EmbedRequest, EmbedResponse, PaperMetadata # Added PaperMetadata
from core.config import settings, logger as core_logger # Use core logger
from . import processing, crud # Use local crud and processing modules
import httpx
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional

# --- Minimal KG Payload Models (defined here for simplicity) ---
# These match the expected input structure of the KG service API endpoints
from pydantic import BaseModel

class KgPaperPayload(BaseModel):
    """Payload to send to KG Service for Paper node creation/update."""
    paper_id: str
    title: Optional[str] = None
    publication_date: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None # Abstract/main URL
    pdf_url: Optional[str] = None # PDF URL

class KgAuthorPayload(BaseModel):
    """Payload to send to KG Service for Author node creation/update."""
    name: str

class AuthoredByPayload(BaseModel):
    """Payload to send to KG Service for AUTHORED_BY relationship creation."""
    paper_id: str
    author_name: str
# --- End KG Payloads ---


logger = core_logger.getChild("DocProcessor") # Child logger for this service

# --- HTTP Client Lifespan Management ---
http_client: httpx.AsyncClient | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifespan of the HTTP client for downstream calls."""
    global http_client
    # Configure timeouts: connect, read, write, pool
    timeout_config = httpx.Timeout(180.0, connect=30.0)
    http_client = httpx.AsyncClient(timeout=timeout_config)
    logger.info("Document Processor started. HTTPX Client initialized.")
    yield # Application runs
    # Cleanup on shutdown
    if http_client:
        await http_client.aclose()
        logger.info("HTTPX Client closed.")
    else:
         logger.warning("HTTP Client was not initialized during shutdown.")

# --- FastAPI App Instance ---
app = FastAPI(
    title="Document Processor Service",
    description="Parses documents, extracts text, chunks, triggers embedding and KG population.",
    version="1.0.0",
    lifespan=lifespan # Use the lifespan manager
)

# --- Helper function to get the shared HTTP client ---
async def get_http_client() -> httpx.AsyncClient:
     """Dependency function to get the application's HTTP client."""
     if http_client is None:
          logger.error("HTTP client dependency not met: Client is not available.")
          # Use 503 Service Unavailable if the client isn't ready
          raise HTTPException(
              status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
              detail="Internal error: HTTP client not ready"
          )
     return http_client

# --- Helper Function to Fetch Metadata for KG ---
async def _get_paper_metadata_for_kg(paper_id: str) -> Optional[PaperMetadata]:
    """
    Fetches metadata required for KG population from the 'papers' table
    using the document processor's CRUD functions.
    """
    job_prefix = f"[{paper_id}]"
    logger.debug(f"{job_prefix} Fetching metadata from DB for KG population...")
    try:
        # Use the local crud module which accesses the 'papers' table
        supabase = await crud.get_supabase_client() # Get client via core utility
        def db_call():
            # Select all fields defined in PaperMetadata that are relevant for KG
            return supabase.table(crud.PAPERS_TABLE)\
                   .select("id, title, authors, publication_date, source, url, pdf_url")\
                   .eq(crud.PAPER_ID_COLUMN, paper_id)\
                   .limit(1)\
                   .maybe_single()\
                   .execute()

        # Run the synchronous DB call in a thread
        response = await asyncio.to_thread(db_call)

        if response and response.data:
             logger.debug(f"{job_prefix} Found metadata in DB for KG.")
             try:
                 # Ensure 'authors' list is handled correctly if DB returns null
                 db_data = response.data
                 if 'authors' not in db_data or db_data.get('authors') is None:
                     db_data['authors'] = [] # Default to empty list

                 # Parse into the Pydantic model
                 return PaperMetadata(**db_data)
             except Exception as parse_err:
                 logger.error(f"{job_prefix} Failed to parse metadata from DB for KG: {parse_err}", exc_info=False)
                 return None # Return None if parsing fails
        else:
             # Metadata record itself not found for this paper_id
             logger.warning(f"{job_prefix} Metadata record not found in DB for KG population (ID: {paper_id}).")
             return None
    except Exception as e:
        # Catch any other errors during DB fetch (Supabase client errors, etc.)
        logger.error(f"{job_prefix} Unexpected error fetching metadata for KG: {e}", exc_info=True)
        return None

# --- Helper Function to Call KG Service Endpoints ---
async def _call_kg_service(client: httpx.AsyncClient, endpoint: str, payload: BaseModel, paper_id_context: str):
    """Helper to make non-blocking calls to KG Service with basic error handling."""
    job_prefix = f"[{paper_id_context}]" # Include paper ID for context
    url = f"{settings.KG_SERVICE_URL}{endpoint}"
    try:
        # Use a reasonable timeout for KG calls (shorter than embedding calls)
        response = await client.post(url, json=payload.model_dump(exclude_none=True), timeout=30.0)
        response.raise_for_status() # Raise HTTPStatusError for 4xx/5xx responses
        logger.debug(f"{job_prefix} Successfully called KG service endpoint: {endpoint} for {payload}")
        return True # Indicate success
    except httpx.HTTPStatusError as e:
        error_detail = "Unknown KG Error"
        try:
            # Try to get specific detail from KG service response
            error_detail = e.response.json().get('detail', e.response.text[:200]) # Limit error text length
        except Exception:
            error_detail = e.response.text[:200] # Fallback to raw text
        logger.error(f"{job_prefix} KG Service returned error ({e.response.status_code}) for {endpoint}: {error_detail}", exc_info=False)
        return False # Indicate failure
    except httpx.RequestError as e:
        # Network errors (connection refused, DNS resolution failed, etc.)
        logger.error(f"{job_prefix} Could not connect to KG Service at {url}: {e}", exc_info=False)
        return False
    except Exception as e:
        # Other unexpected errors during the call
        logger.error(f"{job_prefix} Unexpected error calling KG service endpoint {endpoint}: {e}", exc_info=True)
        return False


# --- Background Task Definition ---
async def process_and_embed_task(request: ProcessRequest, http_client_bg: httpx.AsyncClient):
    """
    Background task for processing:
    1. Get content (URL/Storage)
    2. Parse & Chunk
    3. Embed & Store Chunks (Vector Service)
    4. Update Status
    5. Populate Knowledge Graph (KG Service - Paper/Authors)
    """
    paper_id = request.paper_id
    job_prefix = f"[{paper_id}]"
    final_status = "failed" # Default status unless successful
    status_message = "Processing initiated." # Initial message
    processed_successfully = False # Flag to track if main steps completed

    try:
        logger.info(f"{job_prefix} Starting background processing.")
        # 1. Update status to 'processing'
        await crud.update_paper_status(paper_id, "processing", status_message)

        # 2. Get Document Content using the logic in processing.py
        doc_content, doc_metadata = await processing.get_document_content(request)
        if not doc_content:
            status_message = f"Failed to get document content (source: {request.source_url or request.object_name or 'DB lookup'})."
            logger.error(f"{job_prefix} {status_message}")
            raise ProcessingError(status_message) # Use custom error for flow control

        # 3. Parse and Chunk using the logic in processing.py
        chunks: List[DocumentChunk] = await processing.parse_and_chunk(paper_id, doc_content, doc_metadata)
        if not chunks:
            status_message = f"Failed to parse/chunk document (check format/content)."
            logger.error(f"{job_prefix} {status_message}")
            raise ProcessingError(status_message)
        logger.info(f"{job_prefix} Successfully parsed {len(chunks)} chunks.")

        # 4. Call Vector Service to Generate Embeddings and Store
        logger.info(f"{job_prefix} Calling Vector Service to embed and store {len(chunks)} chunks...")
        # Prepare payload for vector service
        embed_payload = EmbedRequest(chunks=[c.model_dump(exclude={'embedding'}) for c in chunks])
        vector_service_url = f"{settings.VECTOR_SERVICE_URL}/embed"

        try:
            # Make the call using the provided http client
            response = await http_client_bg.post(vector_service_url, json=embed_payload.model_dump())
            response.raise_for_status() # Check for 4xx/5xx errors
            embed_response_data = response.json()
            # Validate response structure (optional but recommended)
            embed_response = EmbedResponse(**embed_response_data)

            processed_count = len(embed_response.processed_chunk_ids)
            failed_count = len(embed_response.failed_chunk_ids)

            if failed_count > 0:
                 logger.warning(f"{job_prefix} Vector Service reported failure for {failed_count} chunks.")
                 if processed_count > 0:
                     final_status = "processed_with_errors"
                     status_message = f"Successfully processed {processed_count} chunks, but {failed_count} embedding/storage errors occurred."
                     processed_successfully = True # Partial success is still success for KG step
                 else:
                     final_status = "failed"
                     status_message = f"Vector Service failed to process any chunks."
                     raise ProcessingError(status_message) # Treat total vector failure as critical error
            else:
                final_status = "processed"
                status_message = f"Successfully processed and stored {processed_count} chunks."
                processed_successfully = True # Full success

            logger.info(f"{job_prefix} Vector Service call completed. Status: {final_status}. Message: {status_message}")

        # Handle specific errors from vector service call
        except httpx.RequestError as e:
            status_message = f"Failed to connect to Vector Service at {vector_service_url}: {e}"
            logger.error(f"{job_prefix} {status_message}", exc_info=False)
            raise ProcessingError(status_message) from e
        except httpx.HTTPStatusError as e:
             error_detail = f"Vector Service Error ({e.response.status_code})"
             try: downstream_error = e.response.json().get('detail', e.response.text)
             except Exception: downstream_error = e.response.text
             status_message = f"{error_detail}: {downstream_error}"
             logger.error(f"{job_prefix} {status_message}", exc_info=False)
             raise ProcessingError(status_message) from e
        except Exception as e: # Catch Pydantic validation errors, JSON decode errors, etc.
             status_message = f"Error processing Vector Service response: {e}"
             logger.error(f"{job_prefix} {status_message}", exc_info=True)
             raise ProcessingError(status_message) from e

        # 5. Update Final Status (before potentially long KG step)
        logger.info(f"{job_prefix} Updating final main processing status to '{final_status}'.")
        await crud.update_paper_status(paper_id, final_status, status_message)

    # Handle expected errors during the main processing flow
    except ProcessingError as pe:
         logger.error(f"{job_prefix} Processing stopped due to error: {pe}. Setting status to failed.")
         final_status = "failed"
         status_message = str(pe) # Use error message from exception
         await crud.update_paper_status(paper_id, final_status, status_message)
         processed_successfully = False # Ensure flag is false

    # Catch truly unexpected errors during main processing
    except Exception as e:
        status_message = f"Unhandled exception during main processing: {e}"
        logger.error(f"{job_prefix} {status_message}", exc_info=True)
        final_status = "failed"
        processed_successfully = False # Ensure flag is false
        try:
            # Try to update status to failed with a generic message
            await crud.update_paper_status(paper_id, final_status, "Unhandled exception during processing.")
        except Exception as db_err:
            logger.critical(f"{job_prefix} CRITICAL: Failed to update status to 'failed' after unhandled error: {db_err}")


    # --- 6. Populate Knowledge Graph (Phase 2 - Paper/Authors) ---
    # This runs *after* the main try-except block, only if processing was successful
    if processed_successfully:
        logger.info(f"{job_prefix} Starting KG population (Phase 2: Paper/Authors)...")
        # Fetch required metadata from the DB using helper function
        paper_meta = await _get_paper_metadata_for_kg(paper_id)

        if paper_meta:
            kg_tasks = [] # List to hold concurrent KG API call tasks

            # a) Prepare Paper Node Payload
            paper_payload = KgPaperPayload(
                paper_id=paper_meta.id,
                title=paper_meta.title,
                publication_date=paper_meta.publication_date,
                source=paper_meta.source,
                url=paper_meta.url,         # Abstract/Main URL
                pdf_url=paper_meta.pdf_url  # PDF URL
            )
            # Add task to create/update paper node
            kg_tasks.append(
                asyncio.create_task(
                    _call_kg_service(http_client_bg, "/nodes/paper", paper_payload, paper_id)
                )
            )

            # b) Prepare Author Nodes and Relationships Payloads
            if paper_meta.authors: # Check if authors list exists and is not empty
                 processed_authors = set() # Avoid duplicate calls for same author in one paper
                 for author_name in paper_meta.authors:
                     cleaned_author_name = author_name.strip() if author_name else None
                     if cleaned_author_name and cleaned_author_name not in processed_authors:
                         processed_authors.add(cleaned_author_name) # Mark as processed

                         # Add task for Author Node
                         author_payload = KgAuthorPayload(name=cleaned_author_name)
                         kg_tasks.append(
                              asyncio.create_task(
                                  _call_kg_service(http_client_bg, "/nodes/author", author_payload, paper_id)
                              )
                         )
                         # Add task for Relationship Node
                         rel_payload = AuthoredByPayload(paper_id=paper_id, author_name=cleaned_author_name)
                         kg_tasks.append(
                             asyncio.create_task(
                                 _call_kg_service(http_client_bg, "/relationships/authored_by", rel_payload, paper_id)
                             )
                         )

            # Wait for all KG calls to complete (concurrently)
            logger.info(f"{job_prefix} Executing {len(kg_tasks)} KG population tasks...")
            results = await asyncio.gather(*kg_tasks, return_exceptions=True)

            # Check results for failures/exceptions (optional detailed logging)
            failed_kg_calls = [res for res in results if isinstance(res, Exception) or res is False]
            if failed_kg_calls:
                logger.warning(f"{job_prefix} Encountered {len(failed_kg_calls)} errors/failures during KG population. Check KG service logs.")
                # Optionally update status message to indicate partial KG failure
                # await crud.update_paper_status(paper_id, final_status, status_message + " (KG population encountered errors)")
            else:
                 logger.info(f"{job_prefix} KG population calls completed successfully.")

        else:
            # Failed to get metadata needed for KG
            logger.error(f"{job_prefix} Could not retrieve metadata for KG population after successful processing.")
            # Optionally update status message
            # await crud.update_paper_status(paper_id, final_status, status_message + " (KG population skipped - metadata error)")
    else:
        # Main processing failed, skip KG population
        logger.warning(f"{job_prefix} Skipping KG population because main processing failed (status: {final_status}).")

    # --- End of Background Task ---
    logger.info(f"{job_prefix} Finished background processing task.")


# --- Custom Exception Class ---
class ProcessingError(Exception):
    """Custom exception for controlled error handling within the processing task."""
    pass

# --- API Endpoint to Trigger Processing ---
@app.post("/process", status_code=status.HTTP_202_ACCEPTED) # Use status constants
async def process_document_endpoint(
    request: ProcessRequest,
    background_tasks: BackgroundTasks,
    http_client_req: httpx.AsyncClient = Depends(get_http_client) # Inject client for task
):
    """
    Receives a request to process a document (parse, chunk, embed, populate KG).
    Validates input, checks current status, and schedules processing in the background.
    """
    paper_id = request.paper_id
    logger.info(f"Received processing request for paper: {paper_id}")

    # Check current status using the fixed crud function
    current_status = await crud.get_paper_status(paper_id)

    # --- Status Check Logic ---
    if current_status is None:
         # Paper metadata doesn't exist in the DB at all
         logger.error(f"Cannot process paper {paper_id}: Metadata record not found in database.")
         raise HTTPException(
             status_code=status.HTTP_404_NOT_FOUND,
             detail="Paper metadata record not found. Ensure paper exists in database before processing."
         )
    elif current_status in ["processing"]:
        logger.warning(f"Paper {paper_id} is currently processing. Skipping new task.")
        # Return 200 OK, it's not an error, just already happening
        return {"message": f"Paper processing is already in progress (status: {current_status}). No new task scheduled."}
    elif current_status in ["processed", "processed_with_errors"]:
       logger.warning(f"Paper {paper_id} has already been processed (status: '{current_status}'). Skipping new task.")
       # Return 200 OK
       return {"message": f"Paper processing already completed (status: {current_status}). No new task scheduled."}
    elif current_status == "failed":
        logger.info(f"Retrying processing for previously failed paper: {paper_id}")
        # Allow re-queueing
    elif current_status == "pending":
         logger.info(f"Queuing processing for pending paper: {paper_id}")
         # Allow queueing
    else:
         # Defensive check for unexpected status values
         logger.error(f"Paper {paper_id} has unknown status '{current_status}'. Blocking processing.")
         raise HTTPException(
             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
             detail=f"Paper has unexpected status '{current_status}'."
         )

    # --- Schedule Background Task ---
    # Pass the request object and the http_client instance to the task
    background_tasks.add_task(process_and_embed_task, request, http_client_req)
    logger.info(f"Scheduled background processing for paper: {paper_id}")

    # Return 202 Accepted immediately
    return {"message": "Document processing scheduled successfully."}