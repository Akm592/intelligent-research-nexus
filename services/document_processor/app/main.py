# services/document_processor/app/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from core.models import ProcessRequest, DocumentChunk, EmbedRequest, EmbedResponse # Added EmbedResponse model
from core.config import settings
from . import processing, crud
import httpx
import logging
import asyncio

from typing import List, Optional

logger = logging.getLogger("IRN_Core").getChild("DocProcessor")

app = FastAPI(title="Document Processor Service")

# Single HTTP client instance for the service lifespan
http_client: httpx.AsyncClient | None = None

@app.on_event("startup")
async def startup_event():
    global http_client
    # Increased timeout for potentially slow embedding/DB calls downstream
    http_client = httpx.AsyncClient(timeout=180.0)
    logger.info("Document Processor started. HTTPX Client initialized.")

@app.on_event("shutdown")
async def shutdown_event():
    if http_client:
        await http_client.aclose()
        logger.info("HTTPX Client closed.")


async def process_and_embed_task(request: ProcessRequest):
    """Background task for processing: fetch, parse, chunk, embed, store."""
    paper_id = request.paper_id
    final_status = "failed" # Default to failed unless explicitly set to processed
    status_message = ""

    try:
        logger.info(f"Starting background processing for paper: {paper_id}")
        await crud.update_paper_status(paper_id, "processing")

        # 1. Get Document Content
        doc_content, doc_metadata = await processing.get_document_content(request)
        if not doc_content:
            status_message = f"Failed to get document content for {paper_id}."
            logger.error(status_message)
            raise ProcessingError(status_message) # Custom exception for clean exit

        # 2. Parse and Chunk
        chunks: List[DocumentChunk] = await processing.parse_and_chunk(paper_id, doc_content, doc_metadata)
        if not chunks:
            status_message = f"Failed to parse/chunk document {paper_id} (likely empty or invalid format)."
            logger.error(status_message)
            raise ProcessingError(status_message)
        logger.info(f"Successfully parsed {len(chunks)} chunks for {paper_id}.")

        # 3. Call Vector Service to Generate Embeddings and Store
        logger.info(f"Calling Vector Service to embed and store {len(chunks)} chunks for {paper_id}...")
        if not http_client:
             status_message = "HTTP Client not available for Vector Service call."
             logger.error(status_message)
             raise ProcessingError(status_message)

        # Prepare payload: Vector service expects list of chunk dicts
        # Exclude embedding field as it's not generated yet
        embed_payload = EmbedRequest(chunks=[c.model_dump(exclude={'embedding'}) for c in chunks])

        vector_service_url = f"{settings.VECTOR_SERVICE_URL}/embed"
        try:
            response = await http_client.post(vector_service_url, json=embed_payload.model_dump())
            response.raise_for_status() # Check for 4xx/5xx errors

            # Process response from vector service
            embed_response_data = response.json()
            # Use Pydantic model for validation and clarity
            embed_response = EmbedResponse(**embed_response_data)

            processed_count = len(embed_response.processed_chunk_ids)
            failed_count = len(embed_response.failed_chunk_ids)

            if failed_count > 0:
                 logger.warning(f"Vector Service reported failure for {failed_count} chunks (Paper: {paper_id}). Check Vector Service logs.")
                 # Decide on final status based on partial success
                 if processed_count > 0:
                     final_status = "processed_with_errors"
                     status_message = f"Processing finished with {failed_count} embedding/storage errors."
                 else:
                     final_status = "failed"
                     status_message = f"Vector Service failed to process any chunks for {paper_id}."
                     raise ProcessingError(status_message) # Treat total failure as critical
            else:
                final_status = "processed"
                status_message = f"Successfully processed and stored {processed_count} chunks."

            logger.info(f"Vector Service call completed for {paper_id}. Status: {final_status}. {status_message}")

        except httpx.RequestError as e:
            status_message = f"Failed to connect to Vector Service at {vector_service_url} for {paper_id}: {e}"
            logger.error(status_message, exc_info=False)
            raise ProcessingError(status_message) from e
        except httpx.HTTPStatusError as e:
             error_detail = f"Vector Service Error ({e.response.status_code})"
             try: downstream_error = e.response.json().get('detail', e.response.text)
             except Exception: downstream_error = e.response.text
             status_message = f"{error_detail} during embedding call for {paper_id}: {downstream_error}"
             logger.error(status_message, exc_info=False)
             raise ProcessingError(status_message) from e
        except Exception as e: # Catch JSON decode errors or other issues
             status_message = f"Error processing Vector Service response for {paper_id}: {e}"
             logger.error(status_message, exc_info=True)
             raise ProcessingError(status_message) from e

        # 4. Update Status (final_status determined above)
        await crud.update_paper_status(paper_id, final_status, status_message)
        logger.info(f"Finished background processing task for paper: {paper_id}. Final Status: {final_status}")

    except ProcessingError as pe:
         # Handle expected processing failures cleanly
         logger.error(f"Processing failed for {paper_id}: {pe}. Setting status to failed.")
         await crud.update_paper_status(paper_id, "failed", str(pe))
    except Exception as e:
        # Catch truly unexpected errors
        status_message = f"Unhandled error during background processing for paper {paper_id}: {e}"
        logger.error(status_message, exc_info=True)
        try:
            await crud.update_paper_status(paper_id, "failed", "Unhandled exception during processing.")
        except Exception as db_err:
            logger.error(f"CRITICAL: Failed to update status to 'failed' for {paper_id} after unhandled error: {db_err}")

# Custom exception for flow control
class ProcessingError(Exception):
    pass

@app.post("/process", status_code=202) # HTTP 202 Accepted
async def process_document_endpoint(request: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Receives a request to process a document (parse, chunk, embed).
    Validates input and schedules the processing in the background.
    """
    # ... (validation logic checking paper status remains the same as previous version) ...
    logger.info(f"Received processing request for paper: {request.paper_id}")
    current_status = await crud.get_paper_status(request.paper_id)
    if current_status in ["processing", "processed", "processed_with_errors"]:
       logger.warning(f"Paper {request.paper_id} is already in status '{current_status}'. Skipping queueing.")
       return {"message": f"Paper processing already initiated or completed (status: {current_status})."}
    elif current_status == "failed":
        logger.info(f"Retrying processing for failed paper: {request.paper_id}")
    elif current_status is None:
         logger.error(f"Cannot process paper {request.paper_id}: Metadata not found.")
         raise HTTPException(status_code=404, detail="Paper metadata not found. Fetch or save metadata before processing.")

    background_tasks.add_task(process_and_embed_task, request)
    logger.info(f"Scheduled background processing for paper: {request.paper_id}")
    return {"message": "Document processing scheduled successfully."}


# --- CRUD Revision ---
# Add status message field to crud functions
async def update_paper_status(paper_id: str, status: str, message: Optional[str] = None):
    """Updates the processing status and message of a paper in Supabase."""
    try:
        supabase = await crud.get_supabase_client() # Assuming crud imports get_supabase_client
        update_data = {crud.STATUS_COLUMN: status}
        if message:
            # Assuming you have a 'status_message' or similar column in your 'papers' table
            update_data["status_message"] = message # Add this column to your SQL setup if needed

        logger.debug(f"Attempting to update status='{status}' msg='{message is not None}' for paper ID: {paper_id}")
        response = await supabase.table(crud.STATUS_TABLE)\
            .update(update_data)\
            .eq(crud.PAPER_ID_COLUMN, paper_id)\
            .execute()
        logger.info(f"Update status request sent for paper ID: {paper_id}. Status: {status}")
        return True
    except Exception as e:
        logger.error(f"Error updating status/message for {paper_id}: {e}", exc_info=True)
        return False

# Make sure crud.py has get_paper_status as implemented before
from . import crud # Ensure crud is importable within main