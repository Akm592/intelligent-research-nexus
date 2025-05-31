# services/document_processor/app/crud.py
import asyncio
from core.config import logger as core_logger
from core.supabase_client import get_supabase_client, PAPERS_TABLE, CHUNKS_TABLE # Use table names from client module
from typing import List, Optional
from core.models import DocumentChunk
from supabase import PostgrestAPIError
# --- REMOVE SingleAPIResponse ---
from postgrest import APIResponse # Keep APIResponse if used elsewhere or for general hinting

logger = core_logger.getChild("DocProcessor").getChild("CRUD")

STATUS_TABLE = PAPERS_TABLE
STATUS_COLUMN = "processing_status"
PAPER_ID_COLUMN = "id"
STATUS_MESSAGE_COLUMN = "status_message" # Ensure this column exists in 'papers' table

async def update_paper_status(paper_id: str, status: str, message: Optional[str] = None) -> bool:
    """Updates the processing_status and optional status_message of a paper."""
    job_prefix = f"[{paper_id}]"
    logger.debug(f"{job_prefix} Attempting status update: status='{status}', message='{message is not None}'.")
    try:
        supabase = await get_supabase_client()
        VALID_STATUSES = {"pending", "processing", "processed", "failed", "processed_with_errors"}
        if status not in VALID_STATUSES:
             logger.error(f"{job_prefix} Invalid status value '{status}' for update.")
             return False

        update_data = {STATUS_COLUMN: status, STATUS_MESSAGE_COLUMN: message} # Update both

        def db_call():
            return supabase.table(STATUS_TABLE)\
                .update(update_data)\
                .eq(PAPER_ID_COLUMN, paper_id)\
                .execute()

        response: APIResponse = await asyncio.to_thread(db_call) # Use general APIResponse hint
        logger.info(f"{job_prefix} Sent update status request to '{status}'.")
        # For update, success is usually indicated by lack of exception
        return True

    except PostgrestAPIError as e:
         logger.error(f"{job_prefix} Supabase API error updating status: {e.message} (Code: {e.code}, Details: {e.details})", exc_info=False)
         return False
    except Exception as e:
        logger.error(f"{job_prefix} Unexpected error updating status: {e}", exc_info=True)
        return False

async def get_paper_status(paper_id: str) -> Optional[str]:
    """Gets the current processing_status of a paper from the 'papers' table."""
    job_prefix = f"[{paper_id}]"
    logger.debug(f"{job_prefix} Attempting retrieve status from table '{STATUS_TABLE}'.")
    try:
        supabase = await get_supabase_client()
        def db_call():
            # maybe_single() is designed to return data directly or None
            return supabase.table(STATUS_TABLE)\
                .select(f"{STATUS_COLUMN}")\
                .eq(PAPER_ID_COLUMN, paper_id)\
                .limit(1)\
                .maybe_single()\
                .execute()

        # --- REMOVE SingleAPIResponse HINT ---
        # The actual type might vary slightly, but logic depends on .data
        response = await asyncio.to_thread(db_call)

        # Check if response object exists AND has data attribute
        if response and hasattr(response, 'data') and response.data:
            status = response.data.get(STATUS_COLUMN)
            logger.info(f"{job_prefix} Retrieved status '{status}'.")
            return status
        else:
            # This covers both "not found" and potential empty responses
            logger.warning(f"{job_prefix} Paper ID not found in '{STATUS_TABLE}' or no data returned, cannot get status.")
            return None # Not found

    except PostgrestAPIError as e:
         logger.error(f"{job_prefix} Supabase API error retrieving status: {e.message} (Code: {e.code}, Details: {e.details})", exc_info=False)
         return None # Error
    except Exception as e:
        logger.error(f"{job_prefix} Unexpected error retrieving status: {e}", exc_info=True)
        return None

# --- Add function to get full metadata if not using the helper in main.py ---
# Example (ensure columns match PaperMetadata fields needed for KG):
# async def get_full_paper_metadata(paper_id: str) -> Optional[PaperMetadata]:
#     ... similar logic using asyncio.to_thread ...