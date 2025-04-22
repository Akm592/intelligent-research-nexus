# services/document_processor/app/crud.py
import asyncio
from core.config import logger as core_logger # Use the core logger
from core.supabase_client import get_supabase_client, PAPERS_TABLE # Import client utility and table name
from typing import List, Optional
from core.models import DocumentChunk # Import if saving chunks metadata here
from supabase import PostgrestAPIError

# Use a child logger specific to this module
logger = core_logger.getChild("DocProcessor").getChild("CRUD")

# Table name where processing status is stored
STATUS_TABLE = PAPERS_TABLE
STATUS_COLUMN = "processing_status" # Column name from your schema
PAPER_ID_COLUMN = "id" # Primary key column name from your schema
# --- Add this if you add the status_message column to your DB ---
STATUS_MESSAGE_COLUMN = "status_message"

async def update_paper_status(paper_id: str, status: str, message: Optional[str] = None) -> bool:
    """Updates the processing_status and optional status_message of a paper."""
    job_prefix = f"[{paper_id}]"
    logger.debug(f"{job_prefix} Attempting update: status='{status}', message='{message is not None}'.")
    try:
        # Get the async client
        supabase = await get_supabase_client() # Use default key
        VALID_STATUSES = {"pending", "processing", "processed", "failed", "processed_with_errors"}
        if status not in VALID_STATUSES:
             logger.error(f"{job_prefix} Invalid status value '{status}' for update.")
             return False

        update_data = {STATUS_COLUMN: status}
        # --- Add this block if you add the status_message column ---
        # if message is not None: # Allow clearing the message with empty string? Or only set if provided?
        #     # Ensure you have a 'status_message' TEXT column in your 'papers' table
        #     update_data[STATUS_MESSAGE_COLUMN] = message

        # Use await with the async client methods
        response = await supabase.table(STATUS_TABLE)\
            .update(update_data)\
            .eq(PAPER_ID_COLUMN, paper_id)\
            .execute()

        logger.info(f"{job_prefix} Sent update status request to '{status}'.")
        return True
    except PostgrestAPIError as e:
         logger.error(f"{job_prefix} Supabase API error updating status: {e.message}", exc_info=False)
         return False
    except Exception as e:
        logger.error(f"{job_prefix} Unexpected error updating status: {e}", exc_info=True)
        return False

async def get_paper_status(paper_id: str) -> Optional[str]:
    """Gets the current processing_status of a paper from the 'papers' table."""
    job_prefix = f"[{paper_id}]"
    logger.debug(f"{job_prefix} Attempting retrieve status from table '{STATUS_TABLE}'.")
    try:
        # Get the async client
        supabase = await get_supabase_client()
        # Use await with the async client methods
        response = await supabase.table(STATUS_TABLE)\
            .select(f"{STATUS_COLUMN}")\
            .eq(PAPER_ID_COLUMN, paper_id)\
            .limit(1)\
            .maybe_single()\
            .execute()

        if response.data:
            status = response.data.get(STATUS_COLUMN)
            logger.info(f"{job_prefix} Retrieved status '{status}'.")
            return status
        else:
            logger.warning(f"{job_prefix} Paper ID not found in '{STATUS_TABLE}', cannot get status.")
            return None # Not found
    except PostgrestAPIError as e:
         logger.error(f"{job_prefix} Supabase API error retrieving status: {e.message}", exc_info=False)
         return None # Error
    except Exception as e:
        logger.error(f"{job_prefix} Unexpected error retrieving status: {e}", exc_info=True)
        return None