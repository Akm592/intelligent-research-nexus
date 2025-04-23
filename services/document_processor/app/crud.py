import asyncio # <-- Import asyncio
from core.config import logger as core_logger # Use the core logger
# Use client utility, table names, and ensure client init uses run_in_executor
from core.supabase_client import get_supabase_client, PAPERS_TABLE, CHUNKS_TABLE
from typing import List, Optional, Any
from core.models import DocumentChunk # Import if saving chunks metadata here
from supabase import PostgrestAPIError
from postgrest import APIResponse  # Remove SingleAPIResponse import


# Use a child logger specific to this module
logger = core_logger.getChild("DocProcessor").getChild("CRUD")

# Table name where processing status is stored (using PAPERS_TABLE)
STATUS_TABLE = PAPERS_TABLE
STATUS_COLUMN = "processing_status" # Column name from your schema
PAPER_ID_COLUMN = "id" # Primary key column name from your schema
# Column for optional status messages
STATUS_MESSAGE_COLUMN = "status_message" # Ensure this column exists in your 'papers' table!

async def update_paper_status(paper_id: str, status: str, message: Optional[str] = None) -> bool:
    """Updates the processing_status and optional status_message of a paper."""
    job_prefix = f"[{paper_id}]"
    logger.debug(f"{job_prefix} Attempting update: status='{status}', message='{message is not None}'.")
    try:
        # Get the async-initialized client
        supabase = await get_supabase_client() # This part is correct

        VALID_STATUSES = {"pending", "processing", "processed", "failed", "processed_with_errors"}
        if status not in VALID_STATUSES:
             logger.error(f"{job_prefix} Invalid status value '{status}' for update.")
             return False

        update_data = {STATUS_COLUMN: status}
        # --- Correctly add status_message ---
        # Update message only if provided (pass None to clear it if DB allows NULL)
        update_data[STATUS_MESSAGE_COLUMN] = message

        # Define the synchronous DB call
        def db_call():
            return supabase.table(STATUS_TABLE)\
                .update(update_data)\
                .eq(PAPER_ID_COLUMN, paper_id)\
                .execute()

        # --- Run the synchronous call in a thread ---
        response = await asyncio.to_thread(db_call)

        # The response object itself doesn't confirm success, lack of exception does.
        # You might check response.data if needed, but for update, checking error is key.
        logger.info(f"{job_prefix} Sent update status request to '{status}'.")
        # Consider adding more robust check based on response if needed
        return True

    except PostgrestAPIError as e:
         # Log the specific Supabase error
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
        # Get the async-initialized client
        supabase = await get_supabase_client() # Correct

        # Define the synchronous DB call
        def db_call():
            # Use maybe_single() to handle 0 or 1 result gracefully
            return supabase.table(STATUS_TABLE)\
                .select(f"{STATUS_COLUMN}")\
                .eq(PAPER_ID_COLUMN, paper_id)\
                .limit(1)\
                .maybe_single()\
                .execute()

        # --- Run the synchronous call in a thread ---
        response = await asyncio.to_thread(db_call)  # Remove type annotation

        # maybe_single() returns data directly if found, or None
        if response and hasattr(response, 'data') and response.data:
            status = response.data.get(STATUS_COLUMN)
            logger.info(f"{job_prefix} Retrieved status '{status}'.")
            return status
        else:
            # This happens if the paper_id doesn't exist in the table
            logger.warning(f"{job_prefix} Paper ID not found in '{STATUS_TABLE}', cannot get status.")
            return None # Not found

    except PostgrestAPIError as e:
         logger.error(f"{job_prefix} Supabase API error retrieving status: {e.message} (Code: {e.code}, Details: {e.details})", exc_info=False)
         return None # Error
    except Exception as e:
        # Catch any other unexpected errors (network issues during thread execution etc.)
        logger.error(f"{job_prefix} Unexpected error retrieving status: {e}", exc_info=True)
        return None
