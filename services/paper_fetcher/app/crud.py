import asyncio 
from core.models import PaperMetadata
from core.config import logger as core_logger
from core.supabase_client import get_supabase_client, PAPERS_TABLE
from typing import Optional, List, Any
from supabase import PostgrestAPIError
from postgrest import APIResponse  # Remove SingleAPIResponse import

# Use a child logger
logger = core_logger.getChild("PaperFetcher").getChild("CRUD")

async def save_paper_metadata(paper: PaperMetadata) -> bool:
    """Saves paper metadata to Supabase using upsert."""
    job_prefix = f"[{paper.id}]"
    try:
        supabase = await get_supabase_client()
        paper_dict = paper.model_dump(exclude_none=True)

        # Ensure arrays are not None if schema expects non-nullable arrays
        if 'authors' in paper_dict and paper_dict['authors'] is None:
            paper_dict['authors'] = []
        if 'keywords' in paper_dict and paper_dict['keywords'] is None:
            paper_dict['keywords'] = []

        logger.debug(f"{job_prefix} Attempting to upsert metadata.")

        # Define the synchronous DB call
        def db_call():
            # Upsert: insert if not exists, update if exists based on 'id'
            return supabase.table(PAPERS_TABLE)\
                   .upsert(paper_dict, on_conflict='id')\
                   .execute()

        # --- Run the synchronous call in a thread ---
        response = await asyncio.to_thread(db_call)

        # Check response for success indicators (e.g., data length)
        if response.data and len(response.data) > 0:
            logger.info(f"{job_prefix} Successfully upserted metadata. Response count: {len(response.data)}")
            return True
        else:
            # This might happen if the upsert didn't change anything or returned unexpectedly
            logger.warning(f"{job_prefix} Upsert call executed but returned no data in response. Check DB state.")
            # Consider if this should be True or False based on desired behavior
            return False # Treat no data return as potential issue

    except PostgrestAPIError as e:
        logger.error(f"{job_prefix} Supabase error saving paper metadata: {e.message} (Code: {e.code}, Details: {e.details})", exc_info=False)
        return False
    except Exception as e:
        logger.error(f"{job_prefix} Unexpected error saving paper metadata: {e}", exc_info=True)
        return False

async def get_paper_metadata(paper_id: str) -> Optional[PaperMetadata]:
    """Retrieves metadata for a specific paper."""
    job_prefix = f"[{paper_id}]"
    try:
        supabase = await get_supabase_client()
        logger.debug(f"{job_prefix} Attempting to retrieve metadata.")

        # Define the synchronous DB call
        def db_call():
            # Use maybe_single for cleaner handling of 0 or 1 result
            return supabase.table(PAPERS_TABLE)\
                   .select("*")\
                   .eq('id', paper_id)\
                   .limit(1)\
                   .maybe_single()\
                   .execute()

        # --- Run the synchronous call in a thread ---
        response = await asyncio.to_thread(db_call)  # Remove type annotation

        if response and hasattr(response, 'data') and response.data:
            logger.info(f"{job_prefix} Successfully retrieved metadata.")
            # Validate data against Pydantic model
            try:
                return PaperMetadata(**response.data)
            except Exception as p_err:
                logger.error(f"{job_prefix} Failed to parse retrieved metadata into Pydantic model: {p_err}", exc_info=False)
                return None # Data format mismatch
        else:
            logger.info(f"{job_prefix} No metadata found.")
            return None

    except PostgrestAPIError as e:
        logger.error(f"{job_prefix} Supabase error retrieving paper metadata: {e.message} (Code: {e.code}, Details: {e.details})", exc_info=False)
        return None
    except Exception as e:
        logger.error(f"{job_prefix} Unexpected error retrieving paper metadata: {e}", exc_info=True)
        return None

async def get_papers_by_ids(paper_ids: List[str]) -> List[PaperMetadata]:
    """Retrieves metadata for a list of paper IDs."""
    if not paper_ids:
        return []
    try:
        supabase = await get_supabase_client()
        logger.debug(f"Attempting to retrieve metadata for {len(paper_ids)} paper IDs.")

        # Define the synchronous DB call
        def db_call():
            return supabase.table(PAPERS_TABLE)\
                   .select("*")\
                   .in_('id', paper_ids)\
                   .execute()

        # --- Run the synchronous call in a thread ---
        response = await asyncio.to_thread(db_call)  # Remove type annotation

        if response.data:
            logger.info(f"Successfully retrieved {len(response.data)} papers by IDs.")
            papers = []
            for item in response.data:
                try:
                    papers.append(PaperMetadata(**item))
                except Exception as p_err:
                     item_id = item.get('id', 'UNKNOWN')
                     logger.warning(f"Failed to parse paper metadata for ID {item_id}: {p_err}", exc_info=False)
            return papers
        else:
            logger.info(f"No metadata found for provided paper IDs: {paper_ids}")
            return []

    except PostgrestAPIError as e:
        logger.error(f"Supabase error retrieving papers by IDs: {e.message} (Code: {e.code}, Details: {e.details})", exc_info=False)
        return []
    except Exception as e:
        logger.error(f"Unexpected error retrieving papers by IDs: {e}", exc_info=True)
        return []
