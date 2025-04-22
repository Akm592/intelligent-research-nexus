import asyncio
from core.models import PaperMetadata
from core.config import logger
from core.supabase_client import get_supabase_client, PAPERS_TABLE
from typing import Optional, List
from supabase import PostgrestAPIError

async def save_paper_metadata(paper: PaperMetadata) -> bool:
    try:
        supabase = await get_supabase_client()
        paper_dict = paper.model_dump(exclude_none=True)
        if 'authors' in paper_dict and paper_dict['authors'] is None:
            paper_dict['authors'] = []
        if 'keywords' in paper_dict and paper_dict['keywords'] is None:
            paper_dict['keywords'] = []
        logger.debug(f"Attempting to upsert metadata for paper ID: {paper.id}")
        response = supabase.table(PAPERS_TABLE).upsert(paper_dict, on_conflict='id').execute()
        logger.info(f"Successfully upserted metadata for paper ID: {paper.id}. Response count: {len(response.data)}")
        return len(response.data) > 0
    except PostgrestAPIError as e:
        logger.error(f"Supabase error saving paper metadata for {paper.id}: {e.message}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving paper metadata for {paper.id}: {e}", exc_info=True)
        return False

async def get_paper_metadata(paper_id: str) -> Optional[PaperMetadata]:
    try:
        supabase = await get_supabase_client()
        logger.debug(f"Attempting to retrieve metadata for paper ID: {paper_id}")
        response = supabase.table(PAPERS_TABLE).select("*").eq('id', paper_id).limit(1).execute()
        if response.data:
            logger.info(f"Successfully retrieved metadata for paper ID: {paper_id}")
            return PaperMetadata(**response.data[0])
        else:
            logger.info(f"No metadata found for paper ID: {paper_id}")
            return None
    except PostgrestAPIError as e:
        logger.error(f"Supabase error retrieving paper metadata for {paper_id}: {e.message}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error retrieving paper metadata for {paper_id}: {e}", exc_info=True)
        return None

async def get_papers_by_ids(paper_ids: List[str]) -> List[PaperMetadata]:
    if not paper_ids:
        return []
    try:
        supabase = await get_supabase_client()
        logger.debug(f"Attempting to retrieve metadata for paper IDs: {paper_ids}")
        response = supabase.table(PAPERS_TABLE).select("*").in_('id', paper_ids).execute()
        if response.data:
            logger.info(f"Successfully retrieved {len(response.data)} papers by IDs.")
            return [PaperMetadata(**item) for item in response.data]
        else:
            logger.info(f"No metadata found for paper IDs: {paper_ids}")
            return []
    except PostgrestAPIError as e:
        logger.error(f"Supabase error retrieving papers by IDs: {e.message}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Unexpected error retrieving papers by IDs: {e}", exc_info=True)
        return []
