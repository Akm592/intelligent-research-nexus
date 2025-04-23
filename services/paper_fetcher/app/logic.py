# services/paper_fetcher/app/logic.py

from core.models import PaperMetadata # Ensure model is imported
from typing import List, Optional
import logging
import asyncio
import datetime # Ensure datetime is imported
import arxiv
import re
from urllib.parse import urlparse
import uuid

logger = logging.getLogger("IRN_Core").getChild("PaperFetcher").getChild("Logic")

def _parse_arxiv_id(entry_id: str) -> Optional[str]:
    """Attempts to extract the core arXiv ID (e.g., '2305.12345v1') from various URL formats."""
    try:
        # Regex to find patterns like /abs/ID or /pdf/ID
        match = re.search(r'/(?:abs|pdf)/([\w.-]+)', entry_id)
        if match:
            arxiv_code = match.group(1)
            if arxiv_code.lower().endswith('.pdf'):
                arxiv_code = arxiv_code[:-4]
            return f"arxiv:{arxiv_code}"
        else:
            # Fallback for simpler ID formats
            parsed = urlparse(entry_id)
            base_id = parsed.path.split('/')[-1]
            if re.match(r'^[\d.]+[vV]?\d*$', base_id): # Basic check for ID format
                 if base_id.lower().endswith('.pdf'):
                     base_id = base_id[:-4]
                 return f"arxiv:{base_id}"

            logger.warning(f"Could not parse standardized arXiv code from entry ID: {entry_id}")
            return f"arxiv:unknown_{uuid.uuid4().hex[:8]}" # Generate placeholder if fails
    except Exception as e:
        logger.error(f"Error parsing arXiv entry ID '{entry_id}': {e}", exc_info=True)
        return f"arxiv:error_{uuid.uuid4().hex[:8]}" # Generate error placeholder


async def search_arxiv(query: str, max_results: int) -> List[PaperMetadata]:
    logger.info(f"Querying arXiv API for: '{query}' (max_results: {max_results})")
    search_results: List[PaperMetadata] = []
    try:
        # Use asyncio.to_thread for synchronous library calls
        search = await asyncio.to_thread(
            arxiv.Search,
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        raw_results = await asyncio.to_thread(list, search.results())
        logger.info(f"arXiv API returned {len(raw_results)} raw results.")

        for result in raw_results:
            paper_id = _parse_arxiv_id(result.entry_id)
            if not paper_id or "unknown" in paper_id or "error" in paper_id:
                logger.warning(f"Skipping result with unparseable or problematic entry ID: {result.entry_id} -> parsed as {paper_id}")
                continue

            # *** GET BOTH URLs ***
            abstract_url = result.entry_id # Keep the abstract page URL
            pdf_url = result.pdf_url      # Get the direct PDF URL

            if not pdf_url:
                 logger.warning(f"No PDF URL found for arXiv entry {paper_id} ({result.entry_id}). Storing abstract URL only.")
                 # Still store the metadata, but pdf_url will be None

            # Prepare other metadata fields (ensure robust handling of None)
            authors = [author.name for author in result.authors] if result.authors else []
            title = result.title.replace('\n', ' ').replace('  ', ' ').strip() if result.title else "N/A"
            abstract = result.summary.replace('\n', ' ').replace('  ', ' ').strip() if result.summary else "N/A"

            publication_date_iso = None
            if result.published and isinstance(result.published, datetime.datetime):
                try:
                    # Ensure timezone-aware datetime
                    if result.published.tzinfo is None:
                         publication_date_iso = result.published.replace(tzinfo=datetime.timezone.utc).isoformat()
                    else:
                         publication_date_iso = result.published.isoformat()
                except Exception as date_err:
                    logger.warning(f"Could not format publication date {result.published} for {paper_id}: {date_err}")

            keywords = result.categories if result.categories else []
            if result.primary_category and result.primary_category not in keywords:
                keywords.insert(0, result.primary_category)

            # *** Create metadata assigning URLs to correct fields ***
            metadata = PaperMetadata(
                id=paper_id,
                title=title,
                authors=authors,
                abstract=abstract,
                publication_date=publication_date_iso,
                source="arxiv",
                url=abstract_url,  # Store abstract URL in 'url'
                pdf_url=pdf_url,     # Store PDF URL in 'pdf_url'
                keywords=keywords,
            )
            search_results.append(metadata)
        logger.info(f"Successfully processed {len(search_results)} papers from arXiv for query '{query}'.")

    except Exception as e:
        logger.error(f"Error querying arXiv API for '{query}': {e}", exc_info=True)
        return search_results # Return partial results if any
    return search_results


async def search_semantic_scholar(query: str, max_results: int) -> List[PaperMetadata]:
    # NOTE: Needs actual implementation. Ensure pdf_url is populated.
    logger.warning(f"Semantic Scholar search simulation for '{query}'. Needs implementation.")
    await asyncio.sleep(0.1)
    results = []
    if max_results > 0:
        s2_id_suffix = f"{query.replace(' ', '_').lower()[:15]}_{uuid.uuid4().hex[:4]}"
        s2_id = f"s2:{s2_id_suffix}"
        # **Crucially, provide both a plausible abstract URL and PDF URL**
        placeholder_abstract_url = f"https://www.semanticscholar.org/paper/{s2_id_suffix}"
        placeholder_pdf_url = f"https://example.com/simulated_pdf/{s2_id_suffix}.pdf" # Or get from S2 API if available
        results.append(PaperMetadata(
            id=s2_id,
            title=f"Simulated S2 Paper on '{query}' (Implementation Pending)",
            authors=["S2 Author"],
            abstract=f"Placeholder abstract for Semantic Scholar result {s2_id}. Implement API call.",
            publication_date=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            source="semantic_scholar",
            url=placeholder_abstract_url, # Abstract/main page URL
            pdf_url=placeholder_pdf_url,  # Direct PDF URL
            keywords=["simulation", "s2", "pending"]
        ))
    return results[:max_results]


async def search_academic_sources(query: str, sources: List[str], max_results: int) -> List[PaperMetadata]:
    # This function just orchestrates calls, no changes needed here assuming
    # search_arxiv and search_semantic_scholar return PaperMetadata with pdf_url populated.
    logger.info(f"Searching academic sources for '{query}'. Sources: {sources}, Max Results: {max_results}")
    all_results: List[PaperMetadata] = []
    tasks = []
    if "arxiv" in sources:
        tasks.append(search_arxiv(query, max_results))
    if "semantic_scholar" in sources:
        tasks.append(search_semantic_scholar(query, max_results))
    if not tasks:
        logger.warning(f"No valid sources provided in request: {sources}")
        return []
    source_results_list = await asyncio.gather(*tasks, return_exceptions=True)
    processed_ids = set()
    for result_item in source_results_list:
        if isinstance(result_item, Exception):
            logger.error(f"An error occurred during concurrent source fetching: {result_item}", exc_info=False)
        elif isinstance(result_item, list):
            for paper in result_item:
                # Ensure paper object is valid before accessing id
                if hasattr(paper, 'id') and paper.id not in processed_ids:
                    all_results.append(paper)
                    processed_ids.add(paper.id)
                elif not hasattr(paper, 'id'):
                     logger.warning(f"Fetched item is not a valid PaperMetadata object: {paper}")
        else:
            logger.warning(f"Unexpected item type returned from gather: {type(result_item)}")
    logger.info(f"Total unique papers fetched from requested sources: {len(all_results)}")
    final_results = all_results[:max_results]
    logger.info(f"Returning final {len(final_results)} papers after applying max_results limit.")
    return final_results