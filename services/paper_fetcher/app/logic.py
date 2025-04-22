from core.models import PaperMetadata
from typing import List, Optional
import logging
import asyncio
import datetime
import arxiv
import re
from urllib.parse import urlparse
import uuid

logger = logging.getLogger("IRN_Core").getChild("PaperFetcher").getChild("Logic")

def _parse_arxiv_id(entry_id: str) -> Optional[str]:
    try:
        parsed = urlparse(entry_id)
        match = re.search(r'/(abs|pdf)/([\w.-]+)', parsed.path)
        if match:
            arxiv_code = match.group(2)
            return f"arxiv:{arxiv_code}"
        else:
            logger.warning(f"Could not parse arXiv code from entry ID: {entry_id}")
            return None
    except Exception as e:
        logger.error(f"Error parsing arXiv entry ID '{entry_id}': {e}", exc_info=True)
        return None

async def search_arxiv(query: str, max_results: int) -> List[PaperMetadata]:
    logger.info(f"Querying arXiv API for: '{query}' (max_results: {max_results})")
    search_results: List[PaperMetadata] = []
    try:
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
            if not paper_id:
                logger.warning(f"Skipping result with unparseable entry ID: {result.entry_id}")
                continue
            authors = [author.name for author in result.authors]
            title = result.title.replace('\n', ' ').replace('  ', ' ').strip()
            abstract = result.summary.replace('\n', ' ').replace('  ', ' ').strip()
            publication_date_iso = None
            if result.published:
                try:
                    publication_date_iso = result.published.isoformat()
                except Exception:
                    logger.warning(f"Could not format publication date {result.published} for {paper_id}")
            keywords = result.categories or []
            if result.primary_category and result.primary_category not in keywords:
                keywords.insert(0, result.primary_category)
            metadata = PaperMetadata(
                id=paper_id,
                title=title,
                authors=authors,
                abstract=abstract,
                publication_date=publication_date_iso,
                source="arxiv",
                url=result.entry_id,
                keywords=keywords
            )
            search_results.append(metadata)
        logger.info(f"Successfully processed {len(search_results)} papers from arXiv for query '{query}'.")
    except Exception as e:
        logger.error(f"Error querying arXiv API for '{query}': {e}", exc_info=True)
        return search_results
    return search_results

async def search_semantic_scholar(query: str, max_results: int) -> List[PaperMetadata]:
    logger.warning(f"Semantic Scholar search simulation for '{query}'. Needs implementation.")
    await asyncio.sleep(0.1)
    results = []
    if max_results > 0:
        s2_id_suffix = f"{query.replace(' ', '_').lower()[:15]}_{uuid.uuid4().hex[:4]}"
        s2_id = f"s2:{s2_id_suffix}"
        results.append(PaperMetadata(
            id=s2_id,
            title=f"Simulated S2 Paper on '{query}' (Implementation Pending)",
            authors=["S2 Author"],
            abstract=f"Placeholder abstract for Semantic Scholar result {s2_id}. Implement API call.",
            publication_date=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            source="semantic_scholar",
            url=f"https://api.semanticscholar.org/{s2_id_suffix}",
            keywords=["simulation", "s2", "pending"]
        ))
    return results[:max_results]

async def search_academic_sources(query: str, sources: List[str], max_results: int) -> List[PaperMetadata]:
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
                if paper.id not in processed_ids:
                    all_results.append(paper)
                    processed_ids.add(paper.id)
        else:
            logger.warning(f"Unexpected item type returned from gather: {type(result_item)}")
    logger.info(f"Total unique papers fetched from requested sources: {len(all_results)}")
    final_results = all_results[:max_results]
    return final_results
