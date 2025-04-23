from fastapi import FastAPI, HTTPException
from typing import List
from core.models import PaperMetadata, FetchRequest
from . import logic, crud
from core.config import logger

app = FastAPI(title="Paper Fetcher Service")

@app.post("/fetch", response_model=List[PaperMetadata])
async def fetch_papers(request: FetchRequest):
    logger.info(f"Received fetch request: query='{request.query}', sources={request.sources}")
    try:
        papers = await logic.search_academic_sources(request.query, request.sources, request.max_results)
        for paper in papers:
            await crud.save_paper_metadata(paper)
        logger.info(f"Saved metadata for {len(papers)} papers.")
        logger.info(f"Successfully fetched {len(papers)} paper(s) for query '{request.query}'.")
        return papers
    except Exception as e:
        logger.error(f"Error fetching papers for query '{request.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch papers.")

@app.get("/paper/{paper_id}", response_model=PaperMetadata)
async def get_paper_metadata(paper_id: str):
    logger.info(f"Received request for paper metadata: ID={paper_id}")
    metadata = PaperMetadata(id=paper_id, title=f"Dummy Title for {paper_id}", source="dummy")
    if not metadata:
        raise HTTPException(status_code=404, detail="Paper metadata not found")
    return metadata
