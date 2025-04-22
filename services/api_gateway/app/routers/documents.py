# services/api_gateway/app/routers/documents.py
from fastapi import APIRouter, HTTPException, Body, Request, Query, Depends # Added Request, Depends
from core.models import (
    GatewayFetchRequest, FetchRequest,
    GatewayProcessRequest, ProcessRequest,
    GatewayResponse, PaperMetadata
)
from core.config import settings
import httpx
import logging
from typing import List
# Removed direct import of http_client from main
# from ..main import http_client
from ..main import rate_limiter # Import dependencies if applying per-router

# Use logger configured in core.config, get child logger
logger = logging.getLogger("IRN_Core").getChild("APIGateway").getChild("DocumentRouter")

router = APIRouter(dependencies=[Depends(rate_limiter)]) # Apply dependency here


def get_http_client(request: Request) -> httpx.AsyncClient:
    """Dependency function to get the HTTP client from app state."""
    client = getattr(request.app.state, 'http_client', None)
    if not client:
        logger.error("HTTP client dependency not met: Client not available in application state.")
        raise HTTPException(status_code=503, detail="Gateway internal error: HTTP client not ready")
    return client


@router.post("/fetch", response_model=GatewayResponse)
async def route_fetch_papers(
    # Inject client via dependency function
    http_client: httpx.AsyncClient = Depends(get_http_client),
    payload: GatewayFetchRequest = Body(...) # Rename body variable
):
    """Route paper fetching requests to the Paper Fetcher Service."""
    logger.info(f"Routing fetch request: query='{payload.query}', max_results={payload.max_results}")
    fetch_payload = FetchRequest(**payload.model_dump())
    downstream_url = f"{settings.PAPER_FETCHER_URL}/fetch"

    try:
        response = await http_client.post(downstream_url, json=fetch_payload.model_dump())
        response.raise_for_status()
        logger.info(f"Paper Fetcher call successful (Status: {response.status_code})")
        return GatewayResponse(status="success", data=response.json())

    # ... (Error handling remains the same as previous version) ...
    except httpx.HTTPStatusError as e:
        error_detail = f"Paper Fetcher Error ({e.response.status_code})"
        try: downstream_error = e.response.json().get('detail', e.response.text)
        except Exception: downstream_error = e.response.text
        logger.error(f"{error_detail} calling {downstream_url}: {downstream_error}", exc_info=False)
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)
    except httpx.RequestError as e:
        logger.error(f"Could not connect to Paper Fetcher at {downstream_url}: {e}")
        raise HTTPException(status_code=503, detail="Paper Fetcher service unavailable.")
    except Exception as e:
         logger.error(f"Gateway error during fetch routing: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail="Internal Gateway Error")


@router.post("/process", response_model=GatewayResponse)
async def route_process_document(
    # Inject client via dependency function
    http_client: httpx.AsyncClient = Depends(get_http_client),
    payload: GatewayProcessRequest = Body(...) # Rename body variable
):
    """Route document processing requests to the Document Processor Service."""
    logger.info(f"Routing process request for paper_id: {payload.paper_id}, bucket: {payload.bucket_name}, object: {payload.object_name}, url: {payload.source_url}")
    process_payload = ProcessRequest(
        paper_id=payload.paper_id,
        bucket_name=payload.bucket_name,
        object_name=payload.object_name,
        source_url=payload.source_url
    )
    downstream_url = f"{settings.DOC_PROCESSOR_URL}/process"

    try:
        response = await http_client.post(downstream_url, json=process_payload.model_dump(exclude_none=True))
        response.raise_for_status() # Doc processor returns 202 Accepted
        logger.info(f"Document Processor call accepted (Status: {response.status_code}) for paper {payload.paper_id}")
        return GatewayResponse(status="success", data=response.json(), message="Processing request accepted")

    # ... (Error handling remains the same as previous version) ...
    except httpx.HTTPStatusError as e:
        error_detail = f"Document Processor Error ({e.response.status_code})"
        try: downstream_error = e.response.json().get('detail', e.response.text)
        except Exception: downstream_error = e.response.text
        logger.error(f"{error_detail} while routing process request for {payload.paper_id}: {downstream_error}", exc_info=False)
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)
    except httpx.RequestError as e:
        logger.error(f"Could not connect to Document Processor at {downstream_url} for {payload.paper_id}: {e}")
        raise HTTPException(status_code=503, detail="Document Processor service unavailable.")
    except Exception as e:
         logger.error(f"Gateway error during process routing for {payload.paper_id}: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail="Internal Gateway Error")


@router.get("/{paper_id}", response_model=GatewayResponse)
async def route_get_paper_metadata(
    paper_id: str,
    # Inject client via dependency function
    http_client: httpx.AsyncClient = Depends(get_http_client)
):
    """Route request to get metadata for a specific paper."""
    logger.info(f"Routing get metadata request for paper_id: {paper_id}")
    downstream_url = f"{settings.PAPER_FETCHER_URL}/paper/{paper_id}"

    try:
        response = await http_client.get(downstream_url)
        response.raise_for_status()
        logger.info(f"Paper Fetcher GET call successful (Status: {response.status_code})")
        return GatewayResponse(status="success", data=response.json())

    # ... (Error handling remains the same as previous version) ...
    except httpx.HTTPStatusError as e:
        error_detail = f"Paper Fetcher Error ({e.response.status_code})"
        try: downstream_error = e.response.json().get('detail', e.response.text)
        except Exception: downstream_error = e.response.text
        logger.error(f"{error_detail} calling {downstream_url}: {downstream_error}", exc_info=False)
        status_code = e.response.status_code if e.response.status_code != 404 else 404
        detail = error_detail if status_code != 404 else "Paper metadata not found."
        raise HTTPException(status_code=status_code, detail=detail)
    except httpx.RequestError as e:
        logger.error(f"Could not connect to Paper Fetcher at {downstream_url}: {e}")
        raise HTTPException(status_code=503, detail="Paper Fetcher service unavailable.")
    except Exception as e:
         logger.error(f"Gateway error during get metadata routing: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail="Internal Gateway Error")