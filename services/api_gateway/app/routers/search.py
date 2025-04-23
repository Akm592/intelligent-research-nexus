# services/api_gateway/app/routers/search.py
from fastapi import APIRouter, HTTPException, Body, Request, Depends # Added Request, Depends
from core.models import GatewaySearchRequest, SearchQuery, GatewayResponse
from core.config import settings
import httpx
import logging
# Removed direct import of http_client from main
# from ..main import http_client
from ..main import rate_limiter # Import dependencies if applying per-router
# Import the dependency function created in documents.py (or create it here)
from .documents import get_http_client

# Use logger configured in core.config, get child logger
logger = logging.getLogger("IRN_Core").getChild("APIGateway").getChild("SearchRouter")

router = APIRouter(dependencies=[Depends(rate_limiter)]) # Apply dependency here

@router.post("/", response_model=GatewayResponse)
async def route_search(
    # Inject client via dependency function
    http_client: httpx.AsyncClient = Depends(get_http_client),
    payload: GatewaySearchRequest = Body(...) # Rename body variable
):
    """Route search requests to the Vector Service."""
    logger.info(f"Routing search request: query='{payload.query}', top_k={payload.top_k}")
    search_payload = SearchQuery(query_text=payload.query, top_k=payload.top_k)
    downstream_url = f"{settings.VECTOR_SERVICE_URL}/search"

    try:
        response = await http_client.post(downstream_url, json=search_payload.model_dump())
        response.raise_for_status()
        logger.info(f"Vector Service search call successful (Status: {response.status_code})")
        return GatewayResponse(status="success", data=response.json())

    # ... (Error handling remains the same as previous version) ...
    except httpx.HTTPStatusError as e:
        error_detail = f"Vector Service Error ({e.response.status_code})"
        try: downstream_error = e.response.json().get('detail', e.response.text)
        except Exception: downstream_error = e.response.text
        logger.error(f"{error_detail} calling {downstream_url}: {downstream_error}", exc_info=False)
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)
    except httpx.RequestError as e:
        logger.error(f"Could not connect to Vector Service at {downstream_url}: {e}")
        raise HTTPException(status_code=503, detail="Vector Search service unavailable.")
    except Exception as e:
         logger.error(f"Gateway error during search routing: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail="Internal Gateway Error")