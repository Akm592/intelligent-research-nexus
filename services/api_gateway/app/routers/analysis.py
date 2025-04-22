# services/api_gateway/app/routers/analysis.py
from fastapi import APIRouter, HTTPException, Body, Request, Depends # Added Request
from core.models import GatewayAnalysisRequest, AnalysisRequest, GatewayResponse
from core.config import settings
import httpx
import logging
# Removed direct import of http_client from main
# from ..main import http_client
from ..main import rate_limiter # Import dependencies if applying per-router

# Use logger configured in core.config, get child logger
logger = logging.getLogger("IRN_Core").getChild("APIGateway").getChild("AnalysisRouter")

router = APIRouter(dependencies=[Depends(rate_limiter)]) # Apply dependency here

@router.post("/", response_model=GatewayResponse)
async def route_analysis(
    request: Request, # Inject request object to access app.state
    payload: GatewayAnalysisRequest = Body(...) # Rename body variable to avoid conflict
):
    """Route analysis requests to the Analysis Engine."""
    # Access client from app.state via the request object
    http_client = getattr(request.app.state, 'http_client', None)
    if not http_client:
         logger.error("HTTP client not available in application state.")
         # Use 503 Service Unavailable as the gateway cannot fulfill request due to missing dependency
         raise HTTPException(status_code=503, detail="Gateway internal error: HTTP client not ready")

    logger.info(f"Routing analysis request: type={payload.analysis_type}, query={payload.query is not None}, papers={payload.paper_ids}")
    analysis_payload = AnalysisRequest(**payload.model_dump())
    downstream_url = f"{settings.ANALYSIS_ENGINE_URL}/analyze"

    try:
        response = await http_client.post(downstream_url, json=analysis_payload.model_dump())
        response.raise_for_status()
        logger.info(f"Analysis Engine call successful (Status: {response.status_code})")
        return GatewayResponse(status="success", data=response.json())

    except httpx.HTTPStatusError as e:
        error_detail = f"Analysis Engine Error ({e.response.status_code})"
        try: downstream_error = e.response.json().get('detail', e.response.text)
        except Exception: downstream_error = e.response.text
        logger.error(f"{error_detail} calling {downstream_url}: {downstream_error}", exc_info=False)
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)

    except httpx.RequestError as e:
        logger.error(f"Could not connect to Analysis Engine at {downstream_url}: {e}")
        raise HTTPException(status_code=503, detail="Analysis Engine service unavailable.")

    except Exception as e:
         logger.error(f"Gateway error during analysis routing: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail="Internal Gateway Error")