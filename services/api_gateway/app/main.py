# services/api_gateway/app/main.py
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.routing import APIRoute
from core.config import settings
from core.models import GatewayResponse # Using the specific GatewayResponse
import httpx
import time
from typing import Callable
import logging
from contextlib import asynccontextmanager
import os
from pydantic import BaseModel

# Use logger configured in core.config
logger = logging.getLogger("IRN_Core").getChild("APIGateway")


# --- Basic Placeholder Authentication (Example - adapt as needed) ---
async def verify_api_key(request: Request):
    EXPECTED_API_KEY = os.getenv("API_GATEWAY_KEY") # Example: Get key from env for flexibility
    if not EXPECTED_API_KEY:
        # logger.warning("API Key verification skipped (API_GATEWAY_KEY not set)")
        return # Skip check if no key is configured

    provided_key = request.headers.get("X-API-Key")
    if not provided_key or provided_key != EXPECTED_API_KEY:
        logger.warning(f"Unauthorized access attempt: Missing or incorrect API Key from {request.client.host}.")
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")

# --- Rate Limiting Placeholder ---
# Consider using a proper library like 'slowapi' for production
# This in-memory version is NOT suitable for multi-instance deployments
RATE_LIMIT_STORE = {}
RATE_LIMIT_MAX_CALLS = 100
RATE_LIMIT_PERIOD = 60 # seconds

async def rate_limiter(request: Request):
    client_ip = request.client.host if request.client else "unknown"
    current_time = time.time()

    # Basic cleanup (inefficient for high load)
    for ip in list(RATE_LIMIT_STORE.keys()):
        if current_time - RATE_LIMIT_STORE[ip]['timestamp'] > RATE_LIMIT_PERIOD * 1.5: # Longer cleanup period
            try: del RATE_LIMIT_STORE[ip]
            except KeyError: pass # Already deleted by another request

    if client_ip not in RATE_LIMIT_STORE:
        RATE_LIMIT_STORE[client_ip] = {'count': 1, 'timestamp': current_time}
    else:
        client_data = RATE_LIMIT_STORE.get(client_ip) # Re-fetch in case it was deleted
        if client_data:
             if current_time - client_data['timestamp'] < RATE_LIMIT_PERIOD:
                if client_data['count'] >= RATE_LIMIT_MAX_CALLS:
                    logger.warning(f"Rate limit exceeded for IP: {client_ip}")
                    # Consider headers like Retry-After
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")
                client_data['count'] += 1
                client_data['timestamp'] = current_time # Update timestamp on activity
             else:
                # Reset period
                RATE_LIMIT_STORE[client_ip] = {'count': 1, 'timestamp': current_time}
        else: # Entry was deleted between check and fetch
             RATE_LIMIT_STORE[client_ip] = {'count': 1, 'timestamp': current_time}


# --- Custom Route Class for Common Dependencies ---
# Uncomment `Depends(verify_api_key)` to enable authentication
class BaseRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()
        async def custom_route_handler(request: Request) -> GatewayResponse:
            # await Depends(verify_api_key)(request=request) # Apply auth if enabled
            await Depends(rate_limiter)(request=request) # Apply rate limiting
            return await original_route_handler(request)
        # return custom_route_handler # Uncomment to enable common deps on all routes

        # If only applying to specific routes, use `dependencies=[Depends(rate_limiter)]` in @app.get/post
        return original_route_handler


# --- Service Client ---
# Using a single client instance managed by lifespan
http_client: httpx.AsyncClient | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the client and store it in app.state
    logger.info("API Gateway lifespan startup: Initializing HTTPX Client.")
    try:
        app.state.http_client = httpx.AsyncClient(timeout=60.0)
        logger.info("HTTPX Client initialized and stored in app.state.")
    except Exception as e:
        logger.error(f"Failed to initialize HTTPX client during startup: {e}", exc_info=True)
        # Prevent app startup if client fails? Or let it start and fail requests?
        # For now, log error and continue; requests will fail.
        app.state.http_client = None

    yield # Application runs here

    # Shutdown: Close the client if it exists
    logger.info("API Gateway lifespan shutdown: Cleaning up resources.")
    if getattr(app.state, 'http_client', None):
        logger.info("Closing HTTPX Client.")
        await app.state.http_client.aclose()
        app.state.http_client = None # Clear the state
        logger.info("HTTPX Client closed.")
    else:
        logger.warning("HTTPX Client was not available in app.state during shutdown.")

# --- FastAPI App ---
app = FastAPI(
    title="IRN API Gateway",
    description="Entry point for the Intelligent Research Nexus",
    version="1.0.0",
    lifespan=lifespan # Manage client lifecycle via lifespan
)

# --- Health Check ---
@app.get("/health", response_model=GatewayResponse, tags=["Meta"])
async def health_check(request: Request): # Add request to access app state
    # Check if client initialized properly during startup
    client_status = "initialized" if getattr(request.app.state, 'http_client', None) else "NOT initialized"
    return GatewayResponse(status="success", message=f"API Gateway is running (HTTP Client: {client_status})")

# --- Routing ---
# Import routers AFTER app is defined
from .routers import analysis, documents, search

# Apply common dependencies like rate limiting here if desired for all routes in a router
# Or apply them individually in the router files
app.include_router(analysis.router, prefix="/analysis", tags=["Analysis"])
app.include_router(documents.router, prefix="/documents", tags=["Documents"])
app.include_router(search.router, prefix="/search", tags=["Search"])

# Example root endpoint
@app.get("/", response_model=GatewayResponse, tags=["Meta"])
async def read_root():
    return GatewayResponse(status="success", message="Welcome to the IRN API Gateway")