import pytest
import pytest_asyncio # Required for async fixtures
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
import httpx # For RequestError

# Assuming your gateway app is here
from services.api_gateway.app.main import app as api_gateway_app
from services.api_gateway.app.main import lifespan as api_gateway_lifespan # Import lifespan
from core.models import (
    GatewayAnalysisRequest, GatewayResponse, AnalysisResult,
    GatewayFetchRequest, PaperMetadata, GatewayProcessRequest, ProcessResponse,
    GatewaySearchRequest, SearchResponse, SearchResultItem
)
from core.config import settings # For checking downstream URLs

# Fixture to manage lifespan of the app
@pytest_asyncio.fixture
async def client():
    # Reset app state before each test, especially http_client
    if hasattr(api_gateway_app.state, 'http_client'):
        delattr(api_gateway_app.state, 'http_client')

    async with api_gateway_lifespan(api_gateway_app): # Correctly use lifespan
        with TestClient(api_gateway_app) as test_client:
            yield test_client
    # No explicit shutdown needed here as lifespan handles it


# Helper function to manage http_client mocking
@pytest_asyncio.fixture
async def mock_app_http_client(request):
    original_http_client = getattr(api_gateway_app.state, 'http_client', None)
    
    mock_client_instance = AsyncMock(spec=httpx.AsyncClient)
    
    # If test provides specific mock behavior, use it
    if hasattr(request, "param") and request.param:
        if "post_return_value" in request.param:
            mock_client_instance.post.return_value = request.param["post_return_value"]
        if "post_side_effect" in request.param:
            mock_client_instance.post.side_effect = request.param["post_side_effect"]
        if "get_return_value" in request.param:
            mock_client_instance.get.return_value = request.param["get_return_value"]
        if "get_side_effect" in request.param:
            mock_client_instance.get.side_effect = request.param["get_side_effect"]

    api_gateway_app.state.http_client = mock_client_instance
    
    yield mock_client_instance # Provide the mock instance to the test

    # Restore original client
    if original_http_client:
        api_gateway_app.state.http_client = original_http_client
    else:
        if hasattr(api_gateway_app.state, 'http_client'): # Check if it was set during the test
            delattr(api_gateway_app.state, 'http_client')


# --- Test Health and Root Endpoints ---
def test_health_check(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "success"
    assert "API Gateway is running" in json_response["message"]

def test_read_root(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "success"
    assert "Welcome to the IRN API Gateway" in json_response["message"]

# --- Test Analysis Route ---
@pytest.mark.asyncio # Mark test as async
async def test_route_analysis_success(client: TestClient):
    # This test needs the app's http_client to be mocked.
    # The client is initialized in the lifespan context manager.
    # We will patch app.state.http_client for the duration of this test.

    mock_downstream_response_data = AnalysisResult(result_text="mocked analysis", cited_sources=[], analysis_type="summary")
    
    # Store the original client if it exists, then patch
    original_http_client = getattr(api_gateway_app.state, 'http_client', None)
    
    mock_client_instance = AsyncMock(spec=httpx.AsyncClient)
    mock_client_instance.post.return_value = AsyncMock(
        status_code=200,
        json=lambda: mock_downstream_response_data.model_dump() # Simulate downstream service response
    )
    api_gateway_app.state.http_client = mock_client_instance

    payload = GatewayAnalysisRequest(analysis_type="summary", query="Test query")
    response = client.post("/analysis/", json=payload.model_dump())

    assert response.status_code == 200
    json_response = response.json()
    
    # Construct expected GatewayResponse
    expected_gateway_response = GatewayResponse(
        status="success",
        message="Analysis request processed successfully.",
        data=mock_downstream_response_data
    )
    
    assert json_response == expected_gateway_response.model_dump()

    # Assert that the mocked client's post method was called correctly
    expected_url = f"{settings.ANALYSIS_ENGINE_URL}/analyze"
    mock_client_instance.post.assert_called_once()
    args, kwargs = mock_client_instance.post.call_args
    assert args[0] == expected_url
    assert kwargs['json'] == payload.model_dump() # Ensure payload matches

    # This test uses the mock_app_http_client fixture to handle setup/teardown
    mock_client_instance = mock_app_http_client 
    
    mock_downstream_response_data = AnalysisResult(result_text="mocked analysis", cited_sources=[], analysis_type="summary")
    mock_client_instance.post.return_value = AsyncMock(
        status_code=200,
        json=lambda: mock_downstream_response_data.model_dump()
    )

    payload = GatewayAnalysisRequest(analysis_type="summary", query="Test query")
    response = client.post("/analysis/", json=payload.model_dump())

    assert response.status_code == 200
    json_response = response.json()
    
    expected_gateway_response = GatewayResponse(
        status="success",
        message="Analysis request processed successfully.",
        data=mock_downstream_response_data
    )
    assert json_response == expected_gateway_response.model_dump()
    expected_url = f"{settings.ANALYSIS_ENGINE_URL}/analyze"
    mock_client_instance.post.assert_called_once_with(expected_url, json=payload.model_dump())


@pytest.mark.asyncio
async def test_route_analysis_downstream_connection_error(client: TestClient, mock_app_http_client: AsyncMock):
    mock_app_http_client.post.side_effect = httpx.RequestError("Mocked connection error", request=AsyncMock(spec=httpx.Request))

    payload = GatewayAnalysisRequest(analysis_type="summary", query="Test query")
    response = client.post("/analysis/", json=payload.model_dump())

    assert response.status_code == 503 # Service Unavailable
    json_response = response.json()
    assert "Analysis Engine service unavailable" in json_response["detail"]


@pytest.mark.asyncio
async def test_route_analysis_downstream_http_error(client: TestClient, mock_app_http_client: AsyncMock):
    downstream_error_response_mock = AsyncMock(spec=httpx.Response)
    downstream_error_response_mock.status_code = 500
    downstream_error_response_mock.json.return_value = {"detail": "Mocked downstream server error"}
    
    http_status_error = httpx.HTTPStatusError(
        "Mocked 500 error",
        request=AsyncMock(spec=httpx.Request),
        response=downstream_error_response_mock 
    )
    downstream_error_response_mock.raise_for_status = AsyncMock(side_effect=http_status_error)
    mock_app_http_client.post.return_value = downstream_error_response_mock
    
    payload = GatewayAnalysisRequest(analysis_type="summary", query="Test query")
    response = client.post("/analysis/", json=payload.model_dump())

    assert response.status_code == 502 # Bad Gateway
    json_response = response.json()
    assert "Error communicating with Analysis Engine" in json_response["detail"]
    assert "status_code=500" in json_response["detail"]
    assert "Mocked downstream server error" in json_response["detail"]


# --- Test /documents/fetch Route (POST) ---
@pytest.mark.asyncio
async def test_route_documents_fetch_success(client: TestClient, mock_app_http_client: AsyncMock):
    mock_paper = PaperMetadata(id="arxiv:1234.5678", title="Test Paper", source="arxiv")
    mock_downstream_response_data = [mock_paper] # Paper Fetcher returns a list
    
    mock_app_http_client.post.return_value = AsyncMock(
        status_code=200,
        json=lambda: [p.model_dump() for p in mock_downstream_response_data]
    )

    payload = GatewayFetchRequest(query="test query", sources=["arxiv"])
    response = client.post("/documents/fetch", json=payload.model_dump())

    assert response.status_code == 200
    json_response = response.json()
    
    expected_gateway_response = GatewayResponse(
        status="success",
        message="Fetch request processed successfully.",
        data=[p.model_dump() for p in mock_downstream_response_data] # Ensure data matches format
    )
    assert json_response == expected_gateway_response.model_dump()
    expected_url = f"{settings.PAPER_FETCHER_URL}/fetch"
    mock_app_http_client.post.assert_called_once_with(expected_url, json=payload.model_dump())

@pytest.mark.asyncio
async def test_route_documents_fetch_connection_error(client: TestClient, mock_app_http_client: AsyncMock):
    mock_app_http_client.post.side_effect = httpx.RequestError("Mocked connection error", request=AsyncMock(spec=httpx.Request))
    payload = GatewayFetchRequest(query="test query", sources=["arxiv"])
    response = client.post("/documents/fetch", json=payload.model_dump())
    assert response.status_code == 503
    assert "Paper Fetcher service unavailable" in response.json()["detail"]

@pytest.mark.asyncio
async def test_route_documents_fetch_http_error(client: TestClient, mock_app_http_client: AsyncMock):
    downstream_error_response_mock = AsyncMock(spec=httpx.Response)
    downstream_error_response_mock.status_code = 500
    downstream_error_response_mock.json.return_value = {"detail": "Mocked fetcher error"}
    http_status_error = httpx.HTTPStatusError("Mocked 500 error", request=AsyncMock(), response=downstream_error_response_mock)
    downstream_error_response_mock.raise_for_status = AsyncMock(side_effect=http_status_error)
    mock_app_http_client.post.return_value = downstream_error_response_mock

    payload = GatewayFetchRequest(query="test query", sources=["arxiv"])
    response = client.post("/documents/fetch", json=payload.model_dump())
    assert response.status_code == 502
    json_response = response.json()
    assert "Error communicating with Paper Fetcher" in json_response["detail"]
    assert "status_code=500" in json_response["detail"]
    assert "Mocked fetcher error" in json_response["detail"]


# --- Test /documents/process Route (POST) ---
@pytest.mark.asyncio
async def test_route_documents_process_success(client: TestClient, mock_app_http_client: AsyncMock):
    # Document Processor usually returns 202 Accepted with a simple message
    mock_downstream_response_data = ProcessResponse(message="Processing started", paper_id="test_paper_id")
    
    mock_app_http_client.post.return_value = AsyncMock(
        status_code=202, # Downstream returns 202
        json=lambda: mock_downstream_response_data.model_dump()
    )

    payload = GatewayProcessRequest(paper_id="test_paper_id")
    response = client.post("/documents/process", json=payload.model_dump())

    # API Gateway forwards the 202 status and the response data
    assert response.status_code == 202 
    json_response = response.json()
    
    # Gateway wraps this in its own GatewayResponse structure if downstream is 200/201
    # For 202, the current gateway code directly returns the downstream response.
    # Let's adjust the test to reflect the new gateway logic for 202 from downstream:
    # The gateway should return its own 202 with a specific message.
    expected_gateway_response = GatewayResponse(
        status="accepted", # Or "success" depending on chosen gateway logic for 202
        message="Document processing request accepted by Document Processor.",
        data=mock_downstream_response_data.model_dump() 
    )
    # Re-evaluating gateway logic: if downstream returns 202, gateway should also return 202
    # and the body should be what the gateway constructs for this scenario.
    # The current main.py returns the downstream response directly for 202.
    # This test should reflect that.

    assert json_response == mock_downstream_response_data.model_dump() # Gateway forwards body for 202

    expected_url = f"{settings.DOC_PROCESSOR_URL}/process"
    mock_app_http_client.post.assert_called_once_with(expected_url, json=payload.model_dump())


@pytest.mark.asyncio
async def test_route_documents_process_connection_error(client: TestClient, mock_app_http_client: AsyncMock):
    mock_app_http_client.post.side_effect = httpx.RequestError("Mocked connection error", request=AsyncMock(spec=httpx.Request))
    payload = GatewayProcessRequest(paper_id="test_paper_id")
    response = client.post("/documents/process", json=payload.model_dump())
    assert response.status_code == 503
    assert "Document Processor service unavailable" in response.json()["detail"]

@pytest.mark.asyncio
async def test_route_documents_process_http_error(client: TestClient, mock_app_http_client: AsyncMock):
    downstream_error_response_mock = AsyncMock(spec=httpx.Response)
    downstream_error_response_mock.status_code = 500
    downstream_error_response_mock.json.return_value = {"detail": "Mocked processor error"}
    http_status_error = httpx.HTTPStatusError("Mocked 500 error", request=AsyncMock(), response=downstream_error_response_mock)
    downstream_error_response_mock.raise_for_status = AsyncMock(side_effect=http_status_error)
    mock_app_http_client.post.return_value = downstream_error_response_mock

    payload = GatewayProcessRequest(paper_id="test_paper_id")
    response = client.post("/documents/process", json=payload.model_dump())
    assert response.status_code == 502
    json_response = response.json()
    assert "Error communicating with Document Processor" in json_response["detail"]
    assert "status_code=500" in json_response["detail"]


# --- Test /documents/{paper_id} Route (GET) ---
@pytest.mark.asyncio
async def test_route_get_document_success(client: TestClient, mock_app_http_client: AsyncMock):
    mock_paper_id = "arxiv:1234.5678"
    mock_downstream_response_data = PaperMetadata(id=mock_paper_id, title="Test Paper", source="arxiv")
    
    mock_app_http_client.get.return_value = AsyncMock(
        status_code=200,
        json=lambda: mock_downstream_response_data.model_dump()
    )

    response = client.get(f"/documents/{mock_paper_id}")

    assert response.status_code == 200
    json_response = response.json()
    
    expected_gateway_response = GatewayResponse(
        status="success",
        message="Paper metadata retrieved successfully.",
        data=mock_downstream_response_data.model_dump()
    )
    assert json_response == expected_gateway_response.model_dump()
    expected_url = f"{settings.PAPER_FETCHER_URL}/paper/{mock_paper_id}"
    mock_app_http_client.get.assert_called_once_with(expected_url)

@pytest.mark.asyncio
async def test_route_get_document_not_found(client: TestClient, mock_app_http_client: AsyncMock):
    mock_paper_id = "nonexistent_paper"
    # Simulate downstream returning 404
    downstream_error_response_mock = AsyncMock(spec=httpx.Response)
    downstream_error_response_mock.status_code = 404
    downstream_error_response_mock.json.return_value = {"detail": "Paper not found in fetcher"}
    # httpx.HTTPStatusError is raised by response.raise_for_status()
    http_status_error = httpx.HTTPStatusError(
        "404 Not Found", 
        request=AsyncMock(spec=httpx.Request), 
        response=downstream_error_response_mock
    )
    downstream_error_response_mock.raise_for_status = AsyncMock(side_effect=http_status_error)
    mock_app_http_client.get.return_value = downstream_error_response_mock

    response = client.get(f"/documents/{mock_paper_id}")

    assert response.status_code == 404 # Gateway should forward the 404
    json_response = response.json()
    assert "Paper not found" in json_response["detail"] # Gateway's own message for 404
    assert "Paper not found in fetcher" in json_response["detail"] # Including downstream detail

@pytest.mark.asyncio
async def test_route_get_document_connection_error(client: TestClient, mock_app_http_client: AsyncMock):
    mock_paper_id = "any_paper_id"
    mock_app_http_client.get.side_effect = httpx.RequestError("Mocked connection error", request=AsyncMock(spec=httpx.Request))
    response = client.get(f"/documents/{mock_paper_id}")
    assert response.status_code == 503
    assert "Paper Fetcher service unavailable" in response.json()["detail"]

@pytest.mark.asyncio
async def test_route_get_document_http_error(client: TestClient, mock_app_http_client: AsyncMock):
    mock_paper_id = "any_paper_id"
    downstream_error_response_mock = AsyncMock(spec=httpx.Response)
    downstream_error_response_mock.status_code = 500
    downstream_error_response_mock.json.return_value = {"detail": "Mocked fetcher internal error"}
    http_status_error = httpx.HTTPStatusError("Mocked 500 error", request=AsyncMock(), response=downstream_error_response_mock)
    downstream_error_response_mock.raise_for_status = AsyncMock(side_effect=http_status_error)
    mock_app_http_client.get.return_value = downstream_error_response_mock

    response = client.get(f"/documents/{mock_paper_id}")
    assert response.status_code == 502
    json_response = response.json()
    assert "Error communicating with Paper Fetcher" in json_response["detail"]
    assert "status_code=500" in json_response["detail"]


# --- Test /search/ Route (POST) ---
@pytest.mark.asyncio
async def test_route_search_success(client: TestClient, mock_app_http_client: AsyncMock):
    mock_search_item = SearchResultItem(chunk_id="chunk1", paper_id="paper1", text_content="content", similarity_score=0.9)
    mock_downstream_response_data = SearchResponse(results=[mock_search_item])
    
    mock_app_http_client.post.return_value = AsyncMock(
        status_code=200,
        json=lambda: mock_downstream_response_data.model_dump()
    )

    payload = GatewaySearchRequest(query_text="search this")
    response = client.post("/search/", json=payload.model_dump())

    assert response.status_code == 200
    json_response = response.json()
    
    expected_gateway_response = GatewayResponse(
        status="success",
        message="Search request processed successfully.",
        data=mock_downstream_response_data.model_dump()
    )
    assert json_response == expected_gateway_response.model_dump()
    expected_url = f"{settings.VECTOR_SERVICE_URL}/search"
    mock_app_http_client.post.assert_called_once_with(expected_url, json=payload.model_dump())

@pytest.mark.asyncio
async def test_route_search_connection_error(client: TestClient, mock_app_http_client: AsyncMock):
    mock_app_http_client.post.side_effect = httpx.RequestError("Mocked connection error", request=AsyncMock(spec=httpx.Request))
    payload = GatewaySearchRequest(query_text="search this")
    response = client.post("/search/", json=payload.model_dump())
    assert response.status_code == 503
    assert "Vector Service unavailable" in response.json()["detail"]

@pytest.mark.asyncio
async def test_route_search_http_error(client: TestClient, mock_app_http_client: AsyncMock):
    downstream_error_response_mock = AsyncMock(spec=httpx.Response)
    downstream_error_response_mock.status_code = 500
    downstream_error_response_mock.json.return_value = {"detail": "Mocked vector service error"}
    http_status_error = httpx.HTTPStatusError("Mocked 500 error", request=AsyncMock(), response=downstream_error_response_mock)
    downstream_error_response_mock.raise_for_status = AsyncMock(side_effect=http_status_error)
    mock_app_http_client.post.return_value = downstream_error_response_mock

    payload = GatewaySearchRequest(query_text="search this")
    response = client.post("/search/", json=payload.model_dump())
    assert response.status_code == 502
    json_response = response.json()
    assert "Error communicating with Vector Service" in json_response["detail"]
    assert "status_code=500" in json_response["detail"]
```
