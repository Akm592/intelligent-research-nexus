# Service: API Gateway

## Purpose

The API Gateway is the primary entry point for all external requests to the Intelligent Research Nexus (IRN) system. It handles incoming traffic, authenticates requests (placeholder), applies rate limiting (placeholder), and routes them to the appropriate backend microservices.

## Key Functionalities

-   Receives all HTTP requests from clients (e.g., UI, external applications).
-   Provides a unified API interface for disparate backend services.
-   Routes requests to:
    -   Paper Fetcher Service (`/documents/fetch`, `/documents/{paper_id}`)
    -   Document Processor Service (`/documents/process`)
    -   Vector Service (`/search/`)
    -   Analysis Engine (`/analysis/`)
-   Implements basic (placeholder) API key authentication (`X-API-Key` header).
-   Implements basic (placeholder) IP-based rate limiting.
-   Manages an HTTP client pool (`httpx.AsyncClient`) for efficient communication with backend services.
-   Provides a health check endpoint (`/health`).

## API Endpoints

### Meta

-   **`GET /`**
    -   **Description:** Welcome endpoint providing a basic greeting message.
    -   **Response:** `{"message": "Welcome to the IRN API Gateway"}`

-   **`GET /health`**
    -   **Description:** Health check for the API Gateway.
    -   **Response:** `{"status": "healthy", "service": "API Gateway"}`

### Documents

-   **`POST /documents/fetch`**
    -   **Description:** Routes requests to the Paper Fetcher Service to fetch new papers based on a query or identifiers.
    -   **Request Body:** `GatewayFetchRequest`
        ```json
        {
          "query": "string (e.g., DOI, arXiv ID, keywords)",
          "max_results": "integer (optional, default 10)"
        }
        ```
    -   **Response:** `GatewayResponse` (containing data from Paper Fetcher, typically a list of `PaperMetadata` or a confirmation message).

-   **`POST /documents/process`**
    -   **Description:** Routes requests to the Document Processor Service to initiate processing of a fetched paper (e.g., chunking, embedding).
    -   **Request Body:** `GatewayProcessRequest`
        ```json
        {
          "paper_id": "string (UUID of the paper)",
          "bucket_name": "string (optional, if PDF is in cloud storage)",
          "object_name": "string (optional, if PDF is in cloud storage)",
          "source_url": "string (optional, direct URL to PDF)"
        }
        ```
    -   **Response:** `GatewayResponse` (confirming that the processing request has been accepted by the Document Processor).

-   **`GET /documents/{paper_id}`**
    -   **Description:** Routes requests to the Paper Fetcher Service to retrieve metadata for a specific paper by its ID.
    -   **Path Parameter:** `paper_id` (string, UUID of the paper)
    -   **Response:** `GatewayResponse` (containing `PaperMetadata` for the requested paper).

### Search

-   **`POST /search/`**
    -   **Description:** Routes requests to the Vector Service to perform semantic search over processed document chunks.
    -   **Request Body:** `GatewaySearchRequest`
        ```json
        {
          "query": "string (search query)",
          "top_k": "integer (optional, default 5, number of results to return)"
        }
        ```
    -   **Response:** `GatewayResponse` (containing a list of search results, typically relevant document chunks).

### Analysis

-   **`POST /analysis/`**
    -   **Description:** Routes requests to the Analysis Engine to perform various analyses (e.g., RAG, summarization) on documents.
    -   **Request Body:** `GatewayAnalysisRequest`
        ```json
        {
          "query": "string (optional, for query-based analysis like RAG)",
          "paper_ids": "list[string] (optional, list of paper UUIDs for context)",
          "analysis_type": "string (e.g., 'rag_summary', 'keyword_extraction')"
        }
        ```
    -   **Response:** `GatewayResponse` (containing the results of the analysis, e.g., generated text, extracted keywords).

## Dependencies

### Core Module

-   `core.config.Settings`: For loading service URLs (downstream services), API keys, and other operational configurations.
-   `core.models`: For request/response validation and data structuring using Pydantic models:
    -   `GatewayResponse`: Standardized response wrapper.
    -   `GatewayFetchRequest`: Schema for `/documents/fetch` requests.
    -   `GatewayProcessRequest`: Schema for `/documents/process` requests.
    -   `GatewaySearchRequest`: Schema for `/search/` requests.
    -   `GatewayAnalysisRequest`: Schema for `/analysis/` requests.

### Services (Downstream)

The API Gateway communicates with the following backend services, whose URLs are configured via `core.config.Settings`:

-   Paper Fetcher Service (at `settings.PAPER_FETCHER_URL`)
-   Document Processor Service (at `settings.DOC_PROCESSOR_URL`)
-   Vector Service (at `settings.VECTOR_SERVICE_URL`)
-   Analysis Engine (at `settings.ANALYSIS_ENGINE_URL`)

### Libraries

-   `fastapi`: The web framework used to build the API Gateway.
-   `uvicorn`: The ASGI server used to run the FastAPI application.
-   `httpx`: An asynchronous HTTP client for making requests to downstream services.
-   `python-dotenv`: For managing environment variables (via `core.config`).
-   `pydantic` & `pydantic-settings`: For data validation and configuration management (via `core.config` and `core.models`).

## Configuration

The following environment variables are used by the API Gateway, typically loaded from a `.env` file into `core.config.Settings`:

-   `API_GATEWAY_URL`: The base URL where the API Gateway itself is running (e.g., `http://localhost:8000`). Although this is the service's own URL, it's part of the shared settings.
-   `PAPER_FETCHER_URL`: URL of the Paper Fetcher service.
-   `DOC_PROCESSOR_URL`: URL of the Document Processor service.
-   `ANALYSIS_ENGINE_URL`: URL of the Analysis Engine service.
-   `VECTOR_SERVICE_URL`: URL of the Vector Service.
-   `API_GATEWAY_KEY`: An optional API key that clients must provide in the `X-API-Key` header to access the gateway. If this variable is not set or is empty, the API key authentication is skipped.
-   Standard logging configurations (e.g., `LOG_LEVEL`) inherited from `core.config`.
