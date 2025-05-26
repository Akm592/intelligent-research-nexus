# Service: UI Service

## Purpose

The UI Service provides a web-based graphical user interface (GUI) for interacting with the Intelligent Research Nexus (IRN) system. It allows users to easily upload documents, fetch papers, perform semantic searches, and request various analyses without needing to interact directly with the API.

## Key Features (Implemented via Gradio Interface)

-   **PDF Upload & Processing:**
    -   Allows users to upload PDF files.
    -   Stores the uploaded PDF in Supabase Storage (configured by `settings.DOC_STORAGE_BUCKET`).
    -   Triggers the document processing pipeline by calling the API Gateway's `/documents/process` endpoint. It provides a `paper_id` (generated as `upload:<filename>_<uuid>`), the `bucket_name`, and the `object_name` (path in the bucket) for the uploaded file.
-   **Paper Fetching:**
    -   Provides an interface to query external academic sources (e.g., arXiv) by sending requests to the API Gateway's `/documents/fetch` endpoint.
    -   Displays titles and IDs of fetched papers.
    -   Allows users to select a fetched paper from the results or manually enter a paper ID to initiate its processing.
    -   When processing a paper selected from fetch results, it attempts to pass the `pdf_url` (if available in the fetched metadata) as the `source_url` to the API Gateway's `/documents/process` endpoint.
-   **Semantic Search:**
    -   Offers a search bar to query processed documents by sending requests to the API Gateway's `/search/` endpoint.
    -   Displays search results, including the chunk ID, paper ID, similarity score, and the text snippet of the matching chunk.
-   **AI Analysis:**
    -   Allows users to select an analysis type (e.g., summary, question answering, comparison) and specify one or more paper IDs for context.
    -   Includes a field for an optional query, which is particularly relevant for "Question Answering" (`qa`) or other query-focused analysis types.
    -   Sends requests to the API Gateway's `/analysis/` endpoint.
    -   Displays the generated analysis text and a list of any cited source chunk IDs.
-   **State Management:**
    -   Uses Gradio's `gr.State` component to temporarily store metadata about fetched papers (specifically `id`, `url`, and `pdf_url`). This allows the `pdf_url` to be easily accessed and passed to the processing request if the user chooses to process a paper from the fetch results.

## Technical Details

-   The user interface is built using **Gradio**.
-   This Gradio application is mounted within a **FastAPI** application, allowing it to be served as a web service.
-   The UI Service interacts exclusively with the **API Gateway** for all backend operations (fetching, processing, searching, analyzing).
-   It uses an `httpx.AsyncClient` for making asynchronous HTTP requests to the API Gateway.

## Running the UI

-   The service runs a FastAPI server, typically started with a command like `uvicorn services.ui_service.app.main:app --reload --port 7860`.
-   The Gradio interface is then accessible via a web browser, usually at the `/ui` path (e.g., `http://localhost:7860/ui`).

## Dependencies

### Core Module

-   `core.config.Settings`: Used to load essential configurations:
    -   `API_GATEWAY_URL`: The base URL of the API Gateway service.
    -   `DOC_STORAGE_BUCKET`: The name of the Supabase Storage bucket where uploaded PDFs are stored.
    -   Supabase credentials (`SUPABASE_URL`, `SUPABASE_KEY`, `SUPABASE_SERVICE_KEY`) for direct file uploads.
-   `core.models`: Provides Pydantic models for structuring request payloads sent to the API Gateway (e.g., `GatewayFetchRequest`, `GatewayProcessRequest`, `GatewaySearchRequest`, `GatewayAnalysisRequest`).
-   `core.supabase_client.get_supabase_storage_client`: Used to obtain an initialized Supabase storage client (authenticated with the service key) for directly uploading PDF files to Supabase Storage.

### Services (Indirectly, via API Gateway)

-   The UI Service communicates directly only with the **API Gateway** (whose URL is specified by `settings.API_GATEWAY_URL`). All interactions with other backend services (Paper Fetcher, Document Processor, etc.) are routed through the API Gateway.

### Libraries

-   `gradio`: The primary library for building the web interface components.
-   `fastapi`: The web framework used to serve the Gradio application.
-   `uvicorn`: The ASGI server for running the FastAPI application.
-   `httpx`: An asynchronous HTTP client library used for making requests to the API Gateway.
-   `python-multipart`: Required by FastAPI/Gradio for handling file uploads.

## Configuration

The UI Service relies on the following environment variables, typically managed by `core.config.Settings` and loaded from a `.env` file:

-   `API_GATEWAY_URL`: The complete base URL of the API Gateway service (e.g., `http://localhost:8000`).
-   `DOC_STORAGE_BUCKET`: The name of the Supabase Storage bucket designated for storing PDF files uploaded directly by users through the UI.
-   `SUPABASE_URL`: The URL of your Supabase project.
-   `SUPABASE_KEY`: The public `anon` key for your Supabase project (can be used here as Gradio runs client-side logic that might interact with Supabase, though uploads use the service key).
-   `SUPABASE_SERVICE_KEY`: The `service_role` key for your Supabase project, used by the backend of the UI service to authenticate with `core.supabase_client` for uploading files directly to Supabase Storage.
-   Standard logging configurations (e.g., `LOG_LEVEL`) inherited from `core.config`.
