# Service: Document Processor

## Purpose

The Document Processor service handles the ingestion and preparation of documents for analysis. It retrieves document content, parses it (currently focusing on PDFs), splits it into manageable text chunks, and then coordinates with the Vector Service to generate and store embeddings for these chunks.

## Key Functionalities

-   Receives requests to process a document identified by `paper_id`.
-   Retrieves document content:
    -   From Supabase Storage if `bucket_name` and `object_name` are provided.
    -   From a `source_url` if provided directly in the request.
    -   Looks up `pdf_url` or fallback `url` in the `papers` table in Supabase if only `paper_id` is given.
-   Parses PDF documents using `pdfminer.six` to extract raw text (`processing.parse_and_chunk`).
-   Splits extracted text into smaller chunks using a robust recursive splitting algorithm (`processing._recursive_split`) based on configured separators, chunk size, and overlap (`settings.PARSER_SEPARATORS`, `settings.PARSER_CHUNK_SIZE`, `settings.PARSER_CHUNK_OVERLAP`).
-   Updates the `processing_status` and `status_message` of the paper in the `papers` table in Supabase (`crud.update_paper_status`).
-   Sends the generated text chunks to the Vector Service (`/embed` endpoint) for embedding generation and storage.
-   Handles processing in a background task to allow the API to respond quickly.

## API Endpoints

### `POST /process`

-   **Description:** Initiates the processing of a document.
-   **Request Body:** `ProcessRequest`
    ```json
    {
      "paper_id": "string (UUID of the paper)",
      "source_url": "string (optional, direct URL to the document)",
      "bucket_name": "string (optional, Supabase Storage bucket name)",
      "object_name": "string (optional, Supabase Storage object name/path)"
    }
    ```
-   **Response:** HTTP 202 Accepted with a message like `{"message": "Document processing scheduled successfully."}`.
-   **Details:**
    -   Checks the current status of the paper using `crud.get_paper_status`.
    -   If the paper is in a processable state (e.g., "fetched", "failed"), it schedules the `process_and_embed_task` to run in the background.
    -   The background task (`process_and_embed_task`) performs the following steps:
        1.  Updates the paper's status to "processing" using `crud.update_paper_status`.
        2.  Fetches the document content using `processing.get_document_content`. This function tries to get content from Supabase Storage (if `bucket_name` and `object_name` are given), then from a `source_url` (if provided), and finally falls back to `pdf_url` or `url` from the paper's metadata in the database.
        3.  Parses the document and splits it into chunks using `processing.parse_and_chunk`. This yields a list of `DocumentChunk` objects.
        4.  Sends these chunks to the Vector Service's `/embed` endpoint.
        5.  Based on the success or failure of the previous steps (especially embedding), updates the paper's status to "processed", "processed_with_errors", or "failed" using `crud.update_paper_status`.

## Dependencies

### Core Module

-   `core.config.Settings`: For loading configuration parameters such as `PARSER_CHUNK_SIZE`, `PARSER_CHUNK_OVERLAP`, `PARSER_SEPARATORS`, the `VECTOR_SERVICE_URL`, Supabase credentials, and logging settings.
-   `core.models`: Uses several Pydantic models:
    -   `ProcessRequest`: For validating the `/process` request payload.
    -   `DocumentChunk`: To structure the text chunks before sending them for embedding.
    -   `EmbedRequest`: The schema for the request to the Vector Service's `/embed` endpoint.
    -   `EmbedResponse`: The expected response schema from the Vector Service.
    -   `PaperMetadata`: Used when fetching paper details from the database.
-   `core.supabase_client.get_supabase_client`, `core.supabase_client.get_supabase_storage_client`: Utilities to obtain initialized Supabase client instances for database operations and storage access.

### Services (Downstream)

-   **Vector Service:** The Document Processor sends text chunks to the Vector Service (at `settings.VECTOR_SERVICE_URL`) for embedding generation and storage. It expects the Vector Service to have an `/embed` endpoint.

### Libraries

-   `fastapi`: The web framework used to build the service's API.
-   `uvicorn`: The ASGI server for running the FastAPI application.
-   `httpx`: An asynchronous HTTP client used for making requests to the Vector Service and for downloading documents from URLs.
-   `pdfminer.six`: The library used for parsing PDF files to extract text content.
-   `supabase-py`: The official Python client library for Supabase, used for interacting with the Supabase database (e.g., `papers` table) and Supabase Storage.
-   `python-dotenv`: For managing environment variables (loaded via `core.config`).
-   `pydantic`: For data validation and settings management (via `core.models` and `core.config`).

## Configuration

The service relies on the following environment variables, typically managed by `core.config.Settings` and loaded from a `.env` file:

-   `VECTOR_SERVICE_URL`: The URL of the Vector Service where text chunks will be sent for embedding.
-   `PARSER_CHUNK_SIZE`: `int`, the target maximum size for each text chunk.
-   `PARSER_CHUNK_OVERLAP`: `int`, the number of characters to overlap between consecutive chunks.
-   `PARSER_SEPARATORS`: `list[str]`, a list of strings used as separators for recursively splitting text (e.g., `["\n\n", "\n", ". ", " ", ""]`).
-   `SUPABASE_URL`: The URL of your Supabase project (from `core.config`).
-   `SUPABASE_KEY` (or `SUPABASE_SERVICE_KEY`): The API key (preferably the service role key) for authenticating with Supabase (from `core.config`).
-   Standard logging configurations (e.g., `LOG_LEVEL`) inherited from `core.config`.

## Database Interaction

The Document Processor service interacts primarily with the `papers` table in the Supabase PostgreSQL database:

-   **Table:** `papers`
    -   `crud.update_paper_status(db: Client, paper_id: str, status: str, message: str | None = None)`: Updates the `processing_status` (e.g., "processing", "processed", "failed") and an optional `status_message` for a given `paper_id`.
    -   `crud.get_paper_status(db: Client, paper_id: str) -> str | None`: Retrieves the current `processing_status` of a paper.
    -   `processing._fetch_paper_metadata_from_db(db: Client, paper_id: str) -> PaperMetadata | None`: Fetches the full metadata for a paper, which includes `url` and `pdf_url`, used to locate the document content if not directly provided in the request.

While the Document Processor generates `DocumentChunk` objects, it does not directly write them to a `chunks` table. Instead, it sends these chunks to the Vector Service, which is responsible for their storage (likely including their embeddings) in a database (e.g., a `chunks` table with vector capabilities).
