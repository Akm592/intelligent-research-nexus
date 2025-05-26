# Service: Paper Fetcher

## Purpose

The Paper Fetcher service is responsible for retrieving academic paper metadata and PDF URLs from external sources like arXiv and Semantic Scholar. It then stores this information in the Supabase database.

## Key Functionalities

-   Searches academic APIs (arXiv, Semantic Scholar - currently simulated for S2) based on user queries.
-   Parses results to extract metadata (title, authors, abstract, publication date, source, URLs, keywords).
-   Specifically attempts to get both the abstract page URL and a direct PDF download URL for each paper.
-   Standardizes paper IDs (e.g., `arxiv:YYMM.NNNNNvX`).
-   Saves or updates paper metadata in the `papers` table in Supabase (`crud.save_paper_metadata`).
-   Provides an endpoint to retrieve metadata for a specific paper by its ID (`crud.get_paper_metadata`).
-   Provides an endpoint to retrieve metadata for multiple papers by their IDs (`crud.get_papers_by_ids`).

## API Endpoints

### `POST /fetch`

-   **Description:** Initiates a search for academic papers from configured sources.
-   **Request Body:** `FetchRequest`
    ```json
    {
      "query": "string (e.g., 'quantum computing', '1903.08050v1')",
      "sources": "list[string] (e.g., ['arxiv', 'semantic_scholar'])",
      "max_results": "integer (optional, default 10 per source)"
    }
    ```
-   **Response:** `list[PaperMetadata]` (A list of `PaperMetadata` objects for the fetched and saved papers).
-   **Details:**
    -   Internally calls `logic.search_academic_sources` to query external APIs (like arXiv) and parse their responses.
    -   For each valid paper found, it calls `crud.save_paper_metadata` to store or update the paper's information in the Supabase `papers` table.

### `GET /paper/{paper_id}`

-   **Description:** Retrieves metadata for a specific paper that has been previously fetched and stored in the database.
-   **Path Parameter:**
    -   `paper_id`: `string` (The unique identifier of the paper, e.g., `arxiv:2305.12345v1`).
-   **Response:** `PaperMetadata` (The metadata object for the requested paper) or an HTTP 404 error if not found.
-   **Details:**
    -   Calls `crud.get_paper_metadata` to retrieve the paper's details from the Supabase `papers` table.
    -   *(Note: The `main.py` file might show a simplified or dummy implementation for this endpoint in some prototype versions, but the `crud.py` module contains the actual database interaction logic.)*

### `POST /papers_by_ids`

-   **Description:** Retrieves metadata for a list of papers by their IDs.
-   **Request Body:** `list[string]` (A list of paper IDs)
    ```json
    [
        "arxiv:2101.00001v1",
        "doi:10.1234/some.journal.paper"
    ]
    ```
-   **Response:** `list[PaperMetadata]` (A list of `PaperMetadata` objects for the requested papers).
-   **Details:**
    -   Calls `crud.get_papers_by_ids` to retrieve the papers' details from the Supabase `papers` table.

## Dependencies

### Core Module

-   `core.config.Settings`: Used indirectly for logging configurations and potentially for global settings if needed.
-   `core.models.PaperMetadata`: The Pydantic model for representing academic paper metadata.
-   `core.models.FetchRequest`: The Pydantic model for validating `/fetch` request payloads.
-   `core.supabase_client.get_supabase_client`: Utility function to obtain an initialized Supabase client instance for database operations.

### Libraries

-   `fastapi`: The web framework used to build the API.
-   `uvicorn`: The ASGI server for running the FastAPI application.
-   `arxiv`: The Python client library for interacting with the arXiv API.
-   `httpx`: Used for making asynchronous HTTP requests to external APIs (e.g., Semantic Scholar, though its direct use might be part of a more complete implementation).
-   `supabase-py`: The official Python client library for Supabase, used for all database interactions.
-   `python-dotenv`: For managing environment variables (loaded via `core.config`).
-   `pydantic`: For data validation and settings management (via `core.models` and `core.config`).

## Configuration

The service relies on the following environment variables, typically managed by `core.config.Settings` and loaded from a `.env` file:

-   `SUPABASE_URL`: The URL of your Supabase project.
-   `SUPABASE_KEY` (or `SUPABASE_SERVICE_KEY`): The API key (preferably the service role key) for authenticating with Supabase.
-   Standard logging configurations (e.g., `LOG_LEVEL`) inherited from `core.config`.

## Database Interaction

The Paper Fetcher service interacts primarily with the `papers` table in the Supabase PostgreSQL database.

-   **Table:** `papers`
    -   Stores metadata for each academic paper, including its ID, title, authors, abstract, source URLs, PDF URLs, publication date, keywords, etc.
-   **`crud.save_paper_metadata(db: Client, paper: PaperMetadata)`:**
    -   Upserts paper data into the `papers` table. If a paper with the same ID already exists, its metadata is updated; otherwise, a new record is created.
-   **`crud.get_paper_metadata(db: Client, paper_id: str) -> PaperMetadata | None`:**
    -   Selects and returns a single paper's metadata from the `papers` table based on its unique `paper_id`.
-   **`crud.get_papers_by_ids(db: Client, paper_ids: list[str]) -> list[PaperMetadata]`:**
    -   Selects and returns a list of paper metadata objects from the `papers` table based on a provided list of `paper_id`s.
