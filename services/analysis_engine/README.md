# Service: Analysis Engine

## Purpose

The Analysis Engine service orchestrates the generation of analytical insights from documents. It employs a Retrieve-Augment-Generate (RAG) pattern by first fetching relevant context from the Vector Service and then using a Large Language Model (LLM) via the Gemini API to perform tasks like question answering, summarization, or comparisons.

## Key Functionalities

### Context Retrieval (RAG - Retrieve)
-   Receives analysis requests specifying `analysis_type`, optional `paper_ids`, and an optional `query`.
-   Constructs a search query for the Vector Service based on the input. If a `query` is provided in the analysis request, that query is used directly for the Vector Service search. If `paper_ids` are provided without a specific query, the engine might form a generic query or use the paper IDs to filter context from the Vector Service.
-   Calls the Vector Service's `/search` endpoint to retrieve relevant text chunks (context) from processed documents.
-   Handles cases where no context is found or if the Vector Service call fails gracefully, potentially informing the user or proceeding with generation without external context if appropriate for the `analysis_type`.

### Prompt Construction (RAG - Augment)
-   Builds a detailed prompt for the Gemini LLM.
-   The prompt includes:
    -   The type of analysis requested (e.g., "Summary", "Question Answering", "Comparison").
    -   Target paper IDs (if any, to scope the analysis).
    -   The user's query (if any, forming the core question or topic for the LLM).
    -   The retrieved context snippets, clearly demarcated and attributed (e.g., "Context from Paper X, Chunk Y: [text of chunk]").
    -   Instructions for the LLM on how to use the context, the desired level of detail, and how to cite sources (e.g., "Answer the query based *only* on the provided context. Cite relevant context snippets using `[Source Chunk (ID: chunk_id_here)]` at the end of sentences or paragraphs they inform.").

### Analysis Generation (RAG - Generate)
-   Sends the constructed prompt to the `GeminiClient` (from `core.gemini_client`) using the `generate_text` method to get the analytical text.
-   Supports using different Gemini models (e.g., "gemini-pro" or "gemini-flash") based on configuration or request parameters (e.g., `detail_level` might influence model choice or prompt instructions).

### Response Formatting
-   Extracts cited source chunk IDs from the LLM's response using regular expressions (e.g., to find all occurrences of `[Source Chunk (ID: <uuid>)]`).
-   Returns the generated analysis text and the list of unique cited source chunk IDs.

## API Endpoints

### `POST /analyze`

-   **Description:** Performs analysis based on the provided request parameters.
-   **Request Body:** `AnalysisRequest` (defined in `core.models`)
    ```json
    {
      "query": "string (optional, e.g., 'What are the main challenges in deploying LLMs in healthcare?')",
      "paper_ids": "list[string] (optional, e.g., ['arxiv:2301.00001v1', 'doi:10.xxxx/yyyyy'])",
      "analysis_type": "string (e.g., 'rag_summary', 'question_answering', 'keyword_extraction')",
      "detail_level": "string (optional, e.g., 'concise', 'detailed', influences prompt or model choice)"
    }
    ```
-   **Response:** `AnalysisResult` (defined in `core.models`)
    ```json
    {
      "result_text": "string (The LLM-generated analysis)",
      "cited_sources": "list[string] (List of chunk_ids cited in the result_text)",
      "analysis_type": "string (The type of analysis performed)"
    }
    ```
-   **Details:**
    1.  The service receives an `AnalysisRequest`.
    2.  It calls the internal `retrieve_context` function. This function:
        -   Forms a `SearchQuery` (from `core.models`) using the `query` from the `AnalysisRequest` (if provided) and `paper_ids` (if provided, used for filtering in the Vector Service).
        -   Makes an HTTP POST request to the Vector Service's `/search` endpoint.
        -   Returns the list of relevant `SearchResultItem` objects.
    3.  A detailed prompt is constructed, incorporating the `analysis_type`, the original `query`, any `paper_ids`, and the text content from the retrieved context chunks. Instructions on how to use and cite context are included.
    4.  The `gemini_client.generate_text` method is called with this prompt.
    5.  The LLM's response text is parsed using regex to extract `[Source Chunk (ID: ...)]` citations, which are collected into the `cited_sources` list.
    6.  An `AnalysisResult` object containing the generated text, cited sources, and original analysis type is returned.

## Dependencies

### Core Module

-   `core.config.Settings`: For accessing service URLs (like `VECTOR_SERVICE_URL`), the `GEMINI_API_KEY`, and standard logging configurations.
-   `core.models`: Uses the following Pydantic models for request/response validation and data structuring:
    -   `AnalysisRequest`: For the `/analyze` endpoint request.
    -   `AnalysisResult`: For the `/analyze` endpoint response.
    -   `SearchQuery`: Used when preparing the request to the Vector Service.
    -   `SearchResponse` (and `SearchResultItem`): Used when interpreting the response from the Vector Service.
-   `core.gemini_client.GeminiClient`: The shared client for interacting with the Google Gemini API to generate the analytical text.

### Services (Downstream)

-   **Vector Service:** The Analysis Engine makes requests to the Vector Service (URL configured via `settings.VECTOR_SERVICE_URL`) to retrieve relevant text chunks for context. It specifically calls the `/search` endpoint of the Vector Service.

### Libraries

-   `fastapi`: The web framework used for building the service's API.
-   `uvicorn`: The ASGI server for running the FastAPI application (typically used during development).
-   `httpx`: An asynchronous HTTP client used for making requests to the Vector Service.
-   `pydantic`: For data validation through Pydantic models and for managing settings (via `core.config`).
-   `python-dotenv`: For managing environment variables (loaded via `core.config`).
-   `re` (built-in Python module): Used for regular expression matching to extract cited sources from the LLM's output.

## Configuration

The Analysis Engine relies on the following environment variables, typically managed by `core.config.Settings` and loaded from a `.env` file:

-   `VECTOR_SERVICE_URL`: The complete URL of the Vector Service (e.g., `http://localhost:8004`).
-   `GEMINI_API_KEY`: Required by `core.gemini_client` to authenticate with the Google Gemini API.
-   Standard logging configurations (e.g., `LOG_LEVEL`) inherited from `core.config`.
-   Implicitly, it is affected by configurations like `SEARCH_MATCH_THRESHOLD` and `EMBEDDING_DIM` (defined in `core.config.Settings` and used by the Vector Service), as these influence the quality and nature of the context retrieved, which in turn impacts the analysis.
