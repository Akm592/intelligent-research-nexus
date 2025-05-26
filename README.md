# Intelligent Research Nexus (IRN) - Prototype v1.0.0

This is the initial prototype code for the IRN system, designed for local testing and development. It sets up the core microservices architecture using FastAPI, integrates with Google Gemini for AI capabilities, and uses Supabase for PostgreSQL metadata and pgvector-based similarity search.

**Features (Prototype Level):**

*   **API Gateway:** Entry point for all requests.
*   **Paper Fetcher:** Simulates fetching paper metadata from academic sources and saves to Supabase.
*   **Document Processor:** Simulates parsing/chunking a local sample PDF and triggers embedding via the Vector Service.
*   **Vector Service:** Generates embeddings using Gemini and stores/searches them in Supabase (pgvector) via RPC.
*   **Analysis Engine:** Performs basic RAG (Retrieve-Augment-Generate) using context from Vector Service and analysis generation via Gemini.
*   **Core Module:** Shared configuration, data models, Gemini client, and Supabase client utility.

## Project Structure

The project is organized into the following main directories:

-   `core/`: Contains shared core modules essential for the functioning of multiple services, including configuration management, AI client integration, database client, and common data models.
-   `services/`: Houses the individual microservices, each with its own FastAPI application, specific logic, and requirements.
-   `docs/`: Includes project documentation, such as the system architecture (`ARCHITECTURE.md`) and this README.
-   `data/`: Used for sample data and local file storage during development (e.g., `sample_paper.pdf` for the Document Processor).
-   `infra/`: Intended for Infrastructure-as-Code (IaC) scripts (e.g., Docker Compose files, Kubernetes manifests). (Currently may contain initial Docker setup or be a placeholder for future expansion).
-   `scripts/`: Contains utility scripts for development, deployment, data migration, or maintenance tasks.
-   `tests/`: Home for automated tests, including unit, integration, and end-to-end tests for the project.

## Core Components

The `core/` directory provides foundational modules used across various services:

-   `core/config.py`: Manages all application settings, API keys (like Gemini and Supabase), database URLs, service ports, and other environment-specific variables using Pydantic Settings.
-   `core/gemini_client.py`: Provides a client for interacting with Google Gemini models, facilitating tasks like text generation and creating vector embeddings.
-   `core/models.py`: Defines Pydantic data models for shared entities (e.g., `PaperMetadata`, `DocumentChunk`) and common API request/response schemas, ensuring data consistency across services.
-   `core/supabase_client.py`: Handles the connection and interactions with the Supabase PostgreSQL database, including its pgvector extension for similarity searches.
-   `core/storage.py`: Intended for interactions with file storage systems (e.g., Supabase Storage, Google Cloud Storage, AWS S3). (Currently may be a placeholder or have basic implementation).
-   `core/utils.py`: Placeholder for common utility functions, helper classes, or constants that can be shared across the project.

## Services Overview

The IRN system is composed of the following microservices:

-   `api_gateway`: The single entry point for all client requests, routing them to appropriate backend services and handling cross-cutting concerns.
-   `paper_fetcher`: Fetches academic paper metadata and potentially PDFs from external sources like arXiv, Semantic Scholar, or CrossRef.
-   `document_processor`: Parses documents (e.g., PDFs) into manageable text chunks and coordinates their embedding via the Vector Service.
-   `vector_service`: Generates vector embeddings for text chunks using AI models (e.g., Gemini) and performs similarity searches against them in the database.
-   `analysis_engine`: Orchestrates Retrieve-Augment-Generate (RAG) pipelines, using context from the Vector Service and AI models to answer queries or generate summaries.
-   `kg_service`: (Placeholder/Future) Intended for constructing and querying a knowledge graph from processed documents and their relationships.
-   `ui_service`: (Placeholder/Future) Aims to provide a user interface for interacting with the IRN system's functionalities.
-   `user_profile_service`: (Placeholder/Future) Designed to manage user accounts, preferences, history, and personalization features.

## Data Models

Key data structures defined in `core/models.py` include:

-   `PaperMetadata`: Represents the metadata for an academic paper, such as its unique ID, title, list of authors, abstract, source (e.g., arXiv, DOI), URLs (to PDF, source page), and processing status.
-   `DocumentChunk`: Represents a segment of text extracted from a paper, containing a chunk ID, the ID of the paper it belongs to, the actual text content, any relevant metadata about the chunk (e.g., section, page number), and optionally, its vector embedding.

Other Pydantic models are also defined for validating API request and response schemas across the different microservices, ensuring standardized communication.

## Setup Instructions

1.  **Prerequisites:**
    *   Python 3.10+
    *   Git (optional, for cloning the repository)
    *   Access to a Supabase project (for database and pgvector capabilities)
    *   Google Gemini API Key (for AI-powered text generation and embeddings)

2.  **Clone Repository (Optional):**
    ```bash
    git clone <your-repo-url>
    cd irn_prototype_v1.0.0
    ```
    Alternatively, download and extract the source code into an `irn_prototype_v1.0.0` directory.

3.  **Supabase Setup:**
    *   In your Supabase project dashboard, navigate to the **SQL Editor**.
    *   Locate the SQL setup script (e.g., in `infra/supabase_schema.sql` or provided separately). This script typically creates tables like `papers` and `chunks`, enables the `vector` extension, creates indexes, and sets up functions like `match_document_chunks`.
    *   Copy the entire content of this SQL script.
    *   Paste it into the Supabase SQL Editor and **Run** it. You might see warnings about destructive operations (like `DROP TABLE IF EXISTS` or `CREATE OR REPLACE FUNCTION`) – confirm and run, as this is intended to set up or reset the schema correctly.

4.  **Create Sample PDF (for initial testing):**
    *   Place a valid PDF file inside the `data/` directory and name it `sample_paper.pdf`. This file is used by the `DocumentProcessor` service in its current placeholder implementation.

5.  **Create Virtual Environment:**
    It's highly recommended to use a virtual environment for managing project dependencies.
    ```bash
    python -m venv venv
    ```
    Activate it:
    *   macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
    *   Windows (Command Prompt/PowerShell):
        ```bash
        .\venv\Scripts\activate
        ```

6.  **Install Dependencies:**
    Each service has its own `requirements.txt`. Install them as follows:
    ```bash
    pip install -r services/api_gateway/requirements.txt
    pip install -r services/paper_fetcher/requirements.txt
    pip install -r services/document_processor/requirements.txt
    pip install -r services/vector_service/requirements.txt
    pip install -r services/analysis_engine/requirements.txt
    ```
    The `core` module dependencies are generally included by the services that use them. However, if you need to install them explicitly or for standalone core module development, you can use:
    ```bash
    pip install google-generativeai>=0.4.0 pydantic-settings python-dotenv supabase psycopg2-binary
    ```
    *(Note: `psycopg2-binary` is often needed for Supabase Python client).*

7.  **Configure Environment:**
    *   Copy the example environment file `.env.example` to a new file named `.env` in the project root:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file and provide your actual credentials and configuration:
        *   `GEMINI_API_KEY`: Your Google Gemini API key.
        *   `SUPABASE_URL`: Your Supabase project URL.
        *   `SUPABASE_KEY`: Your Supabase project `service_role` key (do not use the `anon` key for backend services).
    *   Ensure `EMBEDDING_DIM` in `.env` (e.g., `EMBEDDING_DIM=768`) matches your chosen Gemini embedding model's output dimensionality and the SQL setup in Supabase (especially the vector column and search functions).

## Running the Services

For local development, each microservice needs to be run in its own terminal window. Ensure your virtual environment (`venv`) is activated in each terminal and that you are in the project root directory (`irn_prototype_v1.0.0/`).

1.  **Terminal 1 (API Gateway):**
    ```bash
    uvicorn services.api_gateway.app.main:app --reload --port 8000
    ```
2.  **Terminal 2 (Paper Fetcher):**
    ```bash
    uvicorn services.paper_fetcher.app.main:app --reload --port 8001
    ```
3.  **Terminal 3 (Document Processor):**
    ```bash
    uvicorn services.document_processor.app.main:app --reload --port 8002
    ```
4.  **Terminal 4 (Vector Service):**
    ```bash
    uvicorn services.vector_service.app.main:app --reload --port 8004
    ```
5.  **Terminal 5 (Analysis Engine):**
    ```bash
    uvicorn services.analysis_engine.app.main:app --reload --port 8003
    ```

**Note on Running Services:** The `uvicorn --reload` command is suitable for development as it automatically reloads the server when code changes are detected. For production deployments, you would typically use a more robust WSGI server like Gunicorn or Uvicorn managed by a process supervisor (e.g., systemd, Supervisor) and often run multiple worker processes.

**Basic Testing (using curl or HTTP client):**

Once all services are running, you can test the API Gateway endpoints. (Refer to previous "how to run" guides or API documentation for specific `curl` examples for endpoints like `/health`, `/documents/fetch`, `/documents/process`, `/search/`, and `/analysis/`, all targeting the API Gateway at `http://localhost:8000`).

### Testing

This project uses [pytest](https://docs.pytest.org/) for unit and integration testing.

**1. Install Test Dependencies:**

Ensure you have installed all dependencies, including testing-specific ones:
```bash
pip install -r requirements.txt
```
(If you have a separate `requirements-dev.txt` or similar, adjust the command accordingly. Based on the provided `requirements.txt`, all dependencies are in one file.)

**2. Running Tests:**

To run all tests, navigate to the project root directory and execute:
```bash
python -m pytest
```

**3. Test Coverage:**

To run tests and generate a coverage report:
```bash
python -m pytest --cov=. --cov-report=html
```
This will create an `htmlcov/` directory with a detailed HTML report. Open `htmlcov/index.html` in your browser to view it.
You can also view a summary in the terminal with:
```bash
python -m pytest --cov=.
```

## Next Steps

*   Replace placeholder logic (`TODO` comments) with actual implementations (API calls, parsing libraries, storage clients).
*   Implement robust error handling and validation.
*   Develop the Knowledge Graph and User Profile services.
*   Add comprehensive tests (unit, integration).
*   Set up Docker Compose for easier multi-container management.
*   Integrate MLOps components (monitoring, experiment tracking).