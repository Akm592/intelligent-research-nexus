# Intelligent Research Nexus (IRN) - Prototype v1.0.0

This is the initial prototype code for the IRN system, designed for local testing and development. It sets up the core microservices architecture using FastAPI, integrates with Google Gemini for AI capabilities, and uses Supabase for PostgreSQL metadata and pgvector-based similarity search.

**Features (Prototype Level):**

*   **API Gateway:** Entry point for all requests.
*   **Paper Fetcher:** Simulates fetching paper metadata from academic sources and saves to Supabase.
*   **Document Processor:** Simulates parsing/chunking a local sample PDF and triggers embedding via the Vector Service.
*   **Vector Service:** Generates embeddings using Gemini and stores/searches them in Supabase (pgvector) via RPC.
*   **Analysis Engine:** Performs basic RAG (Retrieve-Augment-Generate) using context from Vector Service and analysis generation via Gemini.
*   **Core Module:** Shared configuration, data models, Gemini client, and Supabase client utility.

**Setup Instructions:**

1.  **Prerequisites:**
    *   Python 3.10+
    *   Git (optional, for cloning)
    *   Access to a Supabase project
    *   Google Gemini API Key

2.  **Clone Repository (Optional):**
    ```bash
    git clone <your-repo-url>
    cd irn_prototype_v1.0.0
    ```
    Or download/extract the code into an `irn_prototype_v1.0.0` directory.

3.  **Supabase Setup:**
    *   In your Supabase project dashboard, navigate to the **SQL Editor**.
    *   Copy the entire content of the SQL setup script provided previously (the one creating `papers`, `chunks` tables, enabling `vector`, creating indexes and the `match_document_chunks` function).
    *   Paste the script into the SQL Editor and **Run** it. You will see a warning about destructive operations (`DROP TRIGGER`, `CREATE OR REPLACE`) - **confirm and run** as this is required for setup.

4.  **Create Sample PDF:**
    *   Place **any valid PDF file** inside the `data/` directory and name it `sample_paper.pdf`. This is used by the Document Processor placeholder.

5.  **Create Virtual Environment:**
    ```bash
    python -m venv venv
    # Activate it:
    # macOS/Linux:
    source venv/bin/activate
    # Windows:
    # .\venv\Scripts\activate
    ```

6.  **Install Dependencies:**
    ```bash
    pip install -r services/api_gateway/requirements.txt
    pip install -r services/paper_fetcher/requirements.txt
    pip install -r services/document_processor/requirements.txt
    pip install -r services/vector_service/requirements.txt
    pip install -r services/analysis_engine/requirements.txt
    # Install core deps explicitly if needed, though service installs should cover them
    pip install google-generativeai>=0.4.0 pydantic-settings python-dotenv
    ```

7.  **Configure Environment:**
    *   Copy `.env.example` to `.env`.
    *   Edit `.env` and add your actual `GEMINI_API_KEY`, `SUPABASE_URL`, and `SUPABASE_KEY` (use the `service_role` key).
    *   Ensure `EMBEDDING_DIM=768` (or matches your Gemini model and SQL setup).

**Running the Services:**

Open **separate terminal windows** for each service. Ensure your virtual environment is active in each terminal. Navigate to the project root directory (`irn_prototype_v1.0.0/`) in each terminal.

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

**Basic Testing (using curl):**

*(See previous "how to run" guide for curl examples for `/health`, `/documents/fetch`, `/documents/process`, `/search/`, `/analysis/` endpoints targeting `http://localhost:8000`)*

**Next Steps:**

*   Replace placeholder logic (`TODO` comments) with actual implementations (API calls, parsing libraries, storage clients).
*   Implement robust error handling and validation.
*   Develop the Knowledge Graph and User Profile services.
*   Add comprehensive tests (unit, integration).
*   Set up Docker Compose for easier multi-container management.
*   Integrate MLOps components (monitoring, experiment tracking).