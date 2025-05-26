# Intelligent Research Nexus (IRN) Architecture

## Overview

The Intelligent Research Nexus (IRN) is a system designed to help researchers discover, process, and analyze academic papers. It leverages a microservices architecture to handle various tasks such as fetching papers, processing documents, generating vector embeddings, performing AI-driven analysis, and managing knowledge graphs.

## Microservices Architecture

IRN is built using a microservices architecture. This approach breaks down the system into smaller, independent services that communicate with each other over well-defined APIs. This design promotes scalability, maintainability, and flexibility, allowing individual services to be developed, deployed, and scaled independently.

## API Gateway

The API Gateway serves as the single entry point for all client requests. It routes requests to the appropriate backend microservice, handles authentication and authorization, and can also perform rate limiting and request/response transformation. This simplifies client interactions and decouples clients from the internal microservice structure.

## Microservice Descriptions

### Paper Fetcher
- **Purpose:** Retrieves academic paper metadata and potentially PDF files from external sources.
- **Responsibilities:**
    - Fetches paper details (title, authors, abstract, publication date, DOI, etc.) from APIs like arXiv, Semantic Scholar, CrossRef, etc.
    - Stores fetched metadata in the Supabase database (e.g., a `Papers` table).
    - Optionally downloads PDF versions of papers and stores them in a designated storage solution (e.g., Supabase Storage).
- **Interactions:** External academic APIs, Supabase Database, Cloud Storage.

### Document Processor
- **Purpose:** Processes raw documents (e.g., PDFs) into a more usable format for analysis.
- **Responsibilities:**
    - Retrieves paper content (e.g., PDF from storage or a given URL).
    - Parses the document text.
    - Splits the text into smaller, manageable chunks (e.g., paragraphs or sections).
    - Stores chunk information (content, order, source paper ID) in the Supabase database (e.g., a `Chunks` table).
    - Sends processed chunks to the Vector Service for embedding generation.
- **Interactions:** Cloud Storage, Supabase Database, Vector Service.

### Vector Service
- **Purpose:** Generates and manages vector embeddings for text chunks.
- **Responsibilities:**
    - Receives text chunks from the Document Processor.
    - Uses a language model (e.g., Gemini) to generate vector embeddings for each chunk.
    - Stores these embeddings in the Supabase database, typically using a vector-capable extension like pgvector within the `Chunks` table or a dedicated `Embeddings` table.
    - Provides an interface to search for similar vectors given a query vector (for RAG).
- **Interactions:** Document Processor, Gemini API, Supabase Database (pgvector).

### Analysis Engine
- **Purpose:** Performs AI-powered analysis on user queries using Retrieval Augmented Generation (RAG).
- **Responsibilities:**
    - Receives user queries for analysis.
    - Communicates with the Vector Service to find document chunks relevant to the user's query.
    - Constructs a detailed prompt for a generative AI model (e.g., Gemini), including the user's query and the retrieved relevant chunks.
    - Sends the prompt to the AI model (e.g., Gemini via a GeminiClient).
    - Returns the AI-generated analysis/response to the user.
- **Interactions:** API Gateway, Vector Service, Gemini API.

### KG Service (Knowledge Graph Service)
- **Purpose:** Builds, maintains, and queries a knowledge graph from the processed academic papers.
- **Responsibilities:** (To be expanded)
    - Extracts entities and relationships from text chunks or metadata.
    - Stores and manages these entities and relationships in a graph database or Supabase.
    - Provides an API for querying the knowledge graph (e.g., finding related papers, authors, concepts).
- **Interactions:** Document Processor, Supabase Database (or dedicated graph database), potentially Analysis Engine.

### UI Service
- **Purpose:** Provides a user-friendly web interface for interacting with the IRN system.
- **Responsibilities:**
    - Allows users to search for and request new papers.
    - Displays paper information and processing status.
    - Enables users to submit queries for analysis and view results.
    - Interacts with the API Gateway to communicate with backend services.
- **Interactions:** API Gateway, Users.

### User Profile Service
- **Purpose:** Manages user accounts, preferences, and history.
- **Responsibilities:** (To be expanded)
    - Handles user authentication and authorization.
    - Stores user profiles, saved papers, search history, and preferences.
    - Likely interacts with Supabase Database for data persistence.
- **Interactions:** API Gateway, Supabase Database.

## Key Interaction Flows

### Flow 1: Fetching and Processing a New Paper

1.  **User Request:** User requests to fetch a paper (e.g., via UI Service or directly to API Gateway), providing an identifier like a DOI or arXiv ID.
2.  **Routing:** API Gateway routes the "fetch paper" request to the **Paper Fetcher** microservice.
3.  **Metadata Retrieval & Storage:**
    *   **Paper Fetcher** queries external academic sources (e.g., arXiv API, Semantic Scholar API) for the paper's metadata.
    *   It stores the retrieved metadata (title, authors, abstract, publication URL, etc.) in the **Supabase database** (e.g., in a `Papers` table).
    *   Optionally, if a PDF URL is available, Paper Fetcher downloads the PDF and stores it in a designated location (e.g., **Supabase Storage** or other cloud storage).
4.  **Processing Initiation:** User (or an automated trigger post-fetch) initiates processing for the fetched paper via the API Gateway.
5.  **Routing to Processor:** API Gateway routes the "process document" request to the **Document Processor** microservice, providing the paper's ID or location.
6.  **Document Retrieval & Chunking:**
    *   **Document Processor** retrieves the paper's content (e.g., PDF from **Supabase Storage** or a URL).
    *   It parses the document, extracts text, and divides it into smaller, semantically meaningful chunks.
    *   Information about these chunks (content, order, reference to the original paper) is stored in the **Supabase database** (e.g., in a `Chunks` table, linked to the `Papers` table).
7.  **Embedding Generation Request:** **Document Processor** sends the text content of each chunk to the **Vector Service**.
8.  **Embedding Storage:**
    *   **Vector Service** receives the text chunks.
    *   For each chunk, it calls the **Gemini API** (or a similar embedding model) to generate a vector embedding.
    *   These embeddings are then stored in the **Supabase database**, typically in the `Chunks` table alongside the chunk content, using pgvector for efficient similarity search.

### Flow 2: Performing Analysis (RAG)

1.  **User Query:** User submits a query for analysis (e.g., "What are the main challenges in X field based on recent papers?") through the UI Service or directly to the API Gateway.
2.  **Routing to Analysis Engine:** API Gateway routes the analysis request to the **Analysis Engine**.
3.  **Query Embedding:**
    *   **Analysis Engine** sends the user's query to the **Vector Service**.
    *   **Vector Service** generates a vector embedding for the user's query using the same model (e.g., **Gemini API**) used for document chunks.
4.  **Similarity Search:**
    *   **Vector Service** performs a similarity search (e.g., cosine similarity) against the stored chunk embeddings in the **Supabase database** (pgvector in `Chunks` table) to find the most relevant document chunks.
5.  **Context Retrieval:** **Vector Service** returns the content of these relevant chunks to the **Analysis Engine**.
6.  **Prompt Construction:** **Analysis Engine** constructs a comprehensive prompt for a generative AI model. This prompt typically includes:
    *   The original user query.
    *   The retrieved relevant document chunks (as context).
    *   Specific instructions for the desired output (e.g., "Summarize the findings...", "Answer the question based on the provided context...").
7.  **AI-Powered Generation:** **Analysis Engine** sends this detailed prompt to the **Gemini API** (via a `GeminiClient` or similar SDK) to generate a coherent and contextually relevant response.
8.  **Response Delivery:** **Analysis Engine** receives the generated analysis from Gemini and returns it to the user via the API Gateway (and subsequently to the UI Service, if applicable).

## Architecture Diagram (Conceptual)

Imagine a central **API Gateway** through which all external requests (e.g., from a UI Service or direct API calls) enter.

- The **API Gateway** communicates with several backend services:
    - **Paper Fetcher:** Talks to external academic APIs and the **Supabase DB** (for metadata). It might also interact with **Cloud Storage** (like Supabase Storage) for PDFs.
    - **Document Processor:** Gets files (e.g., from **Cloud Storage** or a URL), processes them, and sends data to the **Vector Service**. It also updates the **Supabase DB** (chunk info).
    - **Vector Service:** Receives text from the Document Processor, generates embeddings using the **Gemini API**, and stores/retrieves these vectors in the **Supabase DB** (using pgvector).
    - **Analysis Engine:** Takes user queries, fetches relevant context from the **Vector Service**, and uses the **Gemini API** to generate insights.
    - **KG Service:** (Details to be expanded - likely interacts with **Supabase DB** or a dedicated graph database for knowledge graph construction and querying).
    - **User Profile Service:** (Details to be expanded - manages user data, likely in **Supabase DB**).

- The **UI Service** (if present) interacts with the **API Gateway** to provide a user interface.
- All services utilize shared components from the **Core Module** (config, models, DB clients).
- The **Supabase DB** is a key persistent store for metadata, chunks, and vectors.
- The **Gemini API** is used for AI-powered text generation and embeddings by relevant services.
