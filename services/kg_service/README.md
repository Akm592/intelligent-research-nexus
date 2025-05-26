# Service: Knowledge Graph Service (Placeholder)

## Purpose

The Knowledge Graph (KG) Service is intended to manage the creation, storage, and querying of a knowledge graph derived from the processed academic documents. **Currently, this service is a placeholder and its functionalities are not yet implemented.**

## Key Functionalities (Planned)

-   Extracting entities (e.g., authors, topics, methods, datasets) and relationships (e.g., "authored_by", "uses_method", "related_to") from the text content of processed papers and their metadata.
-   Storing these entities and relationships in a graph-oriented database (e.g., Neo4j, AWS Neptune) or potentially using graph extensions/libraries with PostgreSQL in Supabase.
-   Providing an API for querying the knowledge graph (e.g., finding papers by a specific author, discovering related research topics, visualizing connections between concepts).
-   Updating the knowledge graph as new papers are processed or existing ones are re-analyzed.

## API Endpoints

### `GET /health`

-   **Description:** Basic health check for the KG Service. As the service is currently a placeholder, this endpoint confirms that the service is running.
-   **Response:**
    ```json
    {
      "status": "ok",
      "message": "KG Service is running (placeholder)"
    }
    ```

*(Note: Additional endpoints for graph creation, querying, and management will be defined as the service is implemented.)*

## Dependencies

### Core Module

-   `core.config.Settings`: Primarily used for standard logging configurations. In the future, it would also manage configurations related to the graph database connection.

### Libraries

-   `fastapi`: The web framework used to build the API.
-   `uvicorn`: The ASGI server for running the FastAPI application (typically used during development).
-   `pydantic`: For data validation and settings management (via `core.config` and future request/response models).
-   `python-dotenv`: For managing environment variables (loaded via `core.config`).

*(Note: Future dependencies will include clients for the chosen graph database, e.g., `neo4j-driver`, or libraries for graph processing and NLP if entity/relationship extraction is handled within this service.)*

## Configuration

-   **Standard Logging Configurations:** Inherited from `core.config.Settings` (e.g., `LOG_LEVEL`).
-   **(Future Configurations):**
    -   Graph database connection URI (e.g., `NEO4J_URI`).
    -   Authentication credentials for the graph database (e.g., `NEO4J_USER`, `NEO4J_PASSWORD`).
    -   Parameters related to entity extraction models or services.

Currently, as a placeholder, the service has minimal configuration beyond basic logging.
