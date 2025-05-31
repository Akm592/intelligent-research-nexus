# services/kg_service/app/main.py
from fastapi import FastAPI, HTTPException, Depends, Body, status
from contextlib import asynccontextmanager
from neo4j import AsyncDriver, exceptions as neo4j_exceptions

# Use core logger configured in root project
from core.config import logger as core_logger
# Use relative imports for local modules
from . import neo4j_ops
from .models import KgPaperInput, KgAuthorInput, AuthoredByRequest, KgResponse

# Get child logger specific to this service/module
logger = core_logger.getChild("KgService").getChild("Main")

# --- Lifespan Context Manager for Neo4j Driver ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages Neo4j driver initialization and closure."""
    logger.info("KG Service starting up, initializing Neo4j driver...")
    await neo4j_ops.init_neo4j_driver() # Initialize driver on startup
    yield # Application runs here
    logger.info("KG Service shutting down, closing Neo4j driver...")
    await neo4j_ops.close_neo4j_driver() # Close driver on shutdown

# --- FastAPI App Instance ---
app = FastAPI(
    title="Knowledge Graph Service (Neo4j)",
    description="Manages interactions with the Neo4j graph database for IRN.",
    version="1.0.0",
    lifespan=lifespan # Use the lifespan context manager
)

# --- Dependency for Neo4j Driver ---
# This ensures endpoints don't execute if the driver failed to initialize.
async def get_driver_dependency() -> AsyncDriver:
    """Dependency function to get the Neo4j driver, raising 503 if unavailable."""
    try:
        # get_neo4j_driver now raises RuntimeError if not available
        return await neo4j_ops.get_neo4j_driver()
    except RuntimeError as e:
         # Log critical error if driver isn't available during request handling
         logger.critical(f"Dependency Error: Neo4j driver unavailable during request. {e}")
         raise HTTPException(
             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
             detail="Knowledge Graph database connection is currently unavailable."
         )

# --- API Endpoints ---

# -- Meta Endpoints --
@app.get("/health", response_model=KgResponse, tags=["Meta"])
async def health_check():
    """Checks connectivity to the Neo4j database."""
    driver = None
    try:
        # Try to get the driver first - tests basic initialization state
        driver = await neo4j_ops.get_neo4j_driver()
        # Perform a lightweight query to verify connectivity
        async with driver.session(database="neo4j") as session:
             await session.run("RETURN 1") # Simple query
        logger.debug("Health check: Neo4j connectivity verified.")
        return KgResponse(status="success", message="KG Service is running and connected to Neo4j.")
    except RuntimeError:
         # get_neo4j_driver failed
         return KgResponse(status="error", message="KG Service is running but Neo4j driver is not initialized.")
    except neo4j_exceptions.ServiceUnavailable as e:
         logger.error(f"Health check failed: Neo4j Service Unavailable: {e}", exc_info=False)
         return KgResponse(status="error", message=f"KG Service running but cannot connect to Neo4j (ServiceUnavailable): {e}")
    except neo4j_exceptions.AuthError as e:
         logger.error(f"Health check failed: Neo4j Authentication Error: {e}", exc_info=False)
         return KgResponse(status="error", message=f"KG Service running but Neo4j authentication failed: {e}")
    except Exception as e:
        # Catch other potential driver/session errors during health check
        logger.error(f"Health check failed: Unexpected error verifying Neo4j connectivity: {e}", exc_info=True)
        return KgResponse(status="error", message=f"KG Service running but failed to verify Neo4j connection: {e}")


# --- Node Endpoints ---

@app.post(
    "/nodes/paper",
    response_model=KgResponse,
    status_code=status.HTTP_201_CREATED, # Use 201 for successful creation/update
    tags=["Nodes"],
    summary="Add or Update Paper Node"
)
async def add_paper_node(
    paper: KgPaperInput = Body(...),
    driver: AsyncDriver = Depends(get_driver_dependency) # Ensure driver is ready
):
    """
    Adds a new Paper node or updates properties if a node with the same `paper_id` exists.
    Uses `MERGE` for idempotency.
    """
    success = await neo4j_ops.add_update_paper(paper)
    if success:
        return KgResponse(status="success", message=f"Paper node '{paper.paper_id}' merged successfully.")
    else:
        # Log details are in neo4j_ops
        # Return 500 Internal Server Error as the operation failed unexpectedly
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to merge paper node '{paper.paper_id}'. Check KG service logs for details."
        )

@app.post(
    "/nodes/author",
    response_model=KgResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Nodes"],
    summary="Add or Update Author Node"
)
async def add_author_node(
    author: KgAuthorInput = Body(...),
    driver: AsyncDriver = Depends(get_driver_dependency)
):
    """
    Adds a new Author node or updates `last_updated` if an author with the same `name` exists.
    Uses `MERGE` for idempotency. **Note:** Name collision is possible for common names.
    """
    # Basic validation for empty name
    if not author.name or not author.name.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Author name cannot be empty.")

    success = await neo4j_ops.add_update_author(author)
    if success:
        return KgResponse(status="success", message=f"Author node '{author.name}' merged successfully.")
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to merge author node '{author.name}'. Check KG service logs for details."
        )


# --- Relationship Endpoints ---

@app.post(
    "/relationships/authored_by",
    response_model=KgResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Relationships"],
    summary="Create AUTHORED_BY Relationship"
)
async def add_authored_by_rel(
    rel_data: AuthoredByRequest = Body(...),
    driver: AsyncDriver = Depends(get_driver_dependency)
):
    """
    Creates an `AUTHORED_BY` relationship between an existing Paper and an existing Author.
    Uses `MERGE` on the relationship to ensure idempotency. Fails if either node does not exist.
    """
    # Basic validation
    if not rel_data.paper_id or not rel_data.author_name or not rel_data.author_name.strip():
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Both paper_id and non-empty author_name are required.")

    success = await neo4j_ops.add_authored_by_relationship(rel_data.paper_id, rel_data.author_name)
    if success:
        return KgResponse(status="success", message=f"AUTHORED_BY relationship between '{rel_data.paper_id}' and '{rel_data.author_name}' merged.")
    else:
        # The operation might fail because nodes don't exist or due to a DB error.
        # Returning 500 might be appropriate if the expectation is that nodes *should* exist.
        # Alternatively, check node existence first and return 404, but that's more complex.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to merge AUTHORED_BY relationship for paper '{rel_data.paper_id}' and author '{rel_data.author_name}'. Nodes might be missing or a database error occurred."
        )