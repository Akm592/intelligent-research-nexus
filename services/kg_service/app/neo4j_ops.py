# services/kg_service/app/neo4j_ops.py
import asyncio
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession, AsyncTransaction, exceptions as neo4j_exceptions, Result # Added Result import
from core.config import settings, logger as core_logger
from .models import KgPaperInput, KgAuthorInput
from typing import Optional, Any # Added Any import

# Get a child logger
logger = core_logger.getChild("KgService").getChild("Neo4jOps")

# Global driver instance - managed via lifespan in main.py
driver: Optional[AsyncDriver] = None

# --- Driver Management (Keep as is from previous correct version) ---
async def init_neo4j_driver():
    """Initializes the Neo4j Async Driver."""
    global driver
    if driver is not None: logger.warning("Neo4j driver already initialized."); return
    uri = settings.NEO4J_URI; user = settings.NEO4J_USER; password = settings.NEO4J_PASSWORD
    if not all([uri, user, password]) or password == "your_neo4j_password" or password == "your_secure_password":
        logger.error("Neo4j connection details incomplete or using default password. Driver not initialized."); driver = None; return
    try:
        driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        await driver.verify_connectivity()
        logger.info(f"Neo4j Async Driver initialized successfully for URI: {uri}")
    except neo4j_exceptions.AuthError as e: logger.error(f"Neo4j Authentication failed for user '{user}': {e}. Check credentials."); driver = None
    except neo4j_exceptions.ServiceUnavailable as e: logger.error(f"Neo4j Service Unavailable at {uri}: {e}. Check connection URI and database status."); driver = None
    except Exception as e: logger.error(f"Failed to initialize Neo4j Driver: {e}", exc_info=True); driver = None

async def close_neo4j_driver():
    """Closes the Neo4j Driver."""
    global driver
    if driver: logger.info("Closing Neo4j Async Driver."); await driver.close(); driver = None
    else: logger.info("Neo4j driver was not initialized, skipping close.")

async def get_neo4j_driver() -> AsyncDriver:
    """Returns the initialized Neo4j driver instance. Raises RuntimeError if not initialized."""
    if driver is None:
        logger.warning("Neo4j driver requested but was not initialized. Attempting initialization now.")
        await init_neo4j_driver()
        if driver is None: logger.error("Neo4j driver is still not available after re-initialization attempt."); raise RuntimeError("Neo4j driver is not available.")
    return driver

# --- Cypher Operations ---

async def add_update_paper(paper_data: KgPaperInput) -> bool:
    """Adds or updates a Paper node in Neo4j using MERGE."""
    try:
        driver_instance = await get_neo4j_driver() # Renamed variable to avoid conflict
    except RuntimeError:
        return False

    job_prefix=f"[{paper_data.paper_id}]"
    logger.debug(f"{job_prefix} Adding/updating Paper node.")
    query = """
    MERGE (p:Paper {id: $paper_id})
    ON CREATE SET
        p.title = $title, p.publication_date = $publication_date, p.source = $source,
        p.url = $url, p.pdf_url = $pdf_url,
        p.created_at = timestamp(), p.last_updated = timestamp()
    ON MATCH SET
        p.title = $title, p.publication_date = $publication_date, p.source = $source,
        p.url = $url, p.pdf_url = $pdf_url,
        p.last_updated = timestamp()
    RETURN p.id as paperId
    """
    params = paper_data.model_dump()

    # --- Define the async transaction function ---
    async def _execute_merge_paper(tx: AsyncTransaction, query: str, params: dict) -> Optional[Any]:
        result_cursor = await tx.run(query, params) # MUST await tx.run()
        record = await result_cursor.single(strict=True) # MUST await .single()
        return record

    try:
        async with driver_instance.session(database="neo4j") as session:
            # Pass the async transaction function to execute_write
            record = await session.execute_write(_execute_merge_paper, query, params)
            # Check the actual record returned by the transaction function
            if record and record["paperId"] == paper_data.paper_id:
                 logger.info(f"{job_prefix} Successfully merged Paper node.")
                 return True
            else:
                 logger.warning(f"{job_prefix} Paper node merge query ran but did not return expected record.")
                 return False
    except neo4j_exceptions.ConstraintError as e: logger.error(f"{job_prefix} Constraint error merging paper node: {e}"); return False
    except neo4j_exceptions.ClientError as e: logger.error(f"{job_prefix} Client error merging paper node: {e}"); return False
    except neo4j_exceptions.TransientError as e: logger.error(f"{job_prefix} Transient error merging paper node (potential for retry): {e}"); return False
    except Exception as e: logger.error(f"{job_prefix} Unexpected error merging Paper node: {e}", exc_info=True); return False

async def add_update_author(author_data: KgAuthorInput) -> bool:
    """Adds or updates an Author node using MERGE on name."""
    try:
        driver_instance = await get_neo4j_driver()
    except RuntimeError:
        return False

    author_name = author_data.name.strip()
    if not author_name: logger.warning("Attempted to add author with empty name."); return False
    logger.debug(f"Adding/updating Author node: {author_name}")

    query = """
    MERGE (a:Author {name: $name})
    ON CREATE SET a.created_at = timestamp(), a.last_updated = timestamp()
    ON MATCH SET a.last_updated = timestamp()
    RETURN a.name as authorName
    """
    params = {"name": author_name}

    # --- Define the async transaction function ---
    async def _execute_merge_author(tx: AsyncTransaction, query: str, params: dict) -> Optional[Any]:
        result_cursor = await tx.run(query, params) # MUST await tx.run()
        record = await result_cursor.single(strict=True) # MUST await .single()
        return record

    try:
        async with driver_instance.session(database="neo4j") as session:
            record = await session.execute_write(_execute_merge_author, query, params)
            if record and record["authorName"] == author_name:
                logger.info(f"Successfully merged Author node: {author_name}")
                return True
            else:
                logger.warning(f"Author node merge query ran but did not return expected record for: {author_name}")
                return False
    except neo4j_exceptions.ConstraintError as e: logger.error(f"Constraint error merging author node '{author_name}': {e}"); return False
    except neo4j_exceptions.ClientError as e: logger.error(f"Client error merging author node '{author_name}': {e}"); return False
    except neo4j_exceptions.TransientError as e: logger.error(f"Transient error merging author node '{author_name}': {e}"); return False
    except Exception as e: logger.error(f"Unexpected error merging Author node '{author_name}': {e}", exc_info=True); return False


async def add_authored_by_relationship(paper_id: str, author_name: str) -> bool:
    """Creates an AUTHORED_BY relationship between an existing Paper and Author."""
    try:
        driver_instance = await get_neo4j_driver()
    except RuntimeError:
        return False

    job_prefix=f"[{paper_id}]"
    author_name_stripped = author_name.strip()
    if not author_name_stripped: logger.warning(f"{job_prefix} Attempted to link author with empty name."); return False
    logger.debug(f"{job_prefix} Adding AUTHORED_BY relationship -> Author: {author_name_stripped}")

    query = """
    MATCH (p:Paper {id: $paper_id})
    MATCH (a:Author {name: $author_name})
    MERGE (p)-[r:AUTHORED_BY]->(a)
    RETURN count(r) as rel_count
    """
    params = {"paper_id": paper_id, "author_name": author_name_stripped}

    # --- Define the async transaction function ---
    async def _execute_merge_relationship(tx: AsyncTransaction, query: str, params: dict) -> Optional[Any]:
        result_cursor = await tx.run(query, params) # MUST await tx.run()
        # .single() is appropriate here as MATCH/MERGE returns one row with the count
        record = await result_cursor.single() # MUST await .single()
        return record

    try:
        async with driver_instance.session(database="neo4j") as session:
            record = await session.execute_write(_execute_merge_relationship, query, params)
            # Check if the query executed and returned a record (even if count is 0 or 1)
            if record is not None: # Query executed successfully
                 # You could check record["rel_count"] if needed, but MERGE handles idempotency
                 logger.info(f"{job_prefix} Successfully merged AUTHORED_BY relationship for Author: {author_name_stripped}")
                 return True
            else:
                 # This implies the query itself failed to return a single row,
                 # possibly because MATCH failed if strict=True was used, or other issues.
                 # Or if MATCH failed and MERGE didn't run.
                 logger.warning(f"{job_prefix} AUTHORED_BY merge failed, likely missing Paper or Author node for Author: {author_name_stripped}")
                 return False
    except neo4j_exceptions.ClientError as e: logger.error(f"{job_prefix} Client error merging AUTHORED_BY for Author '{author_name_stripped}': {e}"); return False
    except neo4j_exceptions.TransientError as e: logger.error(f"{job_prefix} Transient error merging AUTHORED_BY for Author '{author_name_stripped}': {e}"); return False
    except Exception as e: logger.error(f"{job_prefix} Unexpected error merging AUTHORED_BY for Author '{author_name_stripped}': {e}", exc_info=True); return False