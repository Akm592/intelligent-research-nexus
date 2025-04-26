# services/analysis_engine/app/main.py

from fastapi import FastAPI, HTTPException, Depends, Request
from contextlib import asynccontextmanager
import httpx
import logging
from typing import Dict, Any, Optional

from core.config import settings, logger as core_logger
from core.models import AnalysisRequest, AnalysisResult

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from .rag_chain import VectorServiceRetriever, create_rag_chain # Import from new file
# -------------------------

# Child logger
logger = core_logger.getChild("AnalysisEngine").getChild("Main")

# --- Global variables for LangChain components ---
http_client: httpx.AsyncClient | None = None
llm_pro: ChatGoogleGenerativeAI | None = None
llm_flash: ChatGoogleGenerativeAI | None = None
vector_retriever: VectorServiceRetriever | None = None

# --- Lifespan Manager for Initialization ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, llm_pro, llm_flash, vector_retriever
    logger.info("Analysis Engine starting up...")

    # 1. Initialize shared HTTPX client
    http_client = httpx.AsyncClient(timeout=120.0)
    logger.info("HTTPX client initialized.")

    # 2. Initialize LangChain LLMs - *** WITH API KEY ***
    if not settings.GEMINI_API_KEY or settings.GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
         logger.error("GEMINI_API_KEY is not configured in settings. Cannot initialize LangChain Gemini models.")
         llm_pro = None
         llm_flash = None
    else:
        try:
            logger.info("Initializing LangChain Gemini models using configured API Key...")
            # *** Pass the API key explicitly ***
            llm_pro = ChatGoogleGenerativeAI(
                model=settings.GEMINI_PRO_MODEL,
                google_api_key=settings.GEMINI_API_KEY, # <--- ADD THIS
                temperature=0.5,
                convert_system_message_to_human=True
            )
            llm_flash = ChatGoogleGenerativeAI(
                model=settings.GEMINI_FLASH_MODEL,
                google_api_key=settings.GEMINI_API_KEY, # <--- ADD THIS
                temperature=0.7,
                convert_system_message_to_human=True
            )
            # Optional: Add a simple test call here to verify connectivity if needed
            # await llm_pro.ainvoke("Test prompt")
            logger.info(f"LangChain Gemini models initialized successfully ('{settings.GEMINI_PRO_MODEL}', '{settings.GEMINI_FLASH_MODEL}').")
        except Exception as e:
            # Catch errors during initialization (e.g., invalid key, network issue)
            logger.error(f"Failed to initialize LangChain Gemini models with API Key: {e}", exc_info=True)
            llm_pro = None
            llm_flash = None

    # 3. Initialize Custom Retriever
    if http_client:
        vector_retriever = VectorServiceRetriever(
            vector_service_url=settings.VECTOR_SERVICE_URL,
            httpx_client=http_client,
            top_k=5 # Default k
        )
        logger.info(f"VectorServiceRetriever initialized for URL: {settings.VECTOR_SERVICE_URL}")
    else:
         logger.error("HTTP client not available, cannot initialize VectorServiceRetriever.")
         vector_retriever = None

    yield # Application runs here

    # --- Shutdown ---
    logger.info("Analysis Engine shutting down...")
    if http_client:
        await http_client.aclose()
        logger.info("HTTPX client closed.")

# --- FastAPI App ---
app = FastAPI(
    title="Analysis Engine Service",
    description="Performs RAG analysis using LangChain and Vector Service.",
    lifespan=lifespan
)

# --- Health Check ---
@app.get("/health")
async def health_check():
    # Check status of initialized components
    llm_status = "OK" if llm_pro and llm_flash else "Error: LLM not initialized"
    retriever_status = "OK" if vector_retriever else "Error: Retriever not initialized"
    http_client_status = "OK" if http_client else "Error: HTTP Client not initialized"

    all_ok = all(s == "OK" for s in [llm_status, retriever_status, http_client_status])

    return {
        "status": "ok" if all_ok else "error",
        "dependencies": {
            "llm": llm_status,
            "retriever": retriever_status,
            "http_client": http_client_status
        }
    }

# --- Analysis Endpoint ---

# Dependency function to get initialized components
def get_llm(request: Request, model_preference: str = "pro") -> ChatGoogleGenerativeAI:
    # Simple preference, could be based on request param later
    # Prioritize pro if flash isn't available or vice-versa
    preferred_llm = llm_flash if model_preference == "flash" else llm_pro
    fallback_llm = llm_pro if model_preference == "flash" else llm_flash

    llm = preferred_llm if preferred_llm else fallback_llm

    if not llm:
        # This should only happen if BOTH failed to initialize
        logger.error("LLM dependency error: No initialized LLM available.")
        raise HTTPException(status_code=503, detail="Analysis LLM not available/initialized.")
    # Log which llm is being used
    logger.debug(f"Using LLM: {llm.model}")
    return llm

def get_retriever(request: Request) -> VectorServiceRetriever:
    if not vector_retriever:
        logger.error("Retriever dependency error: VectorServiceRetriever is not available.")
        raise HTTPException(status_code=503, detail="Analysis retriever not initialized.")
    return vector_retriever

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_documents(
    request: AnalysisRequest, # Use the detailed request model
    llm: ChatGoogleGenerativeAI = Depends(get_llm), # Inject LLM
    retriever: VectorServiceRetriever = Depends(get_retriever) # Inject Retriever
):
    """
    Performs analysis using a RAG chain constructed with LangChain.
    """
    logger.info(f"Received analysis request: type={request.analysis_type}, papers={request.paper_ids}, query='{request.query[:50] if request.query else None}'")

    # --- Input Validation & Preparation ---
    analysis_type = request.analysis_type
    retrieval_query = request.query # Use the direct query for retrieval

    # Handle case where retrieval query might be needed but not provided
    if not retrieval_query:
        if analysis_type == "qa":
             raise HTTPException(status_code=400, detail="A 'query' (question) is required for 'qa' analysis type.")
        else:
             # For summary/compare/gap, if no query, we might need to derive one or skip retrieval?
             # Option 1: Derive a generic topic
             retrieval_query = f"General information about papers: {', '.join(request.paper_ids or ['Unknown'])}"
             logger.warning(f"No query provided for analysis type '{analysis_type}'. Using generic retrieval topic: '{retrieval_query[:50]}...'")
             # Option 2: Raise error - uncomment if retrieval is always mandatory
             # raise HTTPException(status_code=400, detail=f"A 'query' is required for retrieval in analysis type '{analysis_type}'.")

    # --- Prepare Retriever Filters ---
    retriever_filters = None
    if request.paper_ids:
        # Current retriever only handles single paper_id filter via 'eq'
        if len(request.paper_ids) == 1:
             retriever_filters = {"paper_id": request.paper_ids[0]}
             logger.info(f"Applying filter to retriever: paper_id={request.paper_ids[0]}")
        else:
            # Log limitation - Vector service/retriever needs update for multi-ID filtering
            logger.warning(f"Multiple paper IDs provided ({request.paper_ids}), but vector service filter currently supports only one. Filtering by the first ID: {request.paper_ids[0]}.")
            retriever_filters = {"paper_id": request.paper_ids[0]}

    # --- Create RAG Chain ---
    try:
        rag_chain = create_rag_chain(llm, retriever, analysis_type)
        logger.info(f"Created RAG chain for analysis type: {analysis_type}")
    except Exception as e:
        logger.error(f"Failed to create RAG chain: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error creating analysis chain.")

    # --- Execute RAG Chain ---
    try:
        logger.info(f"Invoking RAG chain with query='{retrieval_query[:50]}...' and filters={retriever_filters}")

        # Determine input for the specific chain structure
        chain_input: Any
        if analysis_type == "qa":
             # The full_chain for QA expects a dict with 'question'
             chain_input = {"question": request.query}
        else:
             # The full_chain for others expects the retrieval query directly
             chain_input = retrieval_query

        # Prepare config for passing filters to the retriever step
        chain_config = {
             "configurable": {
                 "retriever_filters": retriever_filters # Pass filters here
            }
        }
        # If your retriever doesn't directly support config pass-through,
        # you might need to modify the retriever or the chain structure.
        # Assuming the structure in rag_chain.py handles passing filters down.


        # Invoke the asynchronous chain
        # Add timeout handling?
        result = await rag_chain.ainvoke(chain_input, config=chain_config)

        logger.info("RAG chain execution successful.")

        # --- Format and Return Response ---
        final_text = result.get("answer") or result.get("result_text", "Analysis generated, but no text found.")
        cited_chunk_ids = result.get("cited_sources", [])

        # Basic check if response seems valid
        if not final_text or final_text.strip() == "":
             logger.warning("RAG chain returned an empty result_text/answer.")
             final_text = "(Analysis completed, but the result was empty.)"

        return AnalysisResult(
            result_text=final_text.strip(), # Trim whitespace
            cited_sources=cited_chunk_ids,
            analysis_type=analysis_type
        )

    except Exception as e:
        logger.error(f"Error executing RAG chain for analysis type {analysis_type}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during analysis execution: {e}")