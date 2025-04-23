# services/analysis_engine/app/main.py
from fastapi import FastAPI, HTTPException
from core.models import AnalysisRequest, AnalysisResult, SearchQuery, SearchResponse
from core.config import settings
from core.gemini_client import gemini_client
import httpx
import logging
import asyncio
import re # For basic citation extraction
from typing import List, Optional, Tuple

logger = logging.getLogger("IRN_Core").getChild("AnalysisEngine")

app = FastAPI(title="Analysis Engine Service")

# Single HTTP client instance for the service lifespan
http_client: httpx.AsyncClient | None = None

@app.on_event("startup")
async def startup_event():
    global http_client
    http_client = httpx.AsyncClient(timeout=60.0) # Timeout for search calls
    logger.info("Analysis Engine started. HTTPX Client initialized.")

@app.on_event("shutdown")
async def shutdown_event():
    if http_client:
        await http_client.aclose()
        logger.info("HTTPX Client closed.")

async def retrieve_context(paper_ids: List[str] | None, query: str | None, top_k: int = 10) -> Tuple[str, List[str]]:
    """Retrieves relevant text context using the Vector Service. Returns context string and list of retrieved chunk IDs."""
    if not paper_ids and not query:
        logger.info("No paper IDs or query provided for context retrieval.")
        return "", [] # Nothing to retrieve

    # Construct search query for vector service
    # Use query if available, otherwise formulate based on paper IDs
    search_text = query if query else f"Key information about papers: {', '.join(paper_ids or [])}"
    logger.info(f"Retrieving context. Vector search query: '{search_text[:100]}...'")

    search_filters = {}
    if paper_ids:
        # Assuming Supabase RPC function can handle a list for filtering or needs specific key
        # Adjust based on your `match_document_chunks` function signature
        # Example: if function takes `filter_paper_ids TEXT[]`...
        # search_filters["paper_id_list"] = paper_ids
        # Example: if function takes single `filter_paper_id TEXT`
        if len(paper_ids) == 1:
             search_filters["filter_paper_id"] = paper_ids[0]
        else:
             logger.warning(f"Filtering by multiple paper IDs ({len(paper_ids)}) not implemented in V1 RPC call. Retrieving general context.")
             # Or make multiple calls if necessary

    search_query = SearchQuery(
        query_text=search_text,
        top_k=top_k,
        filters=search_filters if search_filters else None # Pass filters only if created
    )

    if not http_client:
        logger.error("HTTP client not available for context retrieval.")
        return "[Error: Analysis Engine client not ready]", []

    try:
        vector_service_url = f"{settings.VECTOR_SERVICE_URL}/search"
        response = await http_client.post(vector_service_url, json=search_query.model_dump())
        response.raise_for_status()
        search_response_data = response.json()
        search_response = SearchResponse(**search_response_data) # Validate response

        if not search_response.results:
            logger.warning(f"Vector service returned no results for context retrieval (Query: '{search_text[:50]}...')")
            return "[Context note: No relevant text snippets found in the knowledge base for this query/papers.]", []

        # Combine relevant text chunks
        context_parts = []
        retrieved_chunk_ids = []
        for res in search_response.results:
            context_parts.append(f"Source Chunk (ID: {res.chunk_id}, Paper: {res.paper_id}, Score: {res.score:.2f}):\n{res.text}")
            retrieved_chunk_ids.append(res.chunk_id)

        full_context = "\n\n---\n\n".join(context_parts)
        logger.info(f"Retrieved {len(search_response.results)} chunks for context.")
        return full_context, retrieved_chunk_ids

    except httpx.RequestError as e:
        logger.error(f"Failed to connect to Vector Service at {vector_service_url} for context retrieval: {e}", exc_info=False)
        return "[Error: Could not retrieve context from vector store - network issue]", []
    except httpx.HTTPStatusError as e:
        error_detail = f"Vector Service Error ({e.response.status_code})"
        try:
            downstream_error = e.response.json().get('detail', e.response.text)
        except Exception:
            downstream_error = e.response.text
        logger.error(f"{error_detail} during context retrieval: {downstream_error}", exc_info=False)
        return f"[Error: Could not retrieve context from vector store - {error_detail}]", []
    except Exception as e: # Catch JSON decode errors etc.
         logger.error(f"Error processing Vector Service response for context: {e}", exc_info=True)
         return "[Error: Could not process context from vector store]", []

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_content(request: AnalysisRequest):
    """
    Performs analysis (summary, Q&A, comparison) using Gemini, potentially with RAG.
    """
    logger.info(f"Received analysis request: type={request.analysis_type}, papers={request.paper_ids}, query='{request.query is not None}'")

    # 1. Retrieve Context (RAG step)
    context = ""
    retrieved_ids = []
    # Decide if context is needed and how much based on analysis type
    if request.analysis_type in ["qa", "compare_methods", "gap_analysis"] or (request.analysis_type == "summary" and request.paper_ids):
         context_top_k = 15 if request.analysis_type != "summary" else 10 # Retrieve more context for complex tasks
         context, retrieved_ids = await retrieve_context(request.paper_ids, request.query, top_k=context_top_k)
         if not context or "[Error:" in context:
             logger.warning(f"Context retrieval failed or returned empty. Proceeding without RAG context. Message: {context}")
             # Decide if error context should be included in prompt or handled differently
             # context = context # Include error message in prompt? Maybe not.
             context = "[Context Note: Failed to retrieve specific document context. Answering based on general knowledge or query if provided.]"

    # 2. Construct Prompt for Gemini
    # TODO: Implement more sophisticated prompt engineering based on analysis_type
    prompt_lines = []
    prompt_lines.append(f"You are an AI Research Assistant analyzing academic documents.")
    prompt_lines.append(f"Your task is to perform the following analysis: **{request.analysis_type.replace('_', ' ').title()}**")
    if request.paper_ids:
        prompt_lines.append(f"Focus on the following paper IDs if relevant: {', '.join(request.paper_ids)}")
    if request.query:
        prompt_lines.append(f"Address the following user query: \"{request.query}\"")

    prompt_lines.append(f"Provide the analysis at a '{request.detail_level}' level of detail.")

    if context:
         prompt_lines.append("\nBased on the following relevant context snippets retrieved from the documents:\n--- BEGIN CONTEXT ---")
         prompt_lines.append(context)
         prompt_lines.append("--- END CONTEXT ---")
         prompt_lines.append("\nInstructions:")
         prompt_lines.append("- Synthesize the information from the context to perform the analysis.")
         prompt_lines.append("- If the context is insufficient, state that clearly.")
         prompt_lines.append("- When referencing information specific to a context snippet, cite it using the format [Source Chunk (ID: chunk_id_here)].")
    else:
         # This case should ideally not happen if context retrieval error handling is robust
         prompt_lines.append("\nNo specific context was retrieved. Answer based on the query and paper IDs mentioned, or state that the information is not available in the knowledge base.")

    prompt_lines.append("\nBegin Analysis:")
    prompt = "\n".join(prompt_lines)

    # 3. Call Gemini (Choose model based on complexity/cost?)
    # For v1.0.0, let's default to Pro for potentially better analysis
    model_type = "pro" # Or "flash" for simpler tasks or lower cost
    # model_type = "flash" if request.analysis_type == "qa" and len(prompt) < 5000 else "pro"

    logger.info(f"Calling Gemini {model_type.upper()} for analysis. Prompt length: {len(prompt)} chars.")

    # Using the async client method directly
    analysis_text, usage_metadata = await gemini_client.generate_text(prompt, model_type=model_type)

    if analysis_text is None:
        error_msg = usage_metadata.get('error', 'Unknown analysis generation error')
        logger.error(f"Gemini analysis generation failed. Error: {error_msg}")
        # Check for safety blocking specifically
        if "Content blocked" in error_msg:
             raise HTTPException(status_code=400, detail=f"Analysis generation failed due to safety filters: {error_msg}")
        else:
             raise HTTPException(status_code=500, detail=f"Analysis generation failed: {error_msg}")

    # 4. Format Response (Basic citation extraction)
    # Use regex to find citations matching the format used in the prompt
    cited_sources = list(set(re.findall(r"\[Source Chunk \(ID: ([\w:_-]+)\)\]", analysis_text)))

    # Optional: Verify extracted citations against retrieved_ids?
    verified_citations = [cid for cid in cited_sources if cid in retrieved_ids]
    if len(verified_citations) != len(cited_sources):
        logger.warning(f"LLM generated citations not found in retrieved context: {set(cited_sources) - set(verified_citations)}")
        # Decide whether to return all generated citations or only verified ones
        # For now, return all generated ones.

    logger.info(f"Successfully generated analysis for type: {request.analysis_type}. Output length: {len(analysis_text)} chars.")
    return AnalysisResult(
        result_text=analysis_text,
        cited_sources=cited_sources, # Return all found citations
        analysis_type=request.analysis_type
    )
