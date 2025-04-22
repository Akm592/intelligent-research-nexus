# services/ui_service/app/main.py

import gradio as gr
import fastapi
import httpx
import logging
import os
import uuid
import time
from core.config import settings # Access API Gateway URL, Supabase Bucket
from core.supabase_client import get_supabase_client # To upload files
from core.models import ( # Import models for API calls
    GatewayFetchRequest, GatewayProcessRequest, GatewaySearchRequest,
    GatewayAnalysisRequest, PaperMetadata, SearchResultItem, AnalysisResult
)
from typing import Optional, List, Dict, Any

# Setup logger
logger = logging.getLogger("IRN_Core").getChild("UIService")

# Global httpx client for calling API Gateway
api_client = httpx.AsyncClient(base_url=settings.API_GATEWAY_URL, timeout=120.0)

# --- Helper Functions for API Calls ---

async def call_api_gateway(method: str, endpoint: str, payload: Optional[Dict] = None) -> Dict:
    """Helper to call the API Gateway and handle responses."""
    try:
        if method.upper() == "POST":
            response = await api_client.post(endpoint, json=payload)
        elif method.upper() == "GET":
            response = await api_client.get(endpoint)
        else:
            return {"error": f"Unsupported method: {method}"}

        response.raise_for_status() # Raise exceptions for 4xx/5xx
        api_response = response.json()
        
        if api_response.get("status") == "success":
            return {"data": api_response.get("data"), "message": api_response.get("message")}
        else:
            error_msg = api_response.get("message", "Unknown API error")
            logger.error(f"API Gateway returned error at {endpoint}: {error_msg}")
            return {"error": error_msg}
    except httpx.HTTPStatusError as e:
        error_detail = f"API Error ({e.response.status_code})"
        try: 
            downstream_error = e.response.json().get('detail', e.response.text)
        except Exception: 
            downstream_error = e.response.text
        logger.error(f"{error_detail} calling {endpoint}: {downstream_error}")
        return {"error": f"{error_detail}: {downstream_error}"}
    except httpx.RequestError as e:
        logger.error(f"Network error calling {endpoint}: {e}")
        return {"error": f"Cannot reach API Gateway at {settings.API_GATEWAY_URL}"}
    except Exception as e:
        logger.error(f"Unexpected error calling {endpoint}: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred: {e}"}

# --- Updated Supabase Client Helper ---

async def get_supabase_service_client():
    """
    Gets a Supabase client using the service role key to bypass RLS.
    This is a temporary helper until supabase_client.py is updated.
    """
    from supabase import create_client
    import asyncio
    from functools import partial
    
    if not hasattr(settings, 'SUPABASE_SERVICE_KEY') or not settings.SUPABASE_SERVICE_KEY:
        logger.error("SUPABASE_SERVICE_KEY not configured in settings. Add it to your .env file.")
        raise ValueError("SUPABASE_SERVICE_KEY not configured")
    
    try:
        # Run create_client in a thread pool since it's synchronous
        loop = asyncio.get_running_loop()
        client_instance = await loop.run_in_executor(
            None,
            partial(create_client, settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
        )
        return client_instance
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client with service role key: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize Supabase client: {e}")

# --- Gradio Interface Functions ---

async def upload_and_process_pdf(file_obj):
    """Handles PDF upload, stores in Supabase, triggers processing."""
    if file_obj is None:
        return "Please upload a PDF file.", gr.update(value=None)

    file_path = file_obj.name # Gradio provides temp file path
    original_filename = os.path.basename(file_path)
    logger.info(f"Received file upload: {original_filename}")

    # Generate a unique paper ID (e.g., filename prefix + uuid)
    filename_prefix = os.path.splitext(original_filename)[0].replace(" ", "_").lower()[:30]
    paper_id = f"upload:{filename_prefix}_{uuid.uuid4().hex[:8]}"
    object_name = f"uploads/{paper_id}/{original_filename}" # Path in Supabase bucket

    bucket_name = settings.DOC_STORAGE_BUCKET

    try:
        # 1. Upload to Supabase Storage using service role key to bypass RLS
        try:
            # First attempt: Use service role key client (preferred)
            supabase = await get_supabase_service_client()
            logger.info(f"Uploading '{original_filename}' to Supabase Storage: Bucket='{bucket_name}', Object='{object_name}'")
            
            with open(file_path, 'rb') as f:
                upload_response = supabase.storage.from_(bucket_name).upload(
                    path=object_name,
                    file=f,
                    file_options={"content-type": "application/pdf", "upsert": "false"}
                )
        except (ValueError, AttributeError) as e:
            # Fallback: Use regular client with returning:minimal option
            logger.warning(f"Service role key not available, trying alternative approach: {e}")
            supabase = await get_supabase_client()
            
            with open(file_path, 'rb') as f:
                upload_response = supabase.storage.from_(bucket_name).upload(
                    path=object_name,
                    file=f,
                    file_options={"content-type": "application/pdf", "upsert": "false", "returning": "minimal"}
                )
            
        logger.info(f"Successfully uploaded to Supabase Storage for paper ID {paper_id}")

        # 2. Trigger processing via API Gateway
        logger.info(f"Triggering processing via API Gateway for paper ID: {paper_id}")
        process_payload = GatewayProcessRequest(
            paper_id=paper_id,
            bucket_name=bucket_name,
            object_name=object_name
        ).model_dump(exclude_none=True)
        
        api_result = await call_api_gateway("POST", "/documents/process", payload=process_payload)
        
        if "error" in api_result:
            return f"Error triggering processing: {api_result['error']}", gr.update(value=None)
        else:
            status_message = api_result.get("message", "Processing started successfully.")
            return f"{status_message}\nPaper ID: {paper_id}", gr.update(value=None) # Clear file input
            
    except Exception as e:
        logger.error(f"Error during upload/processing trigger for {original_filename}: {e}", exc_info=True)
        return f"An unexpected error occurred: {e}", gr.update(value=None)

async def fetch_papers_ui(query: str):
    """Calls API to fetch papers based on query."""
    if not query:
        return "Please enter a search query.", []
    
    logger.info(f"UI requesting fetch for query: {query}")
    payload = GatewayFetchRequest(query=query).model_dump()
    
    api_result = await call_api_gateway("POST", "/documents/fetch", payload=payload)
    
    if "error" in api_result:
        return f"Error fetching papers: {api_result['error']}", []
    else:
        papers = api_result.get("data", [])
        if not papers:
            return "No papers found for this query.", []
            
        # Format for display
        formatted_papers = [f"ID: {p.get('id', 'N/A')} - Title: {p.get('title', 'N/A')}" for p in papers]
        paper_ids = [p.get('id') for p in papers if p.get('id')]
        
        # Return text list and update dropdown/multiselect choices
        return "\n".join(formatted_papers), gr.update(choices=paper_ids)

async def process_paper_ui(paper_id_to_process: str):
    """Calls API to process a paper (assuming it's already in DB/Storage)."""
    if not paper_id_to_process:
        return "Please enter a Paper ID to process."
        
    logger.info(f"UI requesting processing for ID: {paper_id_to_process}")
    
    # Assume the doc processor knows how to find the doc based on ID if not uploaded now
    process_payload = GatewayProcessRequest(paper_id=paper_id_to_process).model_dump(exclude_none=True)
    
    api_result = await call_api_gateway("POST", "/documents/process", payload=process_payload)
    
    if "error" in api_result:
        return f"Error triggering processing: {api_result['error']}"
    else:
        return api_result.get("message", "Processing request accepted.")

async def search_ui(search_query: str):
    """Calls API to perform semantic search."""
    if not search_query:
        return "Please enter a search query.", ""
        
    logger.info(f"UI requesting search for: {search_query}")
    payload = GatewaySearchRequest(query=search_query).model_dump()
    
    api_result = await call_api_gateway("POST", "/search/", payload=payload)
    
    if "error" in api_result:
        return f"Error during search: {api_result['error']}", ""
    else:
        search_response = api_result.get("data", {})
        results = search_response.get("results", [])
        
        if not results:
            return "No relevant results found.", ""
            
        # Format results
        formatted_results = ["**Search Results:**"]
        for item in results:
            formatted_results.append(f"\n---\n**Chunk ID:** {item.get('chunk_id', 'N/A')} (Paper: {item.get('paper_id', 'N/A')}, Score: {item.get('score', 0.0):.2f})")
            formatted_results.append(f"> {item.get('text', '')}")
            
        return "\n".join(formatted_results), "\n".join(formatted_results) # Update both text and markdown

async def analyze_ui(analysis_type: str, paper_ids_str: str, query: Optional[str]):
    """Calls API to perform analysis."""
    if not paper_ids_str and not query:
        return "Please provide Paper IDs or a Query for analysis."
        
    paper_ids = [pid.strip() for pid in paper_ids_str.split(',') if pid.strip()] if paper_ids_str else None
    
    if not paper_ids and not query: # Double check after parsing
        return "Please provide valid Paper IDs or a Query."
        
    logger.info(f"UI requesting analysis: type={analysis_type}, ids={paper_ids}, query={query is not None}")
    
    payload = GatewayAnalysisRequest(
        analysis_type=analysis_type,
        paper_ids=paper_ids,
        query=query if query else None
    ).model_dump(exclude_none=True)
    
    api_result = await call_api_gateway("POST", "/analysis/", payload=payload)
    
    if "error" in api_result:
        return f"Error during analysis: {api_result['error']}"
    else:
        analysis_data = api_result.get("data", {})
        result_text = analysis_data.get("result_text", "Analysis generated, but no text found.")
        cited_sources = analysis_data.get("cited_sources", [])
        
        response = f"**Analysis ({analysis_data.get('analysis_type', 'N/A')}):**\n\n{result_text}"
        
        if cited_sources:
            response += f"\n\n**Cited Sources:** {', '.join(cited_sources)}"
            
        return response

# --- Build Gradio Interface ---

with gr.Blocks(theme=gr.themes.Soft(), title="Intelligent Research Nexus") as demo:
    gr.Markdown("# Intelligent Research Nexus (IRN) Prototype")
    gr.Markdown("Upload PDFs, fetch papers, search content, and generate analysis.")
    
    paper_id_choices = gr.State([]) # Store fetched/uploaded paper IDs
    
    with gr.Tabs():
        with gr.TabItem("Upload & Process"):
            with gr.Row():
                pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
                upload_status = gr.Textbox(label="Status", interactive=False)
            upload_button = gr.Button("Upload and Start Processing")
            
        with gr.TabItem("Fetch & Process"):
            with gr.Row():
                fetch_query = gr.Textbox(label="Fetch Papers Query (e.g., 'transformer models')", placeholder="Enter keywords...")
                fetch_button = gr.Button("Fetch Papers")
            fetched_results_display = gr.Textbox(label="Fetched Papers (IDs and Titles)", interactive=False, lines=5)
            
            with gr.Row():
                # Allow selecting from fetched papers or manual entry
                paper_id_dropdown = gr.Dropdown(label="Select Paper ID to Process", choices=[], interactive=True)
                paper_id_manual = gr.Textbox(label="Or Enter Paper ID Manually")
            process_button = gr.Button("Start Processing Selected/Entered Paper")
            process_status = gr.Textbox(label="Processing Status", interactive=False)
            
        with gr.TabItem("Search & Analyze"):
            search_query_input = gr.Textbox(label="Search Query", placeholder="Ask a question about the processed documents...")
            search_button = gr.Button("Search")
            search_results_display = gr.Markdown(label="Search Results") # Use Markdown for better formatting
            
            gr.Markdown("---") # Separator
            
            analysis_paper_ids = gr.Textbox(label="Paper IDs for Analysis (comma-separated)", placeholder="e.g., arxiv:xxxx.xxxx, upload:...")
            analysis_type_input = gr.Dropdown(label="Analysis Type", choices=["summary", "qa", "compare_methods", "gap_analysis"], value="summary")
            analysis_query_input = gr.Textbox(label="Specific Query for Analysis (optional, needed for QA)", placeholder="e.g., What methodology did paper X use?")
            analyze_button = gr.Button("Generate Analysis")
            analysis_results_display = gr.Markdown(label="Analysis Results")
    
    # --- Connect UI elements to functions ---
    
    upload_button.click(upload_and_process_pdf, inputs=[pdf_upload], outputs=[upload_status, pdf_upload]) # Clear input on success
    
    fetch_button.click(fetch_papers_ui, inputs=[fetch_query], outputs=[fetched_results_display, paper_id_dropdown])
    
    # Handle processing button click - prioritize dropdown, then manual input
    async def process_wrapper(dropdown_id, manual_id):
        paper_id = dropdown_id if dropdown_id else manual_id
        return await process_paper_ui(paper_id)
        
    process_button.click(process_wrapper, inputs=[paper_id_dropdown, paper_id_manual], outputs=[process_status])
    
    search_button.click(search_ui, inputs=[search_query_input], outputs=[search_results_display, search_results_display]) # Update both text and markdown views if needed
    
    analyze_button.click(analyze_ui, inputs=[analysis_type_input, analysis_paper_ids, analysis_query_input], outputs=[analysis_results_display])

# --- Mount Gradio app within FastAPI ---

app = fastapi.FastAPI()

# Add a root endpoint for basic check
@app.get("/")
async def root():
    return {"message": "IRN UI Service is running. Access the Gradio interface at /ui"}

# Mount the Gradio app
app = gr.mount_gradio_app(app, demo, path="/ui")

logger.info("UI Service Ready. Gradio interface available at /ui")

# Note: When running with uvicorn, start this FastAPI app.
# Example: uvicorn services.ui_service.app.main:app --reload --port 7860
