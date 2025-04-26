# services/ui_service/app/main.py

import gradio as gr
import fastapi
import httpx
import logging
import os
import uuid
import time
import asyncio # <-- Import asyncio
from core.config import settings
from core.supabase_client import get_supabase_client
from core.models import (
    GatewayFetchRequest, GatewayProcessRequest, GatewaySearchRequest,
    GatewayAnalysisRequest, PaperMetadata, SearchResultItem, AnalysisResult, ProcessRequest, SearchResponse
)
from typing import Optional, List, Dict, Any
from pydantic import ValidationError

# Setup logger
logger = logging.getLogger("IRN_Core").getChild("UIService")

# Global httpx client for calling API Gateway
api_client = httpx.AsyncClient(base_url=settings.API_GATEWAY_URL, timeout=120.0)

# --- Helper Functions for API Calls (Keep as is) ---
async def call_api_gateway(method: str, endpoint: str, payload: Optional[Dict] = None) -> Dict:
    # ... (Keep implementation from previous answer) ...
    """Helper to call the API Gateway and handle responses."""
    try:
        if method.upper() == "POST": response = await api_client.post(endpoint, json=payload)
        elif method.upper() == "GET": response = await api_client.get(endpoint)
        else: logger.error(f"Unsupported HTTP method requested: {method}"); return {"error": f"Unsupported method: {method}"}
        response.raise_for_status(); api_response = response.json()
        if api_response.get("status") == "success": return {"data": api_response.get("data"), "message": api_response.get("message")}
        else:
            error_msg = api_response.get("message", "Unknown API error"); gateway_status = response.status_code
            logger.error(f"API Gateway returned error at {endpoint}: {error_msg} (Status: {gateway_status})")
            return {"error": f"API Error ({gateway_status}): {error_msg}"}
    except httpx.HTTPStatusError as e:
        error_detail = f"API Error ({e.response.status_code})"; downstream_error = e.response.text
        try: downstream_error = e.response.json().get('detail', e.response.text)
        except Exception: pass
        logger.error(f"{error_detail} calling {endpoint}: {downstream_error}")
        return {"error": f"{error_detail} from '{e.request.url.path}': {downstream_error}"}
    except httpx.RequestError as e: logger.error(f"Network error calling API Gateway endpoint {endpoint}: {e}"); return {"error": f"Cannot reach API Gateway at {settings.API_GATEWAY_URL}"}
    except Exception as e: logger.error(f"Unexpected error calling API Gateway endpoint {endpoint}: {e}", exc_info=True); return {"error": f"An unexpected error occurred while contacting the API Gateway: {e}"}


# --- Gradio Interface Functions ---

async def upload_and_process_pdf(file_obj, progress=gr.Progress(track_tqdm=True)):
    """Handles PDF upload, stores in Supabase, triggers processing."""
    if file_obj is None:
        return "Please upload a PDF file.", gr.update(value=None)

    # file_obj is a TemporaryFileWrapper in Gradio >= 4
    # Access the temporary file path via file_obj.name
    file_path = file_obj.name
    # Get the original filename the user uploaded
    # Gradio might store original name differently depending on version,
    # but often basename of temp path is a good guess or stored in attributes
    original_filename = getattr(file_obj, 'orig_name', os.path.basename(file_path))
    logger.info(f"Received file upload: '{original_filename}' (temp path: {file_path})")

    if not os.path.exists(file_path):
         logger.error(f"Temporary file path does not exist: {file_path}")
         return "Error: Uploaded file not found on server.", gr.update(value=None)

    progress(0, desc="Preparing upload...")

    # Generate a unique paper ID
    filename_base = os.path.splitext(original_filename)[0]
    filename_prefix = ''.join(c for c in filename_base if c.isalnum() or c in ['_', '-'])[:30] or "file"
    paper_id = f"upload:{filename_prefix}_{uuid.uuid4().hex[:8]}"
    object_name = f"uploads/{paper_id}/{original_filename}" # Include original filename for clarity

    bucket_name = settings.DOC_STORAGE_BUCKET

    try:
        # --- Step 1: Upload to Supabase Storage using SERVICE KEY ---
        progress(0.1, desc=f"Getting Supabase client (Service Key)...")
        try:
            supabase = await get_supabase_client(use_service_key=True)
            logger.info(f"Successfully obtained Supabase client (Service Key) for upload.")
        except ValueError as e:
             logger.error(f"Configuration error: {e}. SUPABASE_SERVICE_KEY might be missing in .env")
             return f"Configuration error: Cannot initialize storage client. Check service logs.", gr.update(value=None)
        except Exception as e:
             logger.error(f"Failed to get Supabase client: {e}", exc_info=True)
             return "Internal error: Failed to get storage client.", gr.update(value=None)

        logger.info(f"Uploading '{original_filename}' to Supabase Storage: Bucket='{bucket_name}', Object='{object_name}'")
        progress(0.2, desc=f"Uploading '{original_filename}' to Storage...")

        # --- Define the synchronous upload call ---
        # Use 'with open' ensures the file handle is closed even if errors occur
        try:
            with open(file_path, 'rb') as f:
                # This inner function reads the file and calls the sync upload method
                def do_upload():
                    # Read file content inside the thread function if file handle might close
                    file_content = f.read()
                    logger.debug(f"Read {len(file_content)} bytes from temp file for upload.")
                    return supabase.storage.from_(bucket_name).upload(
                        path=object_name,
                        file=file_content, # Pass content bytes directly
                        file_options={"content-type": "application/pdf", "upsert": "false"}
                    )
                # --- Run the synchronous upload in a thread ---
                logger.debug("Executing storage upload in background thread...")
                upload_response = await asyncio.to_thread(do_upload)
                logger.debug(f"Storage upload thread finished. Response type: {type(upload_response)}")
                # Check response type and potentially content if needed, but lack of error is main indicator
        except FileNotFoundError:
             logger.error(f"Temporary file not found during upload attempt: {file_path}")
             return "Error: Uploaded file disappeared before upload.", gr.update(value=None)
        except Exception as upload_err: # Catch errors during file read or upload thread
             logger.error(f"Error during file read or upload execution: {upload_err}", exc_info=True)
             return f"Error during file upload: {upload_err}", gr.update(value=None)


        # Supabase upload doesn't throw easily on failure sometimes, might need checks
        # For now, assume success if no exception was raised by asyncio.to_thread/do_upload
        logger.info(f"Successfully sent upload request for '{original_filename}' to Supabase Storage as '{object_name}' for Paper ID: {paper_id}")
        progress(0.6, desc="Upload successful. Triggering processing...")

        # --- Step 2: Trigger processing via API Gateway ---
        logger.info(f"Triggering processing via API Gateway for paper ID: {paper_id}")
        process_payload = GatewayProcessRequest(
            paper_id=paper_id,
            bucket_name=bucket_name,
            object_name=object_name # Pass bucket/object name
        ).model_dump(exclude_none=True)

        api_result = await call_api_gateway("POST", "/documents/process", payload=process_payload)
        progress(0.9, desc="Processing request sent.")

        if "error" in api_result:
            logger.error(f"Failed to trigger processing for {paper_id} via API Gateway: {api_result['error']}")
            # Consider trying to delete the uploaded file if trigger fails? Complex cleanup.
            return f"File uploaded, but failed to start processing: {api_result['error']}", gr.update(value=None)
        else:
            status_message = api_result.get("message", "Processing started successfully.")
            logger.info(f"Processing trigger successful for {paper_id}: {status_message}")
            progress(1, desc="Done.")
            return f"{status_message}\nPaper ID: {paper_id}", gr.update(value=None) # Clear file input

    except Exception as e:
        # Catch broader errors (e.g., getting Supabase client, other unexpected issues)
        logger.error(f"Error during upload/processing trigger workflow for {original_filename} (Paper ID: {paper_id}): {e}", exc_info=True)
        error_type = type(e).__name__
        progress(1, desc="Error occurred.")
        return f"An unexpected error occurred ({error_type}): {e}. Check service logs.", gr.update(value=None)
    # Ensure the temporary file created by Gradio is cleaned up if possible
    # Gradio usually handles this, but good to be aware of.


# --- fetch_papers_ui (Keep implementation from previous answer) ---
async def fetch_papers_ui(query: str, progress=gr.Progress(track_tqdm=True)):
    # ... (Keep implementation from previous answer) ...
    """Calls API to fetch papers and returns data needed for processing (including pdf_url)."""
    if not query: return "Please enter a search query.", [], {}
    logger.info(f"UI requesting fetch for query: {query}"); progress(0, desc=f"Fetching papers for '{query[:30]}...'")
    payload = GatewayFetchRequest(query=query).model_dump()
    api_result = await call_api_gateway("POST", "/documents/fetch", payload=payload)
    progress(0.8, desc="Processing fetch results..."); id_to_data_map = {}
    if "error" in api_result: progress(1, desc="Error."); return f"Error fetching papers: {api_result['error']}", [], {}
    else:
        papers_data = api_result.get("data", [])
        if not papers_data: progress(1, desc="No results."); return "No papers found for this query.", [], {}
        try:
            papers = [PaperMetadata(**p) for p in papers_data]
            id_to_data_map = { p.id: {"url": p.url, "pdf_url": p.pdf_url} for p in papers if p.id }
            logger.info(f"Stored URL/PDF_URL mapping for {len(id_to_data_map)} fetched papers.")
        except Exception as val_err: logger.error(f"Failed to validate fetched paper data: {val_err}"); progress(1, desc="Validation Error."); return f"Received invalid paper data from API.", [], {}
        formatted_papers = [f"ID: {p.id} - Title: {p.title or 'N/A'}" for p in papers]; paper_ids = [p.id for p in papers]
        logger.info(f"Successfully fetched {len(papers)} papers for query '{query}'."); progress(1, desc="Done.")
        return "\n".join(formatted_papers), gr.update(choices=paper_ids, value=None), id_to_data_map

# --- process_paper_ui (Keep implementation from previous answer) ---
async def process_paper_ui(paper_id_to_process: str, fetched_data: Dict[str, Dict], progress=gr.Progress(track_tqdm=True)):
    # ... (Keep implementation from previous answer) ...
    """Calls API to process a paper. If it's a fetched paper ID found in fetched_data, includes the pdf_url as source_url."""
    if fetched_data is None: logger.warning("process_paper_ui called with None fetched_data."); fetched_data = {}
    if not paper_id_to_process: return "Please select or enter a Paper ID to process."
    logger.info(f"UI requesting processing for ID: {paper_id_to_process}"); progress(0, desc=f"Sending processing request for {paper_id_to_process}...")
    pdf_source_url = None
    if fetched_data and paper_id_to_process in fetched_data:
        paper_info = fetched_data.get(paper_id_to_process, {}); pdf_source_url = paper_info.get('pdf_url')
        if pdf_source_url: logger.info(f"Found pdf_url for fetched paper {paper_id_to_process}: {pdf_source_url}")
        else: logger.warning(f"Fetched paper {paper_id_to_process} has no 'pdf_url' in stored UI data. Processor must look up.")
    else: logger.info(f"Processing manually entered ID '{paper_id_to_process}' or uploaded paper. No URL provided by UI.")
    process_payload = GatewayProcessRequest(paper_id=paper_id_to_process, source_url=pdf_source_url).model_dump(exclude_none=True)
    logger.debug(f"Sending process request to gateway: {process_payload}")
    try: ProcessRequest(**process_payload); logger.debug("Process request payload structure seems valid locally.")
    except ValidationError as local_val_err: logger.error(f"Local validation failed for process request payload structure: {local_val_err}"); progress(1, desc="Error."); return f"Internal UI Error: Invalid data prepared for processing request. {local_val_err}"
    api_result = await call_api_gateway("POST", "/documents/process", payload=process_payload); progress(0.9, desc="Request sent.")
    if "error" in api_result: logger.error(f"Failed to trigger processing for {paper_id_to_process}: {api_result['error']}"); progress(1, desc="Error."); return f"Error triggering processing: {api_result['error']}"
    else: logger.info(f"Processing trigger successful for {paper_id_to_process}"); progress(1, desc="Done."); return api_result.get("message", "Processing request accepted.")

# --- search_ui (Keep implementation from previous answer) ---
async def search_ui(search_query: str, progress=gr.Progress(track_tqdm=True)):
    # ... (Keep implementation from previous answer) ...
    """Calls API to perform semantic search."""
    if not search_query: return "Please enter a search query.", ""
    logger.info(f"UI requesting search for: {search_query}"); progress(0, desc=f"Searching for '{search_query[:30]}...'")
    payload = GatewaySearchRequest(query=search_query).model_dump()
    api_result = await call_api_gateway("POST", "/search/", payload=payload); progress(0.8, desc="Processing search results...")
    if "error" in api_result: logger.error(f"Search failed: {api_result['error']}"); progress(1, desc="Error."); return f"**Error during search:**\n\n{api_result['error']}", ""
    else:
        search_response_data = api_result.get("data", {})
        try: search_response = SearchResponse(**search_response_data); results = search_response.results
        except Exception as val_err: logger.error(f"Failed to validate search response data: {val_err}"); progress(1, desc="Validation Error."); return "**Error:** Received invalid search result data from API.", ""
        if not results: logger.info("Search returned no results."); progress(1, desc="No results."); return "**No relevant results found.**", ""
        formatted_results = ["**Search Results:**\n"]
        for i, item in enumerate(results):
             chunk_id = getattr(item, 'chunk_id', 'N/A'); paper_id = getattr(item, 'paper_id', 'N/A'); score = getattr(item, 'score', 0.0); text = getattr(item, 'text', '')
             formatted_results.append(f"---"); formatted_results.append(f"**Result {i+1}:** Chunk `{chunk_id}`"); formatted_results.append(f"   (Source Paper: `{paper_id}`, Similarity Score: {score:.3f})"); formatted_results.append(f"> {text.replace('>','>')}"); formatted_results.append("")
        final_markdown = "\n".join(formatted_results); logger.info(f"Search successful, returning {len(results)} results."); progress(1, desc="Done.")
        return final_markdown, final_markdown

# --- analyze_ui (Keep implementation from previous answer) ---
async def analyze_ui(analysis_type: str, paper_ids_str: str, query: Optional[str], progress=gr.Progress(track_tqdm=True)):
    # ... (Keep implementation from previous answer) ...
    """Calls API to perform analysis."""
    if not paper_ids_str and analysis_type != "qa":
        if not query: return "Please provide Paper IDs (for most analysis types) or a Query (especially for QA)."
        logger.warning(f"Analysis type '{analysis_type}' requested without paper IDs, proceeding with query only."); paper_ids = None
    elif paper_ids_str:
        paper_ids = [pid.strip() for pid in paper_ids_str.split(',') if pid.strip()]
        if not paper_ids: return "Invalid Paper IDs provided. Please use comma-separated IDs."
    else: paper_ids = None
    if not paper_ids and not query: return "Please provide valid Paper IDs or a Query."
    if analysis_type == "qa" and not query: return "A Query is required for 'qa' analysis type."
    log_query_present = query is not None and query.strip() != ""
    logger.info(f"UI requesting analysis: type={analysis_type}, ids={paper_ids}, query_present={log_query_present}"); progress(0, desc=f"Sending analysis request ({analysis_type})...")
    payload = GatewayAnalysisRequest(analysis_type=analysis_type, paper_ids=paper_ids, query=query if log_query_present else None).model_dump(exclude_none=True)
    api_result = await call_api_gateway("POST", "/analysis/", payload=payload); progress(0.8, desc="Processing analysis results...")
    if "error" in api_result: logger.error(f"Analysis failed: {api_result['error']}"); progress(1, desc="Error."); return f"**Error during analysis:**\n\n{api_result['error']}"
    else:
        analysis_data = api_result.get("data", {});
        try: analysis_result = AnalysisResult(**analysis_data); result_text = analysis_result.result_text; cited_sources = analysis_result.cited_sources; actual_analysis_type = analysis_result.analysis_type
        except Exception as val_err: logger.error(f"Failed to validate analysis response data: {val_err}"); progress(1, desc="Validation Error."); return "**Error:** Received invalid analysis result data from API."
        if not result_text or result_text.strip() == "": logger.warning("RAG chain returned an empty result_text/answer."); result_text = "(Analysis completed, but the result was empty.)"
        response = f"**Analysis ({actual_analysis_type}):**\n\n{result_text.strip()}"
        if cited_sources: sources_str = "\n".join([f"- `{source}`" for source in cited_sources]); response += f"\n\n**Cited Sources:**\n{sources_str}"
        logger.info(f"Analysis successful ({actual_analysis_type})."); progress(1, desc="Done.")
        return response


# --- Build Gradio Interface (Keep implementation from previous answer) ---
with gr.Blocks(theme=gr.themes.Soft(), title="Intelligent Research Nexus") as demo:
    # ... (Keep implementation from previous answer) ...
    gr.Markdown("# Intelligent Research Nexus (IRN)")
    gr.Markdown("Upload PDFs, fetch external papers, perform semantic search, and generate AI-driven analyses.")
    fetched_papers_metadata = gr.State({}) # Stores { paper_id: {"url": url, "pdf_url": pdf_url} }
    with gr.Tabs():
        with gr.TabItem("1. Upload & Process"):
            gr.Markdown("Upload a PDF document...")
            with gr.Row():
                with gr.Column(scale=2): pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"], scale=1); upload_button = gr.Button("⬆️ Upload and Start Processing", variant="primary", scale=1)
                with gr.Column(scale=3): upload_status = gr.Textbox(label="Status / Paper ID", interactive=False, lines=3, placeholder="Upload status...")
        with gr.TabItem("2. Fetch & Process"):
             gr.Markdown("Fetch paper metadata...")
             with gr.Row(): fetch_query = gr.Textbox(label="Fetch Papers Query", placeholder="e.g., 'llm'"); fetch_button = gr.Button("🔍 Fetch Papers", variant="primary")
             fetched_results_display = gr.Textbox(label="Fetched Papers (Select ID below)", interactive=False, lines=5, placeholder="Fetched papers...")
             gr.Markdown("Select a fetched paper ID or enter one manually to start processing.")
             with gr.Row(): paper_id_dropdown = gr.Dropdown(label="Select Fetched Paper ID", choices=[], interactive=True, allow_custom_value=False); paper_id_manual = gr.Textbox(label="Or Enter Existing Paper ID", placeholder="e.g., arxiv:...")
             with gr.Row(): process_button = gr.Button("⚙️ Start Processing Paper", variant="primary"); process_status = gr.Textbox(label="Processing Status", interactive=False, placeholder="Status...")
        with gr.TabItem("3. Search & Analyze"):
            gr.Markdown("Perform semantic search or generate analyses...")
            with gr.Accordion("Semantic Search", open=True):
                search_query_input = gr.Textbox(label="Search Query", placeholder="Ask a question..."); search_button = gr.Button("🔎 Search Documents", variant="primary")
                search_results_display = gr.Markdown(label="Search Results", value="Search results...")
            with gr.Accordion("AI Analysis", open=True):
                gr.Markdown("Select papers (by ID) and the type of analysis...")
                analysis_paper_ids = gr.Textbox(label="Paper IDs for Analysis (comma-separated)", placeholder="e.g., arxiv:...")
                analysis_type_input = gr.Dropdown(label="Analysis Type", choices=["summary", "qa", "compare_methods", "gap_analysis"], value="summary")
                analysis_query_input = gr.Textbox(label="Specific Query for Analysis (required for QA)", placeholder="e.g., What method...")
                analyze_button = gr.Button("💡 Generate Analysis", variant="primary")
                analysis_results_display = gr.Markdown(label="Analysis Results", value="Analysis results...")

    # --- Connect UI elements to functions (Keep implementation from previous answer) ---
    upload_button.click(upload_and_process_pdf, inputs=[pdf_upload], outputs=[upload_status, pdf_upload])
    fetch_button.click(fetch_papers_ui, inputs=[fetch_query], outputs=[fetched_results_display, paper_id_dropdown, fetched_papers_metadata])
    async def process_wrapper(dropdown_id, manual_id, current_fetched_metadata, progress=gr.Progress(track_tqdm=True)):
        paper_id = dropdown_id if dropdown_id and str(dropdown_id).strip() else manual_id; paper_id_str = str(paper_id).strip() if paper_id else ""
        if not paper_id_str: return "Please select a fetched Paper ID or enter one manually."
        return await process_paper_ui(paper_id_str, current_fetched_metadata, progress=progress)
    process_button.click(process_wrapper, inputs=[paper_id_dropdown, paper_id_manual, fetched_papers_metadata], outputs=[process_status])
    search_button.click(search_ui, inputs=[search_query_input], outputs=[search_results_display])
    analyze_button.click(analyze_ui, inputs=[analysis_type_input, analysis_paper_ids, analysis_query_input], outputs=[analysis_results_display])


# --- Mount Gradio app within FastAPI (Keep as is) ---
app = fastapi.FastAPI()
@app.get("/")
async def root():
    return {"message": "IRN UI Service is running. Access the Gradio interface at /ui"}
app = gr.mount_gradio_app(app, demo, path="/ui" )
logger.info("UI Service Ready. Gradio interface available at /ui")