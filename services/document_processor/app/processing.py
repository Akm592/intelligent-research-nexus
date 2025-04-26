# services/document_processor/app/processing.py

from core.models import ProcessRequest, DocumentChunk, generate_chunk_id, PaperMetadata # Import PaperMetadata
from typing import Tuple, Optional, Dict, List, Any
import logging
import asyncio
import io # For handling bytes as files
import httpx
from core.supabase_client import get_supabase_client, PAPERS_TABLE # Import PAPERS_TABLE
from core.config import settings
from pdfminer.high_level import extract_text as pdf_extract_text
from pdfminer.layout import LAParams
from pdfminer.pdfparser import PDFSyntaxError
from supabase import PostgrestAPIError # Import specific error

# --- LangChain Import for Text Splitting ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ------------------------------------------

logger = logging.getLogger("IRN_Core").getChild("DocProcessor").getChild("ProcessingLogic")


# --- Helper function to fetch metadata (including pdf_url) from the DB ---
async def _fetch_paper_metadata_from_db(paper_id: str) -> Optional[PaperMetadata]:
    """Helper function to fetch metadata from the DB."""
    job_prefix = f"[{paper_id}]"
    logger.debug(f"{job_prefix} Fetching metadata from DB for source lookup...")
    try:
        supabase = await get_supabase_client()
        def db_call():
            # Fetch fields needed for download: pdf_url and fallback url
            return supabase.table(PAPERS_TABLE)\
                   .select("id, url, pdf_url")\
                   .eq('id', paper_id)\
                   .limit(1)\
                   .maybe_single()\
                   .execute()

        response = await asyncio.to_thread(db_call)

        if response and response.data:
            logger.debug(f"{job_prefix} Found metadata in DB.")
            # Parse into model (even partially is okay here)
            try:
                return PaperMetadata(**response.data)
            except Exception as parse_err:
                 logger.error(f"{job_prefix} Failed to parse metadata from DB: {parse_err}")
                 return None
        else:
            logger.warning(f"{job_prefix} Metadata not found in DB for ID: {paper_id}")
            return None
    except PostgrestAPIError as e:
        logger.error(f"{job_prefix} Supabase API error fetching metadata: {e.message}", exc_info=False)
        return None
    except Exception as e:
        logger.error(f"{job_prefix} Unexpected error fetching metadata: {e}", exc_info=True)
        return None

# --- Helper function to download content from a URL ---
async def _download_from_url(url: str, paper_id: str) -> Optional[bytes]:
    """Helper function to download content from a URL, handles common errors."""
    job_prefix = f"[{paper_id}]"
    if not url or not url.lower().startswith('http'):
        logger.error(f"{job_prefix} Invalid URL provided for download: {url}")
        return None

    logger.info(f"{job_prefix} Attempting download from URL: {url}")
    try:
        # Use a common user agent
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        async with httpx.AsyncClient(follow_redirects=True, timeout=120.0, headers=headers) as client:
            async with client.stream("GET", url) as response:
                # Raise exceptions for 4xx/5xx client/server errors
                response.raise_for_status()

                content_type = response.headers.get("Content-Type", "").lower()
                # Check if it looks like a PDF or binary stream
                if 'pdf' not in content_type and 'octet-stream' not in content_type:
                    logger.warning(f"{job_prefix} URL {url} returned non-PDF Content-Type: {content_type}. Parsing will likely fail.")
                    # Consider returning None here if you *only* want to process PDFs from URLs

                content = await response.aread()

        if not content:
             logger.error(f"{job_prefix} Downloaded 0 bytes from URL {url}.")
             return None

        logger.info(f"{job_prefix} Successfully downloaded {len(content)} bytes from URL.")
        return content

    except httpx.RequestError as e:
        logger.error(f"{job_prefix} Network error downloading from URL {url}: {e}", exc_info=False)
        return None
    except httpx.HTTPStatusError as e:
        # Log specific HTTP errors
        logger.error(f"{job_prefix} HTTP error {e.response.status_code} downloading from URL {url}: {e.response.text[:200]}...", exc_info=False)
        return None
    except Exception as e:
        logger.error(f"{job_prefix} Unexpected error downloading from URL {url}: {e}", exc_info=True)
        return None

# --- Main function to get document content based on request ---
async def get_document_content(request: ProcessRequest) -> Tuple[Optional[bytes], Optional[Dict]]:
    """
    Retrieves document content based on the ProcessRequest.
    Priority order:
    1. Supabase Storage (bucket_name & object_name from request).
    2. Explicit `source_url` from request.
    3. Database Lookup (using paper_id from request):
       a. Try `pdf_url` field from the 'papers' table.
       b. Try `url` field from the 'papers' table (as fallback).
    """
    paper_id = request.paper_id
    job_prefix = f"[{paper_id}]"
    logger.info(f"{job_prefix} Resolving document content source...")
    metadata = {"paper_id": paper_id} # Base metadata to return

    # --- Priority 1: Supabase Storage (from Upload flow) ---
    bucket = request.bucket_name # Use bucket name directly from request if provided
    object_name = request.object_name
    # Check if BOTH bucket and object name are meaningfully provided
    if bucket and object_name:
        logger.info(f"{job_prefix} Source specified: Supabase Storage. Bucket='{bucket}', Object='{object_name}'")
        try:
            supabase = await get_supabase_client()
            # Use asyncio.to_thread if supabase storage methods are sync
            def storage_download():
                 return supabase.storage.from_(bucket).download(object_name)
            storage_response = await asyncio.to_thread(storage_download)

            if isinstance(storage_response, bytes) and len(storage_response) > 0:
                content = storage_response
                logger.info(f"{job_prefix} Successfully downloaded {len(content)} bytes from Supabase Storage.")
                metadata.update({"source_type": "supabase_storage", "bucket": bucket, "object_name": object_name})
                return content, metadata
            elif isinstance(storage_response, bytes): # File exists but is empty
                 logger.error(f"{job_prefix} Supabase storage download returned 0 bytes for {object_name}.")
                 return None, None # Treat empty file as failure
            else: # Response indicates error or file not found
                 logger.error(f"{job_prefix} Supabase storage download failed for {object_name}. Response type: {type(storage_response)}.")
                 return None, None
        except Exception as e:
            # Catch potential exceptions like file not found from Supabase
            logger.error(f"{job_prefix} Error downloading from Supabase Storage: {e}", exc_info=True)
            return None, None # Treat storage error as fatal for this path

    # --- Priority 2: Explicit source_url from ProcessRequest ---
    elif request.source_url:
        logger.info(f"{job_prefix} Source specified: Explicit source_url from request.")
        content = await _download_from_url(request.source_url, paper_id)
        if content:
            metadata.update({"source_type": "explicit_url", "url": request.source_url})
            return content, metadata
        else:
             logger.error(f"{job_prefix} Failed to download from explicit source_url: {request.source_url}")
             # Do not fallback further if an explicit URL was given and failed
             return None, None

    # --- Priority 3 & 4: Lookup URL from DB using paper_id ---
    else:
        logger.info(f"{job_prefix} Source specified: paper_id only. Looking up URL in database...")
        paper_meta = await _fetch_paper_metadata_from_db(paper_id)

        if not paper_meta:
            logger.error(f"{job_prefix} Cannot find metadata in DB for paper_id {paper_id}. Unable to determine download source.")
            return None, None

        content = None
        # --- Try pdf_url first ---
        if paper_meta.pdf_url:
            logger.info(f"{job_prefix} Found pdf_url in DB. Attempting download from: {paper_meta.pdf_url}")
            content = await _download_from_url(paper_meta.pdf_url, paper_id)
            if content:
                metadata.update({"source_type": "db_pdf_url", "url": paper_meta.pdf_url})
                # Successfully downloaded PDF, return it
                return content, metadata
            else:
                # Log failure but continue to fallback url if pdf_url failed
                logger.warning(f"{job_prefix} Failed to download from DB pdf_url: {paper_meta.pdf_url}. Trying fallback url.")

        # --- Fallback to general 'url' if pdf_url failed or was missing ---
        if paper_meta.url:
            logger.info(f"{job_prefix} Found fallback url in DB. Attempting download from: {paper_meta.url}")
            content = await _download_from_url(paper_meta.url, paper_id)
            if content:
                 # Log clearly that we used the fallback and it might not be a PDF
                 logger.warning(f"{job_prefix} Successfully downloaded using FALLBACK url: {paper_meta.url}. Content might not be PDF.")
                 metadata.update({"source_type": "db_fallback_url", "url": paper_meta.url})
                 return content, metadata
            else:
                logger.error(f"{job_prefix} Failed to download from DB fallback url: {paper_meta.url}.")
                # Both pdf_url (if tried) and fallback url failed
                return None, None
        else:
            # This case means pdf_url was tried and failed (or was None), and url was also None
            logger.error(f"{job_prefix} No suitable URL (pdf_url or url) found in DB metadata after checking.")
            return None, None


# --- Main PDF Parsing and Chunking Function (using LangChain)---
async def parse_and_chunk(paper_id: str, content: bytes, metadata: Dict[str, Any]) -> List[DocumentChunk]:
    """Parses PDF content using pdfminer.six and chunks text using LangChain."""
    job_prefix = f"[{paper_id}]"
    logger.info(f"{job_prefix} Starting PDF parsing (source: {metadata.get('source_type', 'unknown')}, content length: {len(content)} bytes)...")

    extracted_text = ""
    # --- PDF Parsing Stage ---
    try:
        pdf_file = io.BytesIO(content)
        # Run potentially blocking pdfminer in thread pool
        extracted_text = await asyncio.to_thread(
            pdf_extract_text, pdf_file, laparams=LAParams() # Use default layout analysis
        )
        if not extracted_text or not extracted_text.strip():
            source_type = metadata.get('source_type')
            # Give specific warnings based on likely source type
            if source_type == 'db_fallback_url' or (source_type == 'explicit_url' and not metadata.get('url','').lower().endswith('.pdf')):
                 logger.warning(f"{job_prefix} pdfminer.six extracted no text. Input was likely HTML or non-PDF (source: {source_type}).")
            else:
                 logger.warning(f"{job_prefix} pdfminer.six extracted no text content. PDF might be image-based, empty, or corrupted.")
            return [] # Cannot proceed without text
        logger.info(f"{job_prefix} Successfully extracted ~{len(extracted_text)} characters.")

    except PDFSyntaxError as e:
        source_type = metadata.get('source_type')
        # Provide better error message if input was likely not PDF
        if source_type == 'db_fallback_url' or (source_type == 'explicit_url' and not metadata.get('url','').lower().endswith('.pdf')):
             logger.error(f"{job_prefix} Invalid PDF syntax. Input was likely HTML or non-PDF (source: {source_type}). Error: {e}", exc_info=False)
        else:
             logger.error(f"{job_prefix} Invalid PDF syntax. Cannot parse PDF. Error: {e}", exc_info=False)
        return [] # Cannot proceed
    except Exception as e:
        logger.error(f"{job_prefix} Unexpected error during PDF parsing using pdfminer.six: {e}", exc_info=True)
        return [] # Cannot proceed

    # --- Text Chunking Stage using LangChain ---
    logger.info(f"{job_prefix} Starting text chunking with LangChain. Size={settings.PARSER_CHUNK_SIZE}, Overlap={settings.PARSER_CHUNK_OVERLAP}")

    try:
        # Initialize LangChain's text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.PARSER_CHUNK_SIZE,
            chunk_overlap=settings.PARSER_CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False, # Treat separators as simple strings
            separators=settings.PARSER_SEPARATORS, # Use separators from config
            # keep_separator=True # Optional: Set True if separators are meaningful boundaries you want preserved
        )

        # Split the text (this is synchronous, run in thread)
        text_chunks_str = await asyncio.to_thread(text_splitter.split_text, extracted_text)

    except Exception as e:
        logger.error(f"{job_prefix} Error during LangChain text chunking: {e}", exc_info=True)
        return [] # Return empty list on chunking error

    if not text_chunks_str:
         logger.warning(f"{job_prefix} LangChain text chunking resulted in zero chunks.")
         return []

    # --- Create DocumentChunk objects ---
    final_chunks: List[DocumentChunk] = []
    for i, text_chunk_str in enumerate(text_chunks_str):
        # Basic check for empty strings after splitting/stripping
        current_text_chunk = text_chunk_str.strip()
        if not current_text_chunk:
            continue # Skip empty chunks

        chunk_id = generate_chunk_id(paper_id, i)
        # Copy metadata obtained from get_document_content (contains source info)
        chunk_meta = metadata.copy()
        # Add chunk-specific metadata
        chunk_meta.update({
            "chunk_index": i,
            "char_count": len(current_text_chunk), # Use length of stripped chunk
        })

        final_chunks.append(DocumentChunk(
            chunk_id=chunk_id,
            paper_id=paper_id,
            text=current_text_chunk, # Use stripped chunk text
            metadata=chunk_meta
            # embedding field will be added later by vector service
        ))

    logger.info(f"{job_prefix} Successfully created {len(final_chunks)} chunks using LangChain splitter.")
    return final_chunks