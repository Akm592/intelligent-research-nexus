# services/document_processor/app/processing.py
from core.models import ProcessRequest, DocumentChunk, generate_chunk_id
from typing import Tuple, Optional, Dict, List, Any
import logging
import asyncio
import io # For handling bytes as files
import httpx
from core.supabase_client import get_supabase_client
from core.config import settings
from pdfminer.high_level import extract_text as pdf_extract_text # Import pdfminer
from pdfminer.layout import LAParams
from pdfminer.pdfparser import PDFSyntaxError

logger = logging.getLogger("IRN_Core").getChild("DocProcessor").getChild("ProcessingLogic")


async def get_document_content(request: ProcessRequest) -> Tuple[Optional[bytes], Optional[Dict]]:
    """
    Retrieves document content from Supabase Storage or a URL.
    Removes the local file placeholder for production readiness.
    """
    paper_id = request.paper_id
    logger.info(f"Retrieving document content for {paper_id}...")
    metadata = {"paper_id": paper_id}

    # --- Priority 1: Supabase Storage ---
    bucket = request.bucket_name or settings.DOC_STORAGE_BUCKET
    object_name = request.object_name

    if bucket and object_name:
        logger.info(f"Attempting download from Supabase Storage: Bucket='{bucket}', Object='{object_name}'")
        try:
            supabase = await get_supabase_client()
            storage_response = await supabase.storage.from_(bucket).download(object_name)

            if isinstance(storage_response, bytes) and len(storage_response) > 0:
                content = storage_response
                logger.info(f"Successfully downloaded {len(content)} bytes from Supabase Storage for {paper_id}.")
                metadata.update({"source_type": "supabase_storage", "bucket": bucket, "object_name": object_name})
                return content, metadata
            elif isinstance(storage_response, bytes) and len(storage_response) == 0:
                 logger.error(f"Supabase storage download returned 0 bytes for {object_name}. Check if file is empty.")
                 return None, None # Treat empty file as failure to process
            else:
                 logger.error(f"Supabase storage download did not return bytes for {object_name}. Response type: {type(storage_response)}. Check object existence and permissions.")
                 return None, None
        except Exception as e:
            # Consider catching specific Supabase exceptions if available/documented
            logger.error(f"Error downloading from Supabase Storage (Bucket: {bucket}, Object: {object_name}) for {paper_id}: {e}", exc_info=True)
            return None, None

    # --- Priority 2: Source URL ---
    elif request.source_url:
        logger.info(f"Attempting download from URL: {request.source_url}")
        try:
            # Use a temporary user agent
            headers = {'User-Agent': 'IRNBot/1.0 (+http://example.com/bot)'} # Be polite
            async with httpx.AsyncClient(follow_redirects=True, timeout=60.0, headers=headers) as client:
                async with client.stream("GET", request.source_url) as response:
                    response.raise_for_status() # Check for HTTP errors early

                    # Optional: Check content type if possible (might require reading start of stream)
                    content_type = response.headers.get("Content-Type", "").lower()
                    if 'pdf' not in content_type and content_type != 'application/octet-stream': # Allow generic stream
                        logger.warning(f"URL {request.source_url} has unexpected Content-Type: {content_type}. Attempting processing anyway.")
                        # Decide whether to reject non-PDFs here

                    content = await response.aread() # Read full response body as bytes

            if not content:
                 logger.error(f"Downloaded 0 bytes from URL {request.source_url} for {paper_id}.")
                 return None, None

            logger.info(f"Successfully downloaded {len(content)} bytes from URL for {paper_id}.")
            metadata.update({"source_type": "url", "url": request.source_url})
            return content, metadata

        except httpx.RequestError as e:
            logger.error(f"Network error downloading from URL {request.source_url} for {paper_id}: {e}", exc_info=False)
            return None, None
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code} downloading from URL {request.source_url} for {paper_id}", exc_info=False)
            return None, None
        except Exception as e:
            logger.error(f"Unexpected error downloading from URL {request.source_url} for {paper_id}: {e}", exc_info=True)
            return None, None

    # --- No Source Found ---
    else:
        logger.error(f"No content source specified (Supabase Storage or URL) for paper {paper_id}. Cannot process.")
        return None, None


def _recursive_split(text: str, separators: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Recursively splits text using a list of separators.
    Similar to LangChain's RecursiveCharacterTextSplitter.
    """
    final_chunks = []
    if not text:
        return final_chunks

    # Use the first separator found in the list
    separator = separators[-1] # Default to last separator if none found
    for s in separators:
        if s in text:
            separator = s
            break
        elif s == "" and separator == separators[-1]: # Handle empty string separator only if others not found
             separator = s

    # Split by the chosen separator
    if separator:
        splits = text.split(separator)
    else:
        # If no separator (or empty string used), split by individual characters
        splits = list(text)

    current_chunk = ""
    for i, part in enumerate(splits):
        # Add separator back unless it was an empty string or we are at the end
        part_to_add = part + (separator if separator and i < len(splits) - 1 else "")

        # If adding the part exceeds chunk size (considering overlap for next chunk)
        if len(current_chunk) + len(part_to_add) > chunk_size:
            # Add the current chunk if it's not empty
            if current_chunk.strip():
                 final_chunks.append(current_chunk.strip())

            # Start new chunk, potentially including overlap from the previous one
            # Simple overlap: take last 'chunk_overlap' chars of current_chunk
            # More robust overlap might consider sentence boundaries.
            overlap_text = current_chunk[-chunk_overlap:] if chunk_overlap > 0 else ""

            # Check if the new part itself is larger than the chunk size
            if len(part_to_add) > chunk_size:
                 logger.warning(f"Split part is larger than chunk size ({len(part_to_add)} > {chunk_size}). Splitting further.")
                 # Recursively split the large part using the remaining separators
                 sub_chunks = _recursive_split(part_to_add, separators[separators.index(separator):], chunk_size, chunk_overlap)
                 final_chunks.extend(sub_chunks)
                 current_chunk = "" # Reset current chunk after handling large part
            else:
                current_chunk = overlap_text + part_to_add

        else:
            current_chunk += part_to_add

    # Add the last remaining chunk
    if current_chunk.strip():
        final_chunks.append(current_chunk.strip())

    # Optional: Secondary check to ensure no chunk vastly exceeds size due to overlap logic
    # final_chunks = [chunk[:chunk_size*2] for chunk in final_chunks] # Simple truncation if needed

    return final_chunks


async def parse_and_chunk(paper_id: str, content: bytes, metadata: Dict[str, Any]) -> List[DocumentChunk]:
    """
    Parses PDF content using pdfminer.six and chunks the text using a
    recursive character splitting strategy.
    """
    logger.info(f"Starting PDF parsing for {paper_id} (source: {metadata.get('source_type', 'unknown')}, content length: {len(content)} bytes)...")

    # --- PDF Parsing using pdfminer.six ---
    extracted_text = ""
    try:
        # pdfminer.six works with file-like objects or filenames
        pdf_file = io.BytesIO(content)
        # Run extraction in a thread pool as it can be CPU-bound
        extracted_text = await asyncio.to_thread(
            pdf_extract_text,
            pdf_file,
            laparams=LAParams() # Use default layout analysis parameters
            # You might experiment with parameters like char_margin, line_margin, word_margin
        )
        if not extracted_text or not extracted_text.strip():
            logger.warning(f"pdfminer.six extracted no text content from {paper_id}. The PDF might be image-based or empty.")
            # Return empty list, processing cannot continue
            return []
        logger.info(f"Successfully extracted ~{len(extracted_text)} characters from PDF {paper_id}.")

    except PDFSyntaxError as e:
        logger.error(f"Invalid PDF syntax for {paper_id}. Cannot parse. Error: {e}", exc_info=False)
        return [] # Return empty list on critical parsing error
    except Exception as e:
        logger.error(f"Error during PDF parsing for {paper_id} using pdfminer.six: {e}", exc_info=True)
        return []
    # --- End PDF Parsing ---


    # --- Text Chunking ---
    logger.info(f"Starting text chunking for {paper_id}. Size={settings.PARSER_CHUNK_SIZE}, Overlap={settings.PARSER_CHUNK_OVERLAP}")
    if len(extracted_text) < settings.PARSER_CHUNK_SIZE:
         logger.info(f"Document {paper_id} is smaller than chunk size, creating a single chunk.")
         text_chunks = [extracted_text.strip()]
    else:
        try:
            # Use the recursive splitter
            text_chunks = await asyncio.to_thread(
                _recursive_split,
                extracted_text,
                settings.PARSER_SEPARATORS,
                settings.PARSER_CHUNK_SIZE,
                settings.PARSER_CHUNK_OVERLAP
            )
        except Exception as e:
            logger.error(f"Error during text chunking for {paper_id}: {e}", exc_info=True)
            # Fallback: create one large chunk if splitting fails? Or return empty?
            logger.warning(f"Chunking failed for {paper_id}, returning empty chunk list.")
            return []

    if not text_chunks:
         logger.warning(f"Text chunking resulted in zero chunks for {paper_id}.")
         return []

    # --- Create DocumentChunk objects ---
    final_chunks: List[DocumentChunk] = []
    for i, text_chunk in enumerate(text_chunks):
        if not text_chunk or not text_chunk.strip():
            continue # Skip empty chunks resulting from splitting

        chunk_id = generate_chunk_id(paper_id, i)
        # Base metadata copied from document retrieval
        chunk_meta = metadata.copy()
        # Add chunk-specific metadata
        chunk_meta.update({
            "chunk_index": i,
            "char_count": len(text_chunk),
            # TODO: Add page number mapping if pdfminer provides it (more complex parsing needed)
        })

        final_chunks.append(DocumentChunk(
            chunk_id=chunk_id,
            paper_id=paper_id,
            text=text_chunk,
            metadata=chunk_meta
        ))

    logger.info(f"Successfully created {len(final_chunks)} chunks for paper {paper_id}.")
    return final_chunks