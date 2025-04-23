# core/gemini_client.py
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold
from core.config import settings, logger
from core.models import DocumentChunk
from typing import List, Dict, Any, Tuple, Optional
import time
import asyncio

# Configuration for generation (can be customized per call)
DEFAULT_GENERATION_CONFIG = GenerationConfig(
    # temperature=0.7,
    # top_p=1.0,
    # top_k=40,
    # max_output_tokens=8192, # Adjust based on model limits and needs
)

# Safety settings - adjust as needed, be cautious with NONE
DEFAULT_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}


class GeminiClient:
    def __init__(self, api_key: str | None = settings.GEMINI_API_KEY):
        self.configured = False
        self.pro_model = None
        self.flash_model = None
        self.embedding_model_name = settings.GEMINI_EMBEDDING_MODEL

        if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
            logger.warning("GEMINI_API_KEY not configured. GeminiClient will not function.")
            return

        try:
            genai.configure(api_key=api_key)
            # Initialize models here
            self.pro_model = genai.GenerativeModel(
                settings.GEMINI_PRO_MODEL,
                # generation_config=DEFAULT_GENERATION_CONFIG, # Can override later
                # safety_settings=DEFAULT_SAFETY_SETTINGS # Can override later
            )
            self.flash_model = genai.GenerativeModel(
                settings.GEMINI_FLASH_MODEL,
                # generation_config=DEFAULT_GENERATION_CONFIG,
                # safety_settings=DEFAULT_SAFETY_SETTINGS
            )
            self.configured = True
            logger.info("Gemini Client configured successfully.")
        except Exception as e:
            logger.error(f"Failed to configure Gemini Client: {e}", exc_info=True)
            self.configured = False


    def _get_model(self, model_type: str = "pro"):
        if not self.configured:
            raise RuntimeError("Gemini client not configured.")
        if model_type == "flash" and self.flash_model:
            return self.flash_model
        elif model_type == "pro" and self.pro_model:
            return self.pro_model
        else:
            logger.error(f"Requested Gemini model type '{model_type}' is not available.")
            raise ValueError(f"Model type '{model_type}' unavailable.")

    async def _generate_with_retry(self, model, prompt: str, generation_config: Optional[GenerationConfig] = None, safety_settings: Optional[dict] = None, retries=3, delay=5, **kwargs) -> Tuple[str | None, Dict[str, Any]]:
        """Internal helper for robust generation with retry."""
        last_exception = None
        usage_metadata = {}

        if not self.configured:
             logger.error("Attempted generation with unconfigured Gemini client.")
             return None, {"error": "Client not configured"}

        gen_config = generation_config or DEFAULT_GENERATION_CONFIG
        safety = safety_settings or DEFAULT_SAFETY_SETTINGS

        for attempt in range(retries):
            try:
                # Use generate_content_async for async operation
                response = await model.generate_content_async(
                    prompt,
                    generation_config=gen_config,
                    safety_settings=safety,
                    **kwargs
                )

                # Check for blocked content
                if not response.candidates:
                     block_reason = response.prompt_feedback.block_reason.name if response.prompt_feedback else "Unknown"
                     logger.warning(f"Gemini response blocked. Reason: {block_reason}")
                     return None, {"error": f"Content blocked by safety filter: {block_reason}", "prompt_feedback": response.prompt_feedback}

                # Log usage metadata if available
                if hasattr(response, 'usage_metadata'):
                    usage_metadata = response.usage_metadata
                    # logger.info(f"Gemini Usage: Input={usage_metadata.prompt_token_count}, Output={usage_metadata.candidates_token_count}, Total={usage_metadata.total_token_count}")
                    # Add cost estimation here if desired (requires token prices)

                # TODO: More sophisticated handling of multiple candidates if needed
                return response.text, usage_metadata

            except Exception as e:
                logger.warning(f"Gemini API call failed (attempt {attempt + 1}/{retries}): {e}")
                last_exception = e
                await asyncio.sleep(delay * (attempt + 1)) # Use asyncio.sleep

        logger.error(f"Gemini API call failed after {retries} retries: {last_exception}")
        return None, {"error": str(last_exception)}

    async def generate_text(self, prompt: str, model_type: str = "pro", **kwargs) -> Tuple[str | None, Dict[str, Any]]:
        """Generate text using the specified model type (pro or flash)."""
        model = self._get_model(model_type)
        logger.info(f"Generating text with Gemini {model_type.upper()} model...")
        # Measure time?
        start_time = time.monotonic()
        result, usage = await self._generate_with_retry(model, prompt, **kwargs)
        duration = time.monotonic() - start_time
        logger.info(f"Gemini {model_type.upper()} generation took {duration:.2f}s. Output length: {len(result) if result else 0}")
        return result, usage

    async def generate_embeddings(self, chunks: List[DocumentChunk]) -> Tuple[List[DocumentChunk] | None, Dict[str, Any]]:
        """Generate embeddings for a list of text chunks."""
        if not self.configured:
             logger.error("Attempted embedding generation with unconfigured Gemini client.")
             return None, {"error": "Client not configured"}
        if not chunks:
             return [], {"message": "No chunks provided"}

        logger.info(f"Generating embeddings for {len(chunks)} chunks using {self.embedding_model_name}...")
        start_time = time.monotonic()
        try:
            # Prepare texts, handling potential empty strings robustly
            texts_to_embed: List[str] = []
            original_indices: List[int] = []
            for i, chunk in enumerate(chunks):
                text = chunk.text.strip()
                if text: # Only embed non-empty text
                    texts_to_embed.append(text)
                    original_indices.append(i)
                else:
                     logger.warning(f"Chunk {chunk.chunk_id} has empty text, skipping embedding.")

            if not texts_to_embed:
                logger.warning("No valid text found in chunks for embedding.")
                # Update chunks to have zero embeddings or None? Let's use None for now.
                for chunk in chunks:
                    chunk.embedding = None
                return chunks, {"message": "No valid text to embed"}

            # Use embed_content_async
            # Note: Check batching limits for the specific model if processing very large lists
            result = await genai.embed_content_async(
                model=self.embedding_model_name,
                content=texts_to_embed,
                task_type="retrieval_document" # Appropriate task type for RAG docs
                # task_type="semantic_similarity" / "classification" / "clustering" available too
            )
            embeddings = result.get('embedding') # Correct key is 'embedding'

            if not embeddings or len(embeddings) != len(texts_to_embed):
                 logger.error(f"Mismatch between number of texts sent ({len(texts_to_embed)}) and embeddings received ({len(embeddings) if embeddings else 0})")
                 return None, {"error": "Embedding count mismatch or API error"}

            # Assign embeddings back to the original chunks
            for i, embedding in enumerate(embeddings):
                 original_chunk_index = original_indices[i]
                 chunks[original_chunk_index].embedding = embedding

            # Assign None or zero vector to skipped chunks
            for i, chunk in enumerate(chunks):
                if i not in original_indices:
                    # Decide: None or zero vector? Using None for clarity.
                    chunk.embedding = None
                    # Or: chunk.embedding = [0.0] * settings.EMBEDDING_DIM

            duration = time.monotonic() - start_time
            logger.info(f"Successfully generated {len(embeddings)} embeddings in {duration:.2f}s.")
            # TODO: Add actual token count / cost estimation if API provides it for embeddings
            usage_metadata = {"estimated_input_chars": sum(len(t) for t in texts_to_embed)}

            return chunks, usage_metadata

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}", exc_info=True)
            return None, {"error": str(e)}

    async def generate_query_embedding(self, query_text: str) -> Tuple[Optional[List[float]], Dict[str, Any]]:
        """Generates an embedding for a single query string."""
        if not self.configured:
             logger.error("Attempted query embedding generation with unconfigured Gemini client.")
             return None, {"error": "Client not configured"}

        logger.info(f"Generating query embedding using {self.embedding_model_name}...")
        start_time = time.monotonic()
        try:
            query_text = query_text.strip()
            if not query_text:
                logger.warning("Attempted to embed empty query string.")
                return None, {"error": "Empty query string"}

            result = await genai.embed_content_async(
                model=self.embedding_model_name,
                content=query_text,
                task_type="retrieval_query" # Use the query-specific task type
            )
            embedding = result.get('embedding')

            if not embedding:
                logger.error("Failed to generate query embedding, API returned no result.")
                return None, {"error": "API returned no embedding"}

            duration = time.monotonic() - start_time
            logger.info(f"Successfully generated query embedding in {duration:.2f}s.")
            usage_metadata = {"estimated_input_chars": len(query_text)}
            return embedding, usage_metadata

        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}", exc_info=True)
            return None, {"error": str(e)}


# Instantiate the client for easy import across services
gemini_client = GeminiClient()