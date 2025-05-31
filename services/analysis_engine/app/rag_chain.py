# services/analysis_engine/app/rag_chain.py

import logging
import httpx
from typing import List, Dict, Any, Optional, Sequence

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda

from langchain_google_genai import ChatGoogleGenerativeAI

from core.config import settings
from core.models import SearchQuery, SearchResponse, SearchResultItem # For typing

logger = logging.getLogger("IRN_Core").getChild("AnalysisEngine").getChild("RAGChain")

# --- Custom Retriever ---

class VectorServiceRetriever(BaseRetriever):
    """
    A LangChain Retriever that fetches context from our Vector Service microservice.
    """
    vector_service_url: str
    top_k: int = 5 # Default k for retrieval
    httpx_client: httpx.AsyncClient # Use a shared client

    # Define input schema if needed (using Pydantic V1 style for Langchain compatibility if issues arise)
    # class Config:
    #     arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Synchronous version (required by LangChain BaseRetriever).
        We will primarily use the async version.
        """
        # This sync version is less ideal in an async service.
        # Consider raising NotImplementedError or using httpx sync client if absolutely needed.
        logger.warning("Sync _get_relevant_documents called in VectorServiceRetriever - async preferred.")
        # For simplicity, let's raise error - force use of async path
        raise NotImplementedError("VectorServiceRetriever requires async retrieval via 'aget_relevant_documents'")


    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Asynchronous retrieval of documents from the vector service.
        'filters' should be a dict like {'paper_id': 'some_id'}
        """
        search_endpoint = f"{self.vector_service_url}/search"
        payload = SearchQuery(
            query_text=query,
            top_k=self.top_k,
            filters=filters # Pass filters directly
        ).model_dump()

        logger.info(f"Retriever querying Vector Service: query='{query[:30]}...', k={self.top_k}, filters={filters}")

        try:
            response = await self.httpx_client.post(search_endpoint, json=payload)
            response.raise_for_status()
            response_data = response.json()

            # Parse the response using Pydantic model
            search_response = SearchResponse(**response_data)

            # Convert SearchResultItem to LangChain Document objects
            documents = []
            for item in search_response.results:
                # Ensure metadata includes essential info like paper_id, chunk_id
                doc_metadata = item.metadata or {}
                doc_metadata['source_paper_id'] = item.paper_id # Ensure paper_id is in metadata
                doc_metadata['source_chunk_id'] = item.chunk_id # Ensure chunk_id is in metadata
                doc_metadata['similarity_score'] = item.score # Add score

                documents.append(Document(
                    page_content=item.text,
                    metadata=doc_metadata
                ))

            logger.info(f"Retriever received {len(documents)} documents from Vector Service.")
            return documents

        except httpx.HTTPStatusError as e:
            error_detail = f"Vector Service Error ({e.response.status_code})"
            try: downstream_error = e.response.json().get('detail', e.response.text)
            except Exception: downstream_error = e.response.text
            logger.error(f"{error_detail} during retrieval from {search_endpoint}: {downstream_error}")
            return [] # Return empty list on error
        except httpx.RequestError as e:
            logger.error(f"Could not connect to Vector Service at {search_endpoint} for retrieval: {e}")
            return []
        except Exception as e:
            logger.error(f"Error processing Vector Service response during retrieval: {e}", exc_info=True)
            return []


# --- Helper Functions ---

def format_docs(docs: Sequence[Document]) -> str:
    """Formats retrieved documents into a single string for the prompt."""
    formatted = []
    for i, doc in enumerate(docs):
        # Include key metadata
        source_id = doc.metadata.get('source_chunk_id', f"chunk_{i}")
        paper_id = doc.metadata.get('source_paper_id', 'unknown')
        score = doc.metadata.get('similarity_score', None)
        score_str = f" (Score: {score:.3f})" if score is not None else ""
        header = f"--- Document Chunk {i+1} (ID: {source_id}, Source Paper: {paper_id}){score_str} ---"
        formatted.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(formatted) if formatted else "No relevant documents found."

def extract_cited_sources(docs: Sequence[Document]) -> List[str]:
    """Extracts unique chunk IDs from retrieved documents."""
    return sorted(list(set(doc.metadata.get("source_chunk_id", "") for doc in docs if doc.metadata.get("source_chunk_id"))))

# --- RAG Chain Definition ---

def create_rag_chain(llm: ChatGoogleGenerativeAI, retriever: VectorServiceRetriever, analysis_type: str):
    """Creates a LangChain RAG chain based on the analysis type."""

    # --- Base Prompt Structure ---
    # Adapt these templates based on desired output for each analysis type
    if analysis_type == "summary":
        system_message = """You are an expert research assistant. Summarize the key findings and methodology from the following document chunks. Focus on the main points and conclusions presented in the text. Be concise and informative. If no relevant documents are found, state that clearly."""
        human_template = """Please summarize the provided context below:

Context:
{context}

Summary:"""

    elif analysis_type == "qa":
        system_message = """You are an expert Q&A assistant. Answer the user's question based *only* on the provided context document chunks. If the answer is not found in the context, state that clearly. Do not make up information. Cite the source chunk IDs if possible (e.g., [ref: paper_id_chunk_xxxx])."""
        human_template = """Context:
{context}

Question: {question}

Answer:"""

    elif analysis_type == "compare_methods":
         system_message = """You are an expert research analyst. Compare and contrast the methodologies described in the following document chunks from different papers. Highlight similarities, differences, strengths, and weaknesses mentioned in the text. If the context is insufficient for comparison, state that."""
         human_template = """Please compare the methodologies described in the context below:

Context:
{context}

Comparison:"""

    elif analysis_type == "gap_analysis":
         system_message = """You are an expert research strategist. Based on the provided document chunks, identify potential research gaps, limitations mentioned, or unanswered questions suggested by the authors. If the context doesn't mention gaps or limitations, state that."""
         human_template = """Analyze the context below to identify potential research gaps or limitations mentioned:

Context:
{context}

Identified Gaps/Limitations:"""
    else:
        # Default fallback or raise error
        logger.warning(f"Unknown analysis type '{analysis_type}'. Using default summarization prompt.")
        system_message = "Summarize the following context:"
        human_template = """Context:
{context}

Summary:"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_template),
    ])

    # --- Chain Construction (LCEL) ---

    # Define how context is fetched and formatted
    rag_retrieval_chain = (
        # Pass input query/topic to retriever, get Documents
        retriever
        # Format Documents into a single string context
        | RunnableLambda(format_docs).with_config(run_name="FormatDocs")
    )

    # Define the final chain combining context and question (if applicable)
    if analysis_type == "qa":
        # QA requires both context (from query) and the original question
        chain = (
            RunnableParallel(
                # Retrieve context based on the question
                context=rag_retrieval_chain,
                 # Pass the original question through
                question=RunnablePassthrough()
            ).with_config(run_name="PrepareQAInput")
            # Feed context and question to the prompt
            | prompt
            # Send prompt to LLM
            | llm
            # Parse the LLM output string
            | StrOutputParser()
        ).with_config(run_name="QA_Chain")
    else:
        # Other types primarily need context based on the analysis topic/query
        chain = (
            # Retrieve and format context
            rag_retrieval_chain
            # Feed context into the prompt (implicitly handles {context})
            | prompt
            | llm
            | StrOutputParser()
        ).with_config(run_name=f"{analysis_type.upper()}_Chain")

    # --- Chain that also returns sources ---
    # We need to pass the retrieved documents through to the end

    retrieval_and_docs_chain = RunnableParallel(
        # Get formatted context string
        context=rag_retrieval_chain,
        # Also pass through the raw documents retrieved
        retrieved_docs=retriever
    ).with_config(run_name="RetrieveContextAndDocs")


    if analysis_type == "qa":
         # Chain for QA that returns answer and sources
         chain_with_sources = RunnableParallel(
             # Get the answer using the main chain
             answer = (
                 RunnablePassthrough.assign( # Assign context and question
                    context = lambda x: format_docs(x["retrieved_docs"]), # Format docs passed in
                    question = lambda x: x["question"] # Pass question through
                 )
                 | prompt
                 | llm
                 | StrOutputParser()
             ),
             # Get the source chunk IDs
             cited_sources = lambda x: extract_cited_sources(x["retrieved_docs"])
         ).with_config(run_name="QA_Chain_With_Sources")

         # Combine retrieval and the final step
         full_chain = (
             RunnableParallel(
                retrieved_docs = retriever,
                question = RunnablePassthrough() # Pass original question
             )
             | chain_with_sources
         )

    else: # For summary, compare, gap_analysis
         # Chain for other types that returns result and sources
         chain_with_sources = RunnableParallel(
             # Get the analysis result
             result_text = (
                 RunnablePassthrough.assign(context = lambda x: format_docs(x["retrieved_docs"]))
                 | prompt
                 | llm
                 | StrOutputParser()
             ),
              # Get the source chunk IDs
             cited_sources = lambda x: extract_cited_sources(x["retrieved_docs"])
         ).with_config(run_name=f"{analysis_type.upper()}_Chain_With_Sources")

         # Combine retrieval and the final step
         full_chain = (
            retriever.with_config(run_name="RetrieveDocs") # Retrieve docs based on input query/topic
            | RunnableLambda(lambda docs: {"retrieved_docs": docs}).with_config(run_name="PackageDocs") # Package docs for next step
            | chain_with_sources
         )

    return full_chain