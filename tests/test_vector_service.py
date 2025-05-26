import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock

from services.vector_service.app.main import app
from core.models import (
    EmbedRequest, EmbedResponse, SearchQuery, SearchResponse,
    DocumentChunk, SearchResultItem
)
from core.gemini_client import GeminiClient 
from services.vector_service.app.main import get_supabase_client # To mock for health check & ops

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

# --- Test Health Endpoint ---
@patch("services.vector_service.app.main.get_supabase_client", new_callable=AsyncMock)
@patch("services.vector_service.app.main.gemini_client", spec=GeminiClient)
def test_health_check_ok(mock_gemini_client_instance, mock_get_db, client: TestClient):
    mock_gemini_client_instance.configured = True
    mock_db_client = AsyncMock() 
    mock_get_db.return_value = mock_db_client 

    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "ok"
    assert json_data["dependencies"]["database"] == "connected"
    assert json_data["dependencies"]["gemini"] == "configured"
    mock_get_db.assert_called_once()


@patch("services.vector_service.app.main.get_supabase_client", new_callable=AsyncMock)
@patch("services.vector_service.app.main.gemini_client", spec=GeminiClient)
def test_health_check_deps_error(mock_gemini_client_instance, mock_get_db, client: TestClient):
    mock_gemini_client_instance.configured = False
    mock_get_db.side_effect = Exception("DB connection error") 

    response = client.get("/health")
    assert response.status_code == 200 
    json_data = response.json()
    assert json_data["status"] == "ok" 
    assert json_data["dependencies"]["database"] == "error"
    assert json_data["dependencies"]["gemini"] == "not_configured"
    mock_get_db.assert_called_once()


# --- Tests for /embed endpoint ---
@patch("services.vector_service.app.main.gemini_client.generate_embeddings", new_callable=AsyncMock)
@patch("services.vector_service.app.vector_ops.upsert_embeddings", new_callable=AsyncMock)
@patch("services.vector_service.app.main.get_supabase_client", new_callable=AsyncMock)
def test_embed_chunks_success(mock_get_db, mock_upsert, mock_generate_embeds, client: TestClient):
    mock_db_client = AsyncMock()
    mock_get_db.return_value = mock_db_client

    mock_input_chunks_data = [
        {"chunk_id": "c1", "paper_id": "p1", "text_content": "text1", "metadata":{}},
        {"chunk_id": "c2", "paper_id": "p1", "text_content": "text2", "metadata":{}}
    ]
    expected_doc_chunks_for_gemini = [
        DocumentChunk(chunk_id="c1", paper_id="p1", text_content="text1", metadata={}),
        DocumentChunk(chunk_id="c2", paper_id="p1", text_content="text2", metadata={})
    ]
    mock_embedded_chunks_from_gemini = [
        DocumentChunk(chunk_id="c1", paper_id="p1", text_content="text1", metadata={}, embedding=[0.1]),
        DocumentChunk(chunk_id="c2", paper_id="p1", text_content="text2", metadata={}, embedding=[0.2])
    ]
    
    mock_generate_embeds.return_value = (mock_embedded_chunks_from_gemini, {"total_tokens": 10})
    mock_upsert.return_value = (["c1", "c2"], []) 

    payload = EmbedRequest(chunks=mock_input_chunks_data)
    response = client.post("/embed", json=payload.model_dump())

    assert response.status_code == 200
    json_res = response.json()
    assert sorted(json_res["processed_chunk_ids"]) == sorted(["c1", "c2"])
    assert json_res["failed_chunk_ids"] == []
    
    mock_generate_embeds.assert_called_once()
    call_args_list = mock_generate_embeds.call_args[0][0]
    assert len(call_args_list) == len(expected_doc_chunks_for_gemini)
    for called_arg, expected_arg in zip(call_args_list, expected_doc_chunks_for_gemini):
        assert called_arg.chunk_id == expected_arg.chunk_id
        assert called_arg.text_content == expected_arg.text_content

    mock_upsert.assert_called_once_with(db_client=mock_db_client, chunks_with_embeddings=mock_embedded_chunks_from_gemini)

def test_embed_chunks_empty_list(client: TestClient):
    payload = EmbedRequest(chunks=[])
    response = client.post("/embed", json=payload.model_dump())
    assert response.status_code == 200
    json_res = response.json()
    assert json_res["processed_chunk_ids"] == []
    assert json_res["failed_chunk_ids"] == []

def test_embed_chunks_invalid_data(client: TestClient):
    # Missing 'text_content' which is required by DocumentChunk
    payload = {"chunks": [{"chunk_id": "c1", "paper_id": "p1"}]} 
    response = client.post("/embed", json=payload)
    assert response.status_code == 422 # FastAPI Pydantic validation error

@patch("services.vector_service.app.main.gemini_client.generate_embeddings", new_callable=AsyncMock)
@patch("services.vector_service.app.main.get_supabase_client", new_callable=AsyncMock)
def test_embed_chunks_gemini_fails_returns_none(mock_get_db, mock_generate_embeds, client: TestClient):
    mock_input_chunks_data = [{"chunk_id": "c1", "paper_id": "p1", "text_content": "text1", "metadata": {}}]
    mock_generate_embeds.return_value = (None, {"error": "Gemini client error", "total_tokens": 0})

    payload = EmbedRequest(chunks=mock_input_chunks_data)
    response = client.post("/embed", json=payload.model_dump())

    assert response.status_code == 500
    assert "Embedding generation failed: Gemini client error" in response.json()["detail"]

@patch("services.vector_service.app.main.gemini_client.generate_embeddings", new_callable=AsyncMock)
@patch("services.vector_service.app.main.get_supabase_client", new_callable=AsyncMock)
def test_embed_chunks_gemini_raises_exception(mock_get_db, mock_generate_embeds, client: TestClient):
    mock_input_chunks_data = [{"chunk_id": "c1", "paper_id": "p1", "text_content": "text1", "metadata": {}}]
    mock_generate_embeds.side_effect = Exception("Gemini exploded")

    payload = EmbedRequest(chunks=mock_input_chunks_data)
    response = client.post("/embed", json=payload.model_dump())

    assert response.status_code == 500
    assert "Embedding generation failed: Gemini exploded" in response.json()["detail"]

@patch("services.vector_service.app.main.gemini_client.generate_embeddings", new_callable=AsyncMock)
@patch("services.vector_service.app.vector_ops.upsert_embeddings", new_callable=AsyncMock)
@patch("services.vector_service.app.main.get_supabase_client", new_callable=AsyncMock)
def test_embed_chunks_upsert_fails(mock_get_db, mock_upsert, mock_generate_embeds, client: TestClient):
    mock_db_client = AsyncMock()
    mock_get_db.return_value = mock_db_client
    mock_input_chunks_data = [{"chunk_id": "c1", "paper_id": "p1", "text_content": "text1", "metadata": {}}]
    mock_embedded_chunks = [DocumentChunk(chunk_id="c1", paper_id="p1", text_content="text1", metadata={}, embedding=[0.1])]
    
    mock_generate_embeds.return_value = (mock_embedded_chunks, {"total_tokens": 5})
    mock_upsert.side_effect = Exception("DB upsert failed")

    payload = EmbedRequest(chunks=mock_input_chunks_data)
    response = client.post("/embed", json=payload.model_dump())

    assert response.status_code == 500
    assert "Upserting embeddings failed: DB upsert failed" in response.json()["detail"]

@patch("services.vector_service.app.main.gemini_client.generate_embeddings", new_callable=AsyncMock)
@patch("services.vector_service.app.vector_ops.upsert_embeddings", new_callable=AsyncMock)
@patch("services.vector_service.app.main.get_supabase_client", new_callable=AsyncMock)
def test_embed_chunks_some_skipped_by_gemini(mock_get_db, mock_upsert, mock_generate_embeds, client: TestClient):
    mock_db_client = AsyncMock()
    mock_get_db.return_value = mock_db_client
    mock_input_chunks_data = [
        {"chunk_id": "c1", "paper_id": "p1", "text_content": "text1", "metadata":{}},
        {"chunk_id": "c2_skip", "paper_id": "p1", "text_content": "text_skip", "metadata":{}}, # This one will be skipped
        {"chunk_id": "c3", "paper_id": "p1", "text_content": "text3", "metadata":{}}
    ]
    # Gemini returns only c1 and c3 with embeddings, c2_skip has embedding=None
    mock_embedded_chunks_from_gemini = [
        DocumentChunk(chunk_id="c1", paper_id="p1", text_content="text1", metadata={}, embedding=[0.1]),
        DocumentChunk(chunk_id="c2_skip", paper_id="p1", text_content="text_skip", metadata={}, embedding=None),
        DocumentChunk(chunk_id="c3", paper_id="p1", text_content="text3", metadata={}, embedding=[0.3])
    ]
    # Upsert should only be called with chunks that have embeddings
    expected_chunks_for_upsert = [mock_embedded_chunks_from_gemini[0], mock_embedded_chunks_from_gemini[2]]

    mock_generate_embeds.return_value = (mock_embedded_chunks_from_gemini, {"total_tokens": 10})
    mock_upsert.return_value = (["c1", "c3"], []) # Upsert reports success for c1, c3

    payload = EmbedRequest(chunks=mock_input_chunks_data)
    response = client.post("/embed", json=payload.model_dump())

    assert response.status_code == 200
    json_res = response.json()
    assert sorted(json_res["processed_chunk_ids"]) == sorted(["c1", "c3"])
    assert json_res["failed_chunk_ids"] == ["c2_skip"]
    
    mock_upsert.assert_called_once_with(db_client=mock_db_client, chunks_with_embeddings=expected_chunks_for_upsert)


# --- Tests for /search endpoint ---
@patch("services.vector_service.app.main.gemini_client.generate_query_embedding", new_callable=AsyncMock)
@patch("services.vector_service.app.vector_ops.search_similar", new_callable=AsyncMock)
@patch("services.vector_service.app.main.get_supabase_client", new_callable=AsyncMock)
def test_search_vectors_success(mock_get_db, mock_search_similar, mock_generate_query_embed, client: TestClient):
    mock_db_client = AsyncMock()
    mock_get_db.return_value = mock_db_client

    mock_query_embedding = [0.1, 0.2, 0.3]
    mock_search_results_data = [
        SearchResultItem(paper_id="p1", chunk_id="c1", text_content="found text 1", similarity_score=0.91, metadata={}),
        SearchResultItem(paper_id="p2", chunk_id="c2", text_content="found text 2", similarity_score=0.85, metadata={})
    ]

    mock_generate_query_embed.return_value = (mock_query_embedding, {"total_tokens": 5})
    mock_search_similar.return_value = mock_search_results_data

    payload = SearchQuery(query_text="test search query", top_k=3, filters={"paper_id": "p1"})
    response = client.post("/search", json=payload.model_dump())

    assert response.status_code == 200
    json_res = response.json()
    
    expected_response_data = SearchResponse(results=mock_search_results_data).model_dump()
    assert json_res == expected_response_data
    
    mock_generate_query_embed.assert_called_once_with(query_text="test search query")
    mock_search_similar.assert_called_once_with(
        db_client=mock_db_client,
        query_embedding=mock_query_embedding,
        top_k=3,
        filters={"paper_id": "p1"}
    )

@patch("services.vector_service.app.main.gemini_client.generate_query_embedding", new_callable=AsyncMock)
@patch("services.vector_service.app.main.get_supabase_client", new_callable=AsyncMock)
def test_search_vectors_gemini_query_embed_fails_returns_none(mock_get_db, mock_generate_query_embed, client: TestClient):
    mock_generate_query_embed.return_value = (None, {"error": "Gemini query embedding error", "total_tokens": 0})

    payload = SearchQuery(query_text="another test query", top_k=2)
    response = client.post("/search", json=payload.model_dump())

    assert response.status_code == 500
    assert "Failed to generate query embedding: Gemini query embedding error" in response.json()["detail"]

@patch("services.vector_service.app.main.gemini_client.generate_query_embedding", new_callable=AsyncMock)
@patch("services.vector_service.app.main.get_supabase_client", new_callable=AsyncMock)
def test_search_vectors_gemini_query_embed_raises_exception(mock_get_db, mock_generate_query_embed, client: TestClient):
    mock_generate_query_embed.side_effect = Exception("Gemini query exploded")

    payload = SearchQuery(query_text="test query exception", top_k=2)
    response = client.post("/search", json=payload.model_dump())

    assert response.status_code == 500
    assert "Failed to generate query embedding: Gemini query exploded" in response.json()["detail"]

@patch("services.vector_service.app.main.gemini_client.generate_query_embedding", new_callable=AsyncMock)
@patch("services.vector_service.app.vector_ops.search_similar", new_callable=AsyncMock)
@patch("services.vector_service.app.main.get_supabase_client", new_callable=AsyncMock)
def test_search_vectors_search_similar_fails(mock_get_db, mock_search_similar, mock_generate_query_embed, client: TestClient):
    mock_db_client = AsyncMock()
    mock_get_db.return_value = mock_db_client
    mock_query_embedding = [0.1, 0.2]
    
    mock_generate_query_embed.return_value = (mock_query_embedding, {"total_tokens": 5})
    mock_search_similar.side_effect = Exception("DB search failed")

    payload = SearchQuery(query_text="test query", top_k=2)
    response = client.post("/search", json=payload.model_dump())

    assert response.status_code == 500
    assert "Vector search failed: DB search failed" in response.json()["detail"]

@patch("services.vector_service.app.main.gemini_client.generate_query_embedding", new_callable=AsyncMock)
@patch("services.vector_service.app.vector_ops.search_similar", new_callable=AsyncMock)
@patch("services.vector_service.app.main.get_supabase_client", new_callable=AsyncMock)
def test_search_vectors_search_similar_value_error(mock_get_db, mock_search_similar, mock_generate_query_embed, client: TestClient):
    mock_db_client = AsyncMock()
    mock_get_db.return_value = mock_db_client
    mock_query_embedding = [0.1, 0.2]

    mock_generate_query_embed.return_value = (mock_query_embedding, {"total_tokens": 5})
    mock_search_similar.side_effect = ValueError("RPC function 'nonexistent_function' not found")

    payload = SearchQuery(query_text="test query", top_k=2)
    response = client.post("/search", json=payload.model_dump())

    assert response.status_code == 400 # As per endpoint logic for ValueError
    assert "Invalid search parameters or configuration: RPC function 'nonexistent_function' not found" in response.json()["detail"]


# --- Tests for vector_ops.py ---
from services.vector_service.app import vector_ops
from core.config import settings as core_settings # For SEARCH_MATCH_THRESHOLD, DB_RPC_FUNCTION_NAME
from postgrest import APIError as PostgrestAPIError

# Mock Supabase client for vector_ops tests
@pytest_asyncio.fixture
async def mock_db_client_vector_ops():
    mock_client = AsyncMock()
    
    # Fluent API mocks for table().upsert().execute()
    mock_table_response = AsyncMock()
    mock_upsert_response = AsyncMock()
    mock_client.table.return_value = mock_table_response
    mock_table_response.upsert.return_value = mock_upsert_response
    mock_upsert_response.execute = AsyncMock() # To be configured per test

    # Fluent API mocks for rpc().execute()
    mock_rpc_response = AsyncMock()
    mock_client.rpc.return_value = mock_rpc_response
    mock_rpc_response.execute = AsyncMock() # To be configured per test
    
    return mock_client

# --- Tests for vector_ops.upsert_embeddings ---
@pytest.mark.asyncio
async def test_upsert_embeddings_success(mock_db_client_vector_ops: AsyncMock):
    chunks_to_upsert = [
        DocumentChunk(chunk_id="c1", paper_id="p1", text_content="text1", embedding=[0.1, 0.2], metadata={"page": 1}),
        DocumentChunk(chunk_id="c2", paper_id="p1", text_content="text2", embedding=[0.3, 0.4], metadata={"page": 2})
    ]
    # Supabase client returns an APIResponse-like object with a 'data' attribute
    # For upsert, the content of 'data' isn't strictly checked by the current vector_ops,
    # but it should not be empty to indicate success.
    mock_api_response = AsyncMock()
    mock_api_response.data = [{"id": "c1"}, {"id": "c2"}] # Dummy data indicating success
    mock_db_client_vector_ops.table.return_value.upsert.return_value.execute.return_value = mock_api_response

    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        processed_ids, failed_ids = await vector_ops.upsert_embeddings(mock_db_client_vector_ops, chunks_to_upsert)

    assert sorted(processed_ids) == sorted(["c1", "c2"])
    assert failed_ids == []
    
    mock_db_client_vector_ops.table.assert_called_once_with(vector_ops.DB_TABLE_NAME)
    
    expected_upsert_data = [
        {
            "id": "c1", "paper_id": "p1", "content": "text1", 
            "embedding": "[0.1,0.2]", "metadata": {"page": 1}
        },
        {
            "id": "c2", "paper_id": "p1", "content": "text2", 
            "embedding": "[0.3,0.4]", "metadata": {"page": 2}
        }
    ]
    # Ensure the data passed to upsert matches the expected structure
    actual_upsert_call_args = mock_db_client_vector_ops.table.return_value.upsert.call_args[0][0]
    assert actual_upsert_call_args == expected_upsert_data

@pytest.mark.asyncio
async def test_upsert_embeddings_no_chunks(mock_db_client_vector_ops: AsyncMock):
    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        processed_ids, failed_ids = await vector_ops.upsert_embeddings(mock_db_client_vector_ops, [])
    assert processed_ids == []
    assert failed_ids == []
    mock_db_client_vector_ops.table.assert_not_called()

@pytest.mark.asyncio
@patch("services.vector_service.app.vector_ops.logger")
async def test_upsert_embeddings_chunks_with_none_embedding(mock_logger, mock_db_client_vector_ops: AsyncMock):
    chunks_with_none_embedding = [
        DocumentChunk(chunk_id="c1", paper_id="p1", text_content="text1", embedding=None),
        DocumentChunk(chunk_id="c2", paper_id="p1", text_content="text2", embedding=[0.1, 0.2]) # One valid
    ]
    
    mock_api_response = AsyncMock()
    mock_api_response.data = [{"id": "c2"}] # Only c2 would be upserted
    mock_db_client_vector_ops.table.return_value.upsert.return_value.execute.return_value = mock_api_response

    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        processed_ids, failed_ids = await vector_ops.upsert_embeddings(mock_db_client_vector_ops, chunks_with_none_embedding)

    assert processed_ids == ["c2"]
    assert failed_ids == ["c1"] # c1 should be in failed_ids
    mock_logger.warning.assert_called_once_with("Chunk c1 has no embedding, skipping.")
    
    expected_upsert_data_c2 = [{"id": "c2", "paper_id": "p1", "content": "text2", "embedding": "[0.1,0.2]", "metadata": None}]
    mock_db_client_vector_ops.table.return_value.upsert.assert_called_once_with(expected_upsert_data_c2)


@pytest.mark.asyncio
async def test_upsert_embeddings_supabase_api_error(mock_db_client_vector_ops: AsyncMock):
    chunks_to_upsert = [DocumentChunk(chunk_id="c1", paper_id="p1", text_content="text1", embedding=[0.1])]
    mock_db_client_vector_ops.table.return_value.upsert.return_value.execute.side_effect = PostgrestAPIError(
        {"message": "DB error"}, response=None, request=None
    )
    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        with pytest.raises(RuntimeError, match="Error during batch upsert to Supabase: DB error"):
            await vector_ops.upsert_embeddings(mock_db_client_vector_ops, chunks_to_upsert)

@pytest.mark.asyncio
async def test_upsert_embeddings_other_exception(mock_db_client_vector_ops: AsyncMock):
    chunks_to_upsert = [DocumentChunk(chunk_id="c1", paper_id="p1", text_content="text1", embedding=[0.1])]
    mock_db_client_vector_ops.table.return_value.upsert.return_value.execute.side_effect = Exception("Unexpected error")
    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        with pytest.raises(RuntimeError, match="Unexpected error during upsert: Unexpected error"):
            await vector_ops.upsert_embeddings(mock_db_client_vector_ops, chunks_to_upsert)


# --- Tests for vector_ops.search_similar ---
@pytest.mark.asyncio
async def test_search_similar_success(mock_db_client_vector_ops: AsyncMock):
    query_embedding = [0.1, 0.2, 0.3]
    top_k = 5
    filters = {"paper_id": "p1"}
    
    db_results_data = [
        {"id": "chunk1", "paper_id": "p1", "similarity": 0.9, "content": "text content 1", "metadata": {"page": 1}},
        {"id": "chunk2", "paper_id": "p1", "similarity": 0.8, "content": "text content 2", "metadata": {"page": 2}}
    ]
    mock_api_response = AsyncMock()
    mock_api_response.data = db_results_data
    mock_db_client_vector_ops.rpc.return_value.execute.return_value = mock_api_response

    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        results = await vector_ops.search_similar(mock_db_client_vector_ops, query_embedding, top_k, filters)

    assert len(results) == 2
    assert isinstance(results[0], SearchResultItem)
    assert results[0].chunk_id == "chunk1"
    assert results[0].similarity_score == 0.9
    assert results[1].chunk_id == "chunk2"
    assert results[1].similarity_score == 0.8
    
    expected_rpc_params = {
        'query_embedding': query_embedding,
        'match_threshold': core_settings.SEARCH_MATCH_THRESHOLD,
        'match_count': top_k,
        'filter_paper_id': "p1" # Filter applied
    }
    mock_db_client_vector_ops.rpc.assert_called_once_with(core_settings.DB_RPC_FUNCTION_NAME, expected_rpc_params)

@pytest.mark.asyncio
async def test_search_similar_no_results(mock_db_client_vector_ops: AsyncMock):
    mock_api_response = AsyncMock()
    mock_api_response.data = [] # Or None, vector_ops handles both
    mock_db_client_vector_ops.rpc.return_value.execute.return_value = mock_api_response

    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        results = await vector_ops.search_similar(mock_db_client_vector_ops, [0.1], 3)
    assert results == []

@pytest.mark.asyncio
async def test_search_similar_no_query_embedding(mock_db_client_vector_ops: AsyncMock):
    with pytest.raises(ValueError, match="Query embedding cannot be None"):
        await vector_ops.search_similar(mock_db_client_vector_ops, None, 3)

@pytest.mark.asyncio
async def test_search_similar_rpc_function_not_found(mock_db_client_vector_ops: AsyncMock):
    # Specific PostgrestError for "relation ... does not exist" which vector_ops catches as ValueError
    error_response = MagicMock() # Mock the response object within PostgrestAPIError
    error_response.json = MagicMock(return_value={"message": "relation public.nonexistent_function does not exist"})
    
    mock_db_client_vector_ops.rpc.return_value.execute.side_effect = PostgrestAPIError(
        {"message": "relation public.nonexistent_function does not exist"},
        response=error_response, request=None
    )
    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        with pytest.raises(ValueError, match="RPC function .* not found."):
            await vector_ops.search_similar(mock_db_client_vector_ops, [0.1], 3)

@pytest.mark.asyncio
async def test_search_similar_other_supabase_api_error(mock_db_client_vector_ops: AsyncMock):
    mock_db_client_vector_ops.rpc.return_value.execute.side_effect = PostgrestAPIError(
        {"message": "Some other DB error"}, response=None, request=None
    )
    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        with pytest.raises(RuntimeError, match="Error during Supabase RPC call: Some other DB error"):
            await vector_ops.search_similar(mock_db_client_vector_ops, [0.1], 3)

@pytest.mark.asyncio
async def test_search_similar_other_exception(mock_db_client_vector_ops: AsyncMock):
    mock_db_client_vector_ops.rpc.return_value.execute.side_effect = Exception("Unexpected RPC error")
    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        with pytest.raises(RuntimeError, match="Unexpected error during search: Unexpected RPC error"):
            await vector_ops.search_similar(mock_db_client_vector_ops, [0.1], 3)

@pytest.mark.asyncio
@patch("services.vector_service.app.vector_ops.logger")
async def test_search_similar_error_parsing_result_item(mock_logger, mock_db_client_vector_ops: AsyncMock):
    db_results_data = [
        {"id": "chunk1", "paper_id": "p1", "similarity": 0.9, "content": "text1"}, # Missing metadata
        {"id": "chunk2", "paper_id": "p2", "similarity": 0.8, "content": "text2", "metadata": {"page": 1}} # Valid
    ]
    mock_api_response = AsyncMock()
    mock_api_response.data = db_results_data
    mock_db_client_vector_ops.rpc.return_value.execute.return_value = mock_api_response

    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        results = await vector_ops.search_similar(mock_db_client_vector_ops, [0.1], 2)

    assert len(results) == 1
    assert results[0].chunk_id == "chunk2"
    mock_logger.warning.assert_called_once()
    # Check that the log message contains information about the parsing error and the problematic item
    args, _ = mock_logger.warning.call_args
    assert "Error parsing search result item" in args[0]
    assert "'id': 'chunk1'" in args[1] # Check if the problematic item's data is in the log
```
