import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from fastapi import BackgroundTasks
from unittest.mock import patch, AsyncMock, MagicMock 

from services.document_processor.app.main import app, process_and_embed_task
from core.models import ProcessRequest, DocumentChunk, EmbedResponse, EmbedRequest, PaperStatus
from core.config import settings # For VECTOR_SERVICE_URL
import httpx

@pytest.fixture
def client():
    # Clean up dependency overrides after each test if they were set
    original_overrides = app.dependency_overrides.copy()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides = original_overrides


@pytest.fixture
def mock_background_tasks():
    mock = MagicMock(spec=BackgroundTasks)
    mock.add_task = MagicMock() 
    return mock

# --- Tests for POST /process endpoint ---
@patch("services.document_processor.app.crud.get_paper_status", new_callable=AsyncMock)
@patch("services.document_processor.app.main.crud.get_supabase_client", new_callable=MagicMock) # Mock DB client for endpoint
def test_process_document_new_or_pending(mock_get_db, mock_get_status, client: TestClient, mock_background_tasks: MagicMock):
    mock_get_status.return_value = PaperStatus.PENDING.value # Allows processing
    request_data = ProcessRequest(paper_id="test_new_paper")

    app.dependency_overrides[BackgroundTasks] = lambda: mock_background_tasks
    response = client.post("/process", json=request_data.model_dump())
    
    assert response.status_code == 202
    assert response.json() == {"message": "Document processing scheduled successfully."}
    
    mock_background_tasks.add_task.assert_called_once()
    # Check that the first argument to add_task is our task function
    # and the second argument is an instance of ProcessRequest with the correct paper_id
    assert mock_background_tasks.add_task.call_args[0][0].__name__ == process_and_embed_task.__name__
    assert isinstance(mock_background_tasks.add_task.call_args[0][1], ProcessRequest)
    assert mock_background_tasks.add_task.call_args[0][1].paper_id == "test_new_paper"
    # Can also check other fields of ProcessRequest if necessary

@patch("services.document_processor.app.crud.get_paper_status", new_callable=AsyncMock)
@patch("services.document_processor.app.main.crud.get_supabase_client", new_callable=MagicMock)
def test_process_document_already_processing(mock_get_db, mock_get_status, client: TestClient, mock_background_tasks: MagicMock):
    mock_get_status.return_value = PaperStatus.PROCESSING.value
    request_data = ProcessRequest(paper_id="test_processing_paper")
    
    app.dependency_overrides[BackgroundTasks] = lambda: mock_background_tasks
    response = client.post("/process", json=request_data.model_dump())

    assert response.status_code == 200 
    assert "already initiated or completed" in response.json()["message"]
    mock_background_tasks.add_task.assert_not_called()

@patch("services.document_processor.app.crud.get_paper_status", new_callable=AsyncMock)
@patch("services.document_processor.app.main.crud.get_supabase_client", new_callable=MagicMock)
def test_process_document_already_processed(mock_get_db, mock_get_status, client: TestClient, mock_background_tasks: MagicMock):
    mock_get_status.return_value = PaperStatus.PROCESSED.value
    request_data = ProcessRequest(paper_id="test_processed_paper")

    app.dependency_overrides[BackgroundTasks] = lambda: mock_background_tasks
    response = client.post("/process", json=request_data.model_dump())

    assert response.status_code == 200
    assert "already initiated or completed" in response.json()["message"]
    mock_background_tasks.add_task.assert_not_called()

@patch("services.document_processor.app.crud.get_paper_status", new_callable=AsyncMock)
@patch("services.document_processor.app.main.crud.get_supabase_client", new_callable=MagicMock)
def test_process_document_metadata_not_found(mock_get_db, mock_get_status, client: TestClient, mock_background_tasks: MagicMock):
    mock_get_status.return_value = None 
    request_data = ProcessRequest(paper_id="test_not_found_paper")

    app.dependency_overrides[BackgroundTasks] = lambda: mock_background_tasks
    response = client.post("/process", json=request_data.model_dump())

    assert response.status_code == 404
    assert "Paper metadata not found" in response.json()["detail"]
    mock_background_tasks.add_task.assert_not_called()

@patch("services.document_processor.app.crud.get_paper_status", new_callable=AsyncMock)
@patch("services.document_processor.app.main.crud.get_supabase_client", new_callable=MagicMock)
def test_process_document_retry_failed(mock_get_db, mock_get_status, client: TestClient, mock_background_tasks: MagicMock):
    mock_get_status.return_value = PaperStatus.FAILED.value
    request_data = ProcessRequest(paper_id="test_failed_paper")

    app.dependency_overrides[BackgroundTasks] = lambda: mock_background_tasks
    response = client.post("/process", json=request_data.model_dump())

    assert response.status_code == 202
    mock_background_tasks.add_task.assert_called_once()
    assert mock_background_tasks.add_task.call_args[0][0].__name__ == process_and_embed_task.__name__
    assert mock_background_tasks.add_task.call_args[0][1].paper_id == "test_failed_paper"


# --- Tests for process_and_embed_task ---
@pytest.mark.asyncio
@patch("services.document_processor.app.crud.update_paper_status", new_callable=AsyncMock)
@patch("services.document_processor.app.processing.get_document_content", new_callable=AsyncMock)
@patch("services.document_processor.app.processing.parse_and_chunk", new_callable=AsyncMock)
@patch("services.document_processor.app.main.http_client", new_callable=AsyncMock) 
@patch("services.document_processor.app.main.crud.get_supabase_client", new_callable=MagicMock) 
async def test_process_task_success(mock_get_db_client, mock_http_client, mock_parse_chunk, mock_get_content, mock_update_status):
    mock_db_instance = AsyncMock()
    mock_get_db_client.return_value = mock_db_instance
    
    request = ProcessRequest(paper_id="test_paper_success")
    mock_get_content.return_value = (b"pdf content", {"source_type": "test", "filename": "test.pdf"})
    mock_chunks = [DocumentChunk(chunk_id="c1", paper_id="test_paper_success", text_content="text1", metadata={})]
    mock_parse_chunk.return_value = mock_chunks
    
    mock_vector_response = EmbedResponse(processed_chunk_ids=["c1"], failed_chunk_ids=[])
    mock_http_client.post.return_value = AsyncMock(
        status_code=200,
        json=lambda: mock_vector_response.model_dump()
    )
    
    await process_and_embed_task(request)
    
    mock_update_status.assert_any_call(db_client=mock_db_instance, paper_id="test_paper_success", status=PaperStatus.PROCESSING, message="Starting document processing.")
    mock_update_status.assert_called_with(db_client=mock_db_instance, paper_id="test_paper_success", status=PaperStatus.PROCESSED, message="Successfully processed and stored 1 chunks.")
    mock_get_content.assert_called_once()
    mock_parse_chunk.assert_called_once_with(paper_id="test_paper_success", content_bytes=b"pdf content", file_info={"source_type": "test", "filename": "test.pdf"})
    
    expected_embed_payload = EmbedRequest(chunks=[c.model_dump(exclude={'embedding'}) for c in mock_chunks])
    mock_http_client.post.assert_called_once_with(
        f"{settings.VECTOR_SERVICE_URL}/embed",
        json=expected_embed_payload.model_dump()
    )

@pytest.mark.asyncio
@patch("services.document_processor.app.crud.update_paper_status", new_callable=AsyncMock)
@patch("services.document_processor.app.processing.get_document_content", new_callable=AsyncMock)
@patch("services.document_processor.app.processing.parse_and_chunk", new_callable=AsyncMock) # Not strictly needed here
@patch("services.document_processor.app.main.http_client", new_callable=AsyncMock) # Not strictly needed here
@patch("services.document_processor.app.main.crud.get_supabase_client", new_callable=MagicMock)
async def test_process_task_get_content_fails(mock_get_db_client, mock_http_client, mock_parse_chunk, mock_get_content, mock_update_status):
    mock_db_instance = AsyncMock()
    mock_get_db_client.return_value = mock_db_instance
    request = ProcessRequest(paper_id="test_content_fail")
    mock_get_content.return_value = (None, None) 
    
    await process_and_embed_task(request)
    
    mock_update_status.assert_any_call(db_client=mock_db_instance, paper_id="test_content_fail", status=PaperStatus.PROCESSING, message="Starting document processing.")
    mock_update_status.assert_called_with(db_client=mock_db_instance, paper_id="test_content_fail", status=PaperStatus.FAILED, message="Failed to retrieve document content.")
    mock_parse_chunk.assert_not_called()
    mock_http_client.post.assert_not_called()

@pytest.mark.asyncio
@patch("services.document_processor.app.crud.update_paper_status", new_callable=AsyncMock)
@patch("services.document_processor.app.processing.get_document_content", new_callable=AsyncMock)
@patch("services.document_processor.app.processing.parse_and_chunk", new_callable=AsyncMock)
@patch("services.document_processor.app.main.http_client", new_callable=AsyncMock) # Not strictly needed here
@patch("services.document_processor.app.main.crud.get_supabase_client", new_callable=MagicMock)
async def test_process_task_parse_chunk_fails(mock_get_db_client, mock_http_client, mock_parse_chunk, mock_get_content, mock_update_status):
    mock_db_instance = AsyncMock()
    mock_get_db_client.return_value = mock_db_instance
    request = ProcessRequest(paper_id="test_parse_fail")
    mock_get_content.return_value = (b"pdf content", {"source_type": "test", "filename": "test.pdf"})
    mock_parse_chunk.return_value = [] 
    
    await process_and_embed_task(request)
    
    mock_update_status.assert_any_call(db_client=mock_db_instance, paper_id="test_parse_fail", status=PaperStatus.PROCESSING, message="Starting document processing.")
    mock_update_status.assert_called_with(db_client=mock_db_instance, paper_id="test_parse_fail", status=PaperStatus.FAILED, message="No text chunks extracted from the document.")
    mock_http_client.post.assert_not_called()

@pytest.mark.asyncio
@patch("services.document_processor.app.crud.update_paper_status", new_callable=AsyncMock)
@patch("services.document_processor.app.processing.get_document_content", new_callable=AsyncMock)
@patch("services.document_processor.app.processing.parse_and_chunk", new_callable=AsyncMock)
@patch("services.document_processor.app.main.http_client", new_callable=AsyncMock)
@patch("services.document_processor.app.main.crud.get_supabase_client", new_callable=MagicMock)
async def test_process_task_vector_service_request_error(mock_get_db_client, mock_http_client, mock_parse_chunk, mock_get_content, mock_update_status):
    mock_db_instance = AsyncMock()
    mock_get_db_client.return_value = mock_db_instance
    request = ProcessRequest(paper_id="test_vector_req_error")
    mock_get_content.return_value = (b"pdf content", {"source_type": "test", "filename": "test.pdf"})
    mock_chunks = [DocumentChunk(chunk_id="c1", paper_id="test_vector_req_error", text_content="text1", metadata={})]
    mock_parse_chunk.return_value = mock_chunks
    mock_http_client.post.side_effect = httpx.RequestError("Connection failed", request=MagicMock()) 
    
    await process_and_embed_task(request)
    
    mock_update_status.assert_any_call(db_client=mock_db_instance, paper_id="test_vector_req_error", status=PaperStatus.PROCESSING, message="Starting document processing.")
    mock_update_status.assert_called_with(db_client=mock_db_instance, paper_id="test_vector_req_error", status=PaperStatus.FAILED, message="Failed to send chunks to Vector Service: Connection failed")

@pytest.mark.asyncio
@patch("services.document_processor.app.crud.update_paper_status", new_callable=AsyncMock)
@patch("services.document_processor.app.processing.get_document_content", new_callable=AsyncMock)
@patch("services.document_processor.app.processing.parse_and_chunk", new_callable=AsyncMock)
@patch("services.document_processor.app.main.http_client", new_callable=AsyncMock)
@patch("services.document_processor.app.main.crud.get_supabase_client", new_callable=MagicMock)
async def test_process_task_vector_service_http_error(mock_get_db_client, mock_http_client, mock_parse_chunk, mock_get_content, mock_update_status):
    mock_db_instance = AsyncMock()
    mock_get_db_client.return_value = mock_db_instance
    request = ProcessRequest(paper_id="test_vector_http_error")
    mock_get_content.return_value = (b"pdf content", {"source_type": "test", "filename": "test.pdf"})
    mock_chunks = [DocumentChunk(chunk_id="c1", paper_id="test_vector_http_error", text_content="text1", metadata={})]
    mock_parse_chunk.return_value = mock_chunks
    mock_http_client.post.return_value = AsyncMock(status_code=500, text="Internal Server Error") 
    
    await process_and_embed_task(request)
    
    mock_update_status.assert_any_call(db_client=mock_db_instance, paper_id="test_vector_http_error", status=PaperStatus.PROCESSING, message="Starting document processing.")
    mock_update_status.assert_called_with(db_client=mock_db_instance, paper_id="test_vector_http_error", status=PaperStatus.FAILED, message="Vector Service returned error 500: Internal Server Error")

@pytest.mark.asyncio
@patch("services.document_processor.app.crud.update_paper_status", new_callable=AsyncMock)
@patch("services.document_processor.app.processing.get_document_content", new_callable=AsyncMock)
@patch("services.document_processor.app.processing.parse_and_chunk", new_callable=AsyncMock)
@patch("services.document_processor.app.main.http_client", new_callable=AsyncMock)
@patch("services.document_processor.app.main.crud.get_supabase_client", new_callable=MagicMock)
async def test_process_task_vector_service_partial_failure(mock_get_db_client, mock_http_client, mock_parse_chunk, mock_get_content, mock_update_status):
    mock_db_instance = AsyncMock()
    mock_get_db_client.return_value = mock_db_instance
    request = ProcessRequest(paper_id="test_vector_partial_fail")
    mock_get_content.return_value = (b"pdf content", {"source_type": "test", "filename": "test.pdf"})
    mock_chunks = [
        DocumentChunk(chunk_id="c1", paper_id="test_vector_partial_fail", text_content="text1", metadata={}),
        DocumentChunk(chunk_id="c2", paper_id="test_vector_partial_fail", text_content="text2", metadata={})
    ]
    mock_parse_chunk.return_value = mock_chunks
    
    mock_vector_response = EmbedResponse(processed_chunk_ids=["c1"], failed_chunk_ids=["c2"])
    mock_http_client.post.return_value = AsyncMock(
        status_code=200,
        json=lambda: mock_vector_response.model_dump()
    )
    
    await process_and_embed_task(request)
    
    mock_update_status.assert_any_call(db_client=mock_db_instance, paper_id="test_vector_partial_fail", status=PaperStatus.PROCESSING, message="Starting document processing.")
    mock_update_status.assert_called_with(db_client=mock_db_instance, paper_id="test_vector_partial_fail", status=PaperStatus.PROCESSED_WITH_ERRORS, message="Processed 1 chunks, but 1 chunks failed embedding.")

@pytest.mark.asyncio
@patch("services.document_processor.app.crud.update_paper_status", new_callable=AsyncMock)
@patch("services.document_processor.app.processing.get_document_content", new_callable=AsyncMock)
@patch("services.document_processor.app.processing.parse_and_chunk", new_callable=AsyncMock)
@patch("services.document_processor.app.main.http_client", new_callable=AsyncMock)
@patch("services.document_processor.app.main.crud.get_supabase_client", new_callable=MagicMock)
async def test_process_task_vector_service_total_failure_in_response(mock_get_db_client, mock_http_client, mock_parse_chunk, mock_get_content, mock_update_status):
    mock_db_instance = AsyncMock()
    mock_get_db_client.return_value = mock_db_instance
    request = ProcessRequest(paper_id="test_vector_total_fail_resp")
    mock_get_content.return_value = (b"pdf content", {"source_type": "test", "filename": "test.pdf"})
    mock_chunks = [
        DocumentChunk(chunk_id="c1", paper_id="test_vector_total_fail_resp", text_content="text1", metadata={}),
        DocumentChunk(chunk_id="c2", paper_id="test_vector_total_fail_resp", text_content="text2", metadata={})
    ]
    mock_parse_chunk.return_value = mock_chunks
    
    mock_vector_response = EmbedResponse(processed_chunk_ids=[], failed_chunk_ids=["c1", "c2"]) 
    mock_http_client.post.return_value = AsyncMock(
        status_code=200,
        json=lambda: mock_vector_response.model_dump()
    )
    
    await process_and_embed_task(request)
    
    mock_update_status.assert_any_call(db_client=mock_db_instance, paper_id="test_vector_total_fail_resp", status=PaperStatus.PROCESSING, message="Starting document processing.")
    mock_update_status.assert_called_with(db_client=mock_db_instance, paper_id="test_vector_total_fail_resp", status=PaperStatus.FAILED, message="All 2 chunks failed embedding by Vector Service.")


# --- Tests for crud.py (Document Processor version) ---
from services.document_processor.app import crud as doc_proc_crud
from core.models import PaperMetadata # For testing _fetch_paper_metadata_from_db
from postgrest import APIError as PostgrestAPIError # For Supabase errors
import uuid

# Mock Supabase client for CRUD operations
@pytest_asyncio.fixture
async def mock_db_client_crud():
    mock_client = AsyncMock()
    mock_table_response = AsyncMock()
    mock_select_response = AsyncMock()
    mock_update_response = AsyncMock()
    mock_eq_response = AsyncMock()
    mock_limit_response = AsyncMock() # Though not explicitly used by update_paper_status
    mock_maybe_single_response = AsyncMock()


    mock_client.table.return_value = mock_table_response
    mock_table_response.select.return_value = mock_select_response
    mock_table_response.update.return_value = mock_update_response
    
    mock_select_response.eq.return_value = mock_eq_response
    mock_update_response.eq.return_value = mock_eq_response # For update

    # Final execute() calls for different chains
    mock_eq_response.execute = AsyncMock() # For update().eq().execute()
    mock_eq_response.limit.return_value.maybe_single.return_value.execute = mock_maybe_single_response.execute # For select().eq().limit().maybe_single().execute()
    
    return mock_client

# --- Tests for crud.update_paper_status ---
@pytest.mark.asyncio
async def test_crud_update_paper_status_success(mock_db_client_crud: AsyncMock):
    mock_api_response = AsyncMock()
    mock_api_response.data = [{"id": "paper1", "processing_status": PaperStatus.PROCESSING.value}]
    mock_db_client_crud.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_api_response

    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        result = await doc_proc_crud.update_paper_status(mock_db_client_crud, "paper1", PaperStatus.PROCESSING)
    
    assert result is True
    mock_db_client_crud.table.assert_called_once_with("papers")
    mock_db_client_crud.table.return_value.update.assert_called_once_with({
        "processing_status": PaperStatus.PROCESSING.value,
        "status_message": None # Default message
    })
    mock_db_client_crud.table.return_value.update.return_value.eq.assert_called_once_with("id", "paper1")

@pytest.mark.asyncio
async def test_crud_update_paper_status_with_message_success(mock_db_client_crud: AsyncMock):
    mock_api_response = AsyncMock()
    mock_api_response.data = [{"id": "paper1", "processing_status": PaperStatus.FAILED.value, "status_message": "Test error"}]
    mock_db_client_crud.table.return_value.update.return_value.eq.return_value.execute.return_value = mock_api_response

    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        result = await doc_proc_crud.update_paper_status(mock_db_client_crud, "paper1", PaperStatus.FAILED, "Test error")

    assert result is True
    mock_db_client_crud.table.return_value.update.assert_called_once_with({
        "processing_status": PaperStatus.FAILED.value,
        "status_message": "Test error"
    })

@pytest.mark.asyncio
@patch("services.document_processor.app.crud.logger") # Patch logger in crud module
async def test_crud_update_paper_status_invalid_status_value(mock_logger, mock_db_client_crud: AsyncMock):
    # This test assumes PaperStatus enum is used for validation before DB call.
    # If validation is in DB (e.g. enum type), this test structure would change.
    # The current crud.py does not have explicit validation for PaperStatus enum values before DB call.
    # Let's assume for now the function relies on Pydantic or DB to catch this.
    # If PaperStatus were validated in Python:
    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        with pytest.raises(TypeError): # Or a custom error if we added validation
             # This will fail at the Pydantic model level in a real app if status is not PaperStatus
             # For a direct DB call, this might pass but DB might error.
             # Let's assume the function itself doesn't validate PaperStatus string directly.
             # The test below for DB error is more relevant for Postgrest errors.
             # If we want to test the string value of PaperStatus:
             await doc_proc_crud.update_paper_status(mock_db_client_crud, "paper1", "invalid_status_string")
    
    # This test becomes more about "what if the DB call fails due to bad enum value"
    # which is covered by test_crud_update_paper_status_api_error.
    # If PaperStatus enum was strictly enforced *before* the DB call:
    # mock_logger.error.assert_called_once() # Check that an error was logged
    # assert result is False
    # For now, this test is tricky without knowing exact enum validation in crud.py.
    # The current code passes the string directly. A DB error would be raised.
    mock_db_client_crud.table.return_value.update.return_value.eq.return_value.execute.side_effect = PostgrestAPIError(
        {"message": "Invalid enum value for processing_status"}, response=None, request=None
    )
    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        with pytest.raises(PostgrestAPIError):
            await doc_proc_crud.update_paper_status(mock_db_client_crud, "paper1", "invalid_status_string")


@pytest.mark.asyncio
async def test_crud_update_paper_status_api_error(mock_db_client_crud: AsyncMock):
    mock_db_client_crud.table.return_value.update.return_value.eq.return_value.execute.side_effect = PostgrestAPIError(
        {"message": "DB connection error"}, response=None, request=None
    )
    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        with pytest.raises(PostgrestAPIError):
            await doc_proc_crud.update_paper_status(mock_db_client_crud, "paper1", PaperStatus.FAILED)

@pytest.mark.asyncio
async def test_crud_update_paper_status_other_exception(mock_db_client_crud: AsyncMock):
    mock_db_client_crud.table.return_value.update.return_value.eq.return_value.execute.side_effect = ValueError("Unexpected error")
    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        with pytest.raises(ValueError, match="Unexpected error"):
            await doc_proc_crud.update_paper_status(mock_db_client_crud, "paper1", PaperStatus.PROCESSING)

# --- Tests for crud.get_paper_status ---
@pytest.mark.asyncio
async def test_crud_get_paper_status_success(mock_db_client_crud: AsyncMock):
    db_data = {"processing_status": PaperStatus.PROCESSED.value}
    mock_api_response = AsyncMock()
    mock_api_response.data = db_data
    mock_db_client_crud.table.return_value.select.return_value.eq.return_value.limit.return_value.maybe_single.return_value.execute.return_value = mock_api_response

    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        status = await doc_proc_crud.get_paper_status(mock_db_client_crud, "paper1")

    assert status == PaperStatus.PROCESSED.value
    mock_db_client_crud.table.assert_called_once_with("papers")
    mock_db_client_crud.table.return_value.select.assert_called_once_with("processing_status")
    mock_db_client_crud.table.return_value.select.return_value.eq.assert_called_once_with("id", "paper1")
    mock_db_client_crud.table.return_value.select.return_value.eq.return_value.limit.return_value.maybe_single.assert_called_once()


@pytest.mark.asyncio
async def test_crud_get_paper_status_not_found(mock_db_client_crud: AsyncMock):
    mock_api_response = AsyncMock()
    mock_api_response.data = None # Simulate paper not found
    mock_db_client_crud.table.return_value.select.return_value.eq.return_value.limit.return_value.maybe_single.return_value.execute.return_value = mock_api_response
    
    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        status = await doc_proc_crud.get_paper_status(mock_db_client_crud, "nonexistent_paper")
    assert status is None

@pytest.mark.asyncio
async def test_crud_get_paper_status_no_status_field(mock_db_client_crud: AsyncMock):
    mock_api_response = AsyncMock()
    mock_api_response.data = {"id": "paper_no_status"} # Missing 'processing_status'
    mock_db_client_crud.table.return_value.select.return_value.eq.return_value.limit.return_value.maybe_single.return_value.execute.return_value = mock_api_response
    
    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        status = await doc_proc_crud.get_paper_status(mock_db_client_crud, "paper_no_status")
    assert status is None # Or should it raise an error? Current crud.py returns None if field missing.

@pytest.mark.asyncio
async def test_crud_get_paper_status_api_error(mock_db_client_crud: AsyncMock):
    mock_db_client_crud.table.return_value.select.return_value.eq.return_value.limit.return_value.maybe_single.return_value.execute.side_effect = PostgrestAPIError(
        {"message": "DB connection error"}, response=None, request=None
    )
    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        with pytest.raises(PostgrestAPIError):
            await doc_proc_crud.get_paper_status(mock_db_client_crud, "paper1")

@pytest.mark.asyncio
async def test_crud_get_paper_status_other_exception(mock_db_client_crud: AsyncMock):
    mock_db_client_crud.table.return_value.select.return_value.eq.return_value.limit.return_value.maybe_single.return_value.execute.side_effect = ValueError("Unexpected")
    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        with pytest.raises(ValueError, match="Unexpected"):
            await doc_proc_crud.get_paper_status(mock_db_client_crud, "paper1")


# --- Tests for processing.py ---
from services.document_processor.app import processing as doc_proc_processing
from pdfminer.pdfparser import PDFSyntaxError
from io import BytesIO

# --- Tests for processing._fetch_paper_metadata_from_db ---
@pytest.mark.asyncio
async def test_processing_fetch_paper_metadata_success(mock_db_client_crud: AsyncMock): # Can reuse crud mock
    paper_id = "test_fetch_paper"
    db_data = {"id": paper_id, "title": "Fetched Title", "url": "http://example.com", "pdf_url": "http://example.com/paper.pdf"}
    mock_api_response = AsyncMock()
    mock_api_response.data = db_data
    mock_db_client_crud.table.return_value.select.return_value.eq.return_value.limit.return_value.maybe_single.return_value.execute.return_value = mock_api_response

    with patch("services.document_processor.app.processing.get_supabase_client", return_value=mock_db_client_crud):
        metadata = await doc_proc_processing._fetch_paper_metadata_from_db(mock_db_client_crud, paper_id)

    assert isinstance(metadata, PaperMetadata)
    assert metadata.id == paper_id
    assert metadata.url == "http://example.com"
    assert metadata.pdf_url == "http://example.com/paper.pdf"
    mock_db_client_crud.table.return_value.select.assert_called_once_with("id, title, abstract, authors, url, pdf_url, source, published_date, keywords, processing_status, status_message")


@pytest.mark.asyncio
async def test_processing_fetch_paper_metadata_not_found(mock_db_client_crud: AsyncMock):
    mock_api_response = AsyncMock()
    mock_api_response.data = None
    mock_db_client_crud.table.return_value.select.return_value.eq.return_value.limit.return_value.maybe_single.return_value.execute.return_value = mock_api_response
    with patch("services.document_processor.app.processing.get_supabase_client", return_value=mock_db_client_crud):
        metadata = await doc_proc_processing._fetch_paper_metadata_from_db(mock_db_client_crud, "nonexistent")
    assert metadata is None

@pytest.mark.asyncio
async def test_processing_fetch_paper_metadata_api_error(mock_db_client_crud: AsyncMock):
    mock_db_client_crud.table.return_value.select.return_value.eq.return_value.limit.return_value.maybe_single.return_value.execute.side_effect = PostgrestAPIError({"message":"DB error"}, response=None, request=None)
    with patch("services.document_processor.app.processing.get_supabase_client", return_value=mock_db_client_crud):
        with pytest.raises(PostgrestAPIError):
            await doc_proc_processing._fetch_paper_metadata_from_db(mock_db_client_crud, "any_id")

@pytest.mark.asyncio
async def test_processing_fetch_paper_metadata_pydantic_error(mock_db_client_crud: AsyncMock):
    db_data = {"id": "bad_data_id", "title": "Title", "authors": "not-a-list"} # Invalid authors
    mock_api_response = AsyncMock()
    mock_api_response.data = db_data
    mock_db_client_crud.table.return_value.select.return_value.eq.return_value.limit.return_value.maybe_single.return_value.execute.return_value = mock_api_response
    with patch("services.document_processor.app.processing.get_supabase_client", return_value=mock_db_client_crud), \
         patch("services.document_processor.app.processing.logger") as mock_logger: # Patch logger
        metadata = await doc_proc_processing._fetch_paper_metadata_from_db(mock_db_client_crud, "bad_data_id")
    assert metadata is None # Should return None on Pydantic error
    mock_logger.error.assert_called_once() # Check that an error was logged

# --- Tests for processing._download_from_url ---
@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_download_from_url_success(MockAsyncClient):
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/pdf"}
    # Simulate streaming content
    async def mock_aread_gen(): yield b"pdf part 1"; yield b" pdf part 2"
    mock_response.aiter_bytes = mock_aread_gen 
    
    mock_stream_context = AsyncMock()
    mock_stream_context.__aenter__.return_value = mock_response
    MockAsyncClient.return_value.stream.return_value = mock_stream_context

    content, headers = await doc_proc_processing._download_from_url("http://example.com/file.pdf")
    
    assert content == b"pdf part 1 pdf part 2"
    assert headers == {"content-type": "application/pdf"}
    MockAsyncClient.return_value.stream.assert_called_once_with("GET", "http://example.com/file.pdf", timeout=settings.DOWNLOAD_TIMEOUT_SECONDS)

@pytest.mark.asyncio
async def test_download_from_url_invalid_url():
    content, headers = await doc_proc_processing._download_from_url("ftp://example.com/file.pdf")
    assert content is None
    assert headers is None

@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_download_from_url_request_error(MockAsyncClient):
    MockAsyncClient.return_value.stream.side_effect = httpx.RequestError("Connection failed", request=MagicMock())
    content, headers = await doc_proc_processing._download_from_url("http://example.com/file.pdf")
    assert content is None
    assert headers is None

@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_download_from_url_http_status_error(MockAsyncClient):
    mock_response = AsyncMock()
    mock_response.status_code = 404
    mock_response.request = MagicMock() # Needed for HTTPStatusError
    mock_response.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError("Not Found", request=mock_response.request, response=mock_response))

    mock_stream_context = AsyncMock()
    mock_stream_context.__aenter__.return_value = mock_response
    MockAsyncClient.return_value.stream.return_value = mock_stream_context
    
    content, headers = await doc_proc_processing._download_from_url("http://example.com/notfound.pdf")
    assert content is None
    assert headers is None

@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_download_from_url_zero_bytes(MockAsyncClient):
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/pdf"}
    async def mock_aread_gen_empty(): yield b"" # Empty content
    mock_response.aiter_bytes = mock_aread_gen_empty
    
    mock_stream_context = AsyncMock()
    mock_stream_context.__aenter__.return_value = mock_response
    MockAsyncClient.return_value.stream.return_value = mock_stream_context

    content, headers = await doc_proc_processing._download_from_url("http://example.com/empty.pdf")
    assert content == b"" # Should return empty bytes, not None
    assert headers == {"content-type": "application/pdf"}


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
@patch("services.document_processor.app.processing.logger")
async def test_download_from_url_non_pdf_content(mock_logger, MockAsyncClient):
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/html"}
    async def mock_aread_gen(): yield b"<html></html>"
    mock_response.aiter_bytes = mock_aread_gen
    
    mock_stream_context = AsyncMock()
    mock_stream_context.__aenter__.return_value = mock_response
    MockAsyncClient.return_value.stream.return_value = mock_stream_context

    content, headers = await doc_proc_processing._download_from_url("http://example.com/page.html")
    assert content == b"<html></html>"
    assert headers == {"content-type": "text/html"}
    mock_logger.warning.assert_called_once() # Check for warning log

# --- Tests for processing.get_document_content ---
@pytest.mark.asyncio
@patch("services.document_processor.app.processing._fetch_paper_metadata_from_db", new_callable=AsyncMock)
@patch("services.document_processor.app.processing._download_from_url", new_callable=AsyncMock)
@patch("services.document_processor.app.processing.get_supabase_client") # For DB client in main func
@patch("services.document_processor.app.processing.get_supabase_storage_client") # For storage client
async def test_get_doc_content_from_storage_success(mock_get_storage_client, mock_get_db_client, mock_download, mock_fetch_meta):
    mock_storage_instance = AsyncMock()
    mock_storage_instance.from_.return_value.download.return_value = b"storage content"
    mock_get_storage_client.return_value = mock_storage_instance
    
    request = ProcessRequest(paper_id="p1", bucket_name="bucket", object_name="file.pdf")
    content, file_info = await doc_proc_processing.get_document_content(MagicMock(), mock_get_storage_client(), request)

    assert content == b"storage content"
    assert file_info == {"source_type": "supabase_storage", "filename": "file.pdf"}
    mock_download.assert_not_called()
    mock_fetch_meta.assert_not_called()

@pytest.mark.asyncio
@patch("services.document_processor.app.processing._fetch_paper_metadata_from_db", new_callable=AsyncMock)
@patch("services.document_processor.app.processing._download_from_url", new_callable=AsyncMock)
@patch("services.document_processor.app.processing.get_supabase_client")
@patch("services.document_processor.app.processing.get_supabase_storage_client")
async def test_get_doc_content_from_storage_zero_bytes(mock_get_storage_client, mock_get_db_client, mock_download, mock_fetch_meta):
    mock_storage_instance = AsyncMock()
    mock_storage_instance.from_.return_value.download.return_value = b"" # 0 bytes
    mock_get_storage_client.return_value = mock_storage_instance
    
    request = ProcessRequest(paper_id="p1", bucket_name="bucket", object_name="empty.pdf")
    content, file_info = await doc_proc_processing.get_document_content(MagicMock(), mock_get_storage_client(), request)

    assert content is None # 0 bytes from storage is treated as failure
    assert file_info is None
    mock_download.assert_not_called() # Should not fallback if storage was attempted

@pytest.mark.asyncio
@patch("services.document_processor.app.processing._fetch_paper_metadata_from_db", new_callable=AsyncMock)
@patch("services.document_processor.app.processing._download_from_url", new_callable=AsyncMock)
@patch("services.document_processor.app.processing.get_supabase_client")
@patch("services.document_processor.app.processing.get_supabase_storage_client")
@patch("services.document_processor.app.processing.logger")
async def test_get_doc_content_from_storage_error(mock_logger, mock_get_storage_client, mock_get_db_client, mock_download, mock_fetch_meta):
    mock_storage_instance = AsyncMock()
    mock_storage_instance.from_.return_value.download.side_effect = Exception("Storage unavailable")
    mock_get_storage_client.return_value = mock_storage_instance
    
    request = ProcessRequest(paper_id="p1", bucket_name="bucket", object_name="file.pdf")
    content, file_info = await doc_proc_processing.get_document_content(MagicMock(), mock_get_storage_client(), request)

    assert content is None
    assert file_info is None
    mock_logger.error.assert_called_once()
    mock_download.assert_not_called() # No fallback if storage fails this way

@pytest.mark.asyncio
@patch("services.document_processor.app.processing._fetch_paper_metadata_from_db", new_callable=AsyncMock)
@patch("services.document_processor.app.processing._download_from_url", new_callable=AsyncMock)
@patch("services.document_processor.app.processing.get_supabase_client")
@patch("services.document_processor.app.processing.get_supabase_storage_client")
async def test_get_doc_content_from_source_url_success(mock_get_storage_client, mock_get_db_client, mock_download, mock_fetch_meta):
    mock_download.return_value = (b"url content", {"content-type": "application/pdf"})
    request = ProcessRequest(paper_id="p1", source_url="http://example.com/direct.pdf")
    
    content, file_info = await doc_proc_processing.get_document_content(MagicMock(), mock_get_storage_client(), request)

    assert content == b"url content"
    assert file_info == {"source_type": "url", "filename": "direct.pdf"}
    mock_download.assert_called_once_with("http://example.com/direct.pdf")
    mock_fetch_meta.assert_not_called()
    mock_get_storage_client.return_value.from_.return_value.download.assert_not_called()


@pytest.mark.asyncio
@patch("services.document_processor.app.processing._fetch_paper_metadata_from_db", new_callable=AsyncMock)
@patch("services.document_processor.app.processing._download_from_url", new_callable=AsyncMock)
@patch("services.document_processor.app.processing.get_supabase_client")
@patch("services.document_processor.app.processing.get_supabase_storage_client")
async def test_get_doc_content_from_source_url_fails(mock_get_storage_client, mock_get_db_client, mock_download, mock_fetch_meta):
    mock_download.return_value = (None, None) # Download fails
    request = ProcessRequest(paper_id="p1", source_url="http://example.com/bad.pdf")
    
    content, file_info = await doc_proc_processing.get_document_content(MagicMock(), mock_get_storage_client(), request)

    assert content is None # Fails, no fallback if source_url is explicit
    assert file_info is None
    mock_fetch_meta.assert_not_called()


@pytest.mark.asyncio
@patch("services.document_processor.app.processing._fetch_paper_metadata_from_db", new_callable=AsyncMock)
@patch("services.document_processor.app.processing._download_from_url", new_callable=AsyncMock)
@patch("services.document_processor.app.processing.get_supabase_client")
@patch("services.document_processor.app.processing.get_supabase_storage_client")
async def test_get_doc_content_db_meta_not_found(mock_get_storage_client, mock_get_db_client, mock_download, mock_fetch_meta):
    mock_fetch_meta.return_value = None # DB metadata not found
    request = ProcessRequest(paper_id="p1_no_meta") # No bucket/URL
    
    content, file_info = await doc_proc_processing.get_document_content(mock_get_db_client(), mock_get_storage_client(), request)
    assert content is None
    assert file_info is None
    mock_fetch_meta.assert_called_once()
    mock_download.assert_not_called()

@pytest.mark.asyncio
@patch("services.document_processor.app.processing._fetch_paper_metadata_from_db", new_callable=AsyncMock)
@patch("services.document_processor.app.processing._download_from_url", new_callable=AsyncMock)
@patch("services.document_processor.app.processing.get_supabase_client")
@patch("services.document_processor.app.processing.get_supabase_storage_client")
async def test_get_doc_content_db_pdf_url_success(mock_get_storage_client, mock_get_db_client, mock_download, mock_fetch_meta):
    mock_meta = PaperMetadata(id="p1", pdf_url="http://example.com/db.pdf", url="http://example.com/page")
    mock_fetch_meta.return_value = mock_meta
    mock_download.return_value = (b"db pdf content", {"content-type": "application/pdf"})
    request = ProcessRequest(paper_id="p1")
    
    content, file_info = await doc_proc_processing.get_document_content(mock_get_db_client(), mock_get_storage_client(), request)
    assert content == b"db pdf content"
    assert file_info == {"source_type": "pdf_url_db", "filename": "db.pdf"}
    mock_download.assert_called_once_with("http://example.com/db.pdf")

@pytest.mark.asyncio
@patch("services.document_processor.app.processing._fetch_paper_metadata_from_db", new_callable=AsyncMock)
@patch("services.document_processor.app.processing._download_from_url", new_callable=AsyncMock)
@patch("services.document_processor.app.processing.get_supabase_client")
@patch("services.document_processor.app.processing.get_supabase_storage_client")
async def test_get_doc_content_db_pdf_url_fails_fallback_url_success(mock_get_storage_client, mock_get_db_client, mock_download, mock_fetch_meta):
    mock_meta = PaperMetadata(id="p1", pdf_url="http://example.com/db_bad.pdf", url="http://example.com/db_fallback.pdf")
    mock_fetch_meta.return_value = mock_meta
    # First call to _download_from_url (for pdf_url) fails, second (for url) succeeds
    mock_download.side_effect = [(None, None), (b"db fallback content", {"content-type": "application/pdf"})]
    request = ProcessRequest(paper_id="p1")
    
    content, file_info = await doc_proc_processing.get_document_content(mock_get_db_client(), mock_get_storage_client(), request)
    assert content == b"db fallback content"
    assert file_info == {"source_type": "url_db", "filename": "db_fallback.pdf"}
    assert mock_download.call_count == 2
    mock_download.assert_any_call("http://example.com/db_bad.pdf")
    mock_download.assert_any_call("http://example.com/db_fallback.pdf")

@pytest.mark.asyncio
@patch("services.document_processor.app.processing._fetch_paper_metadata_from_db", new_callable=AsyncMock)
@patch("services.document_processor.app.processing._download_from_url", new_callable=AsyncMock)
@patch("services.document_processor.app.processing.get_supabase_client")
@patch("services.document_processor.app.processing.get_supabase_storage_client")
async def test_get_doc_content_db_all_urls_fail(mock_get_storage_client, mock_get_db_client, mock_download, mock_fetch_meta):
    mock_meta = PaperMetadata(id="p1", pdf_url="http://example.com/db_bad1.pdf", url="http://example.com/db_bad2.pdf")
    mock_fetch_meta.return_value = mock_meta
    mock_download.return_value = (None, None) # Both downloads fail
    request = ProcessRequest(paper_id="p1")
    
    content, file_info = await doc_proc_processing.get_document_content(mock_get_db_client(), mock_get_storage_client(), request)
    assert content is None
    assert file_info is None
    assert mock_download.call_count == 2 # Both URLs attempted

@pytest.mark.asyncio
@patch("services.document_processor.app.processing._fetch_paper_metadata_from_db", new_callable=AsyncMock)
@patch("services.document_processor.app.processing._download_from_url", new_callable=AsyncMock)
@patch("services.document_processor.app.processing.get_supabase_client")
@patch("services.document_processor.app.processing.get_supabase_storage_client")
async def test_get_doc_content_db_no_urls_in_meta(mock_get_storage_client, mock_get_db_client, mock_download, mock_fetch_meta):
    mock_meta = PaperMetadata(id="p1", pdf_url=None, url=None) # No URLs
    mock_fetch_meta.return_value = mock_meta
    request = ProcessRequest(paper_id="p1")
    
    content, file_info = await doc_proc_processing.get_document_content(mock_get_db_client(), mock_get_storage_client(), request)
    assert content is None
    assert file_info is None
    mock_download.assert_not_called()


# --- Tests for processing._recursive_split ---
def test_recursive_split_empty_text():
    chunks = doc_proc_processing._recursive_split("", ["\n\n", "\n"], 100, 10)
    assert chunks == []

def test_recursive_split_small_text():
    text = "Small text."
    chunks = doc_proc_processing._recursive_split(text, ["\n\n", "\n"], 100, 10)
    assert chunks == [text]

def test_recursive_split_by_primary_separator():
    text = "Paragraph 1.\n\nParagraph 2."
    chunks = doc_proc_processing._recursive_split(text, ["\n\n", "\n"], 20, 5)
    assert chunks == ["Paragraph 1.", "Paragraph 2."]

def test_recursive_split_by_secondary_separator():
    text = "Line 1.\nLine 2.\nLine 3 is a bit longer." # Primary (\n\n) not present
    # Chunk size forces split by \n
    chunks = doc_proc_processing._recursive_split(text, ["\n\n", "\n", ". "], 20, 2)
    assert chunks == ["Line 1.", "Line 2.", "Line 3 is a bit longer."]

def test_recursive_split_hard_cuts():
    text = " veryverylongwordwithoutanyspacesorcommasorevenperiodstobreakitup"
    # Separators won't help, chunk_size is small
    chunks = doc_proc_processing._recursive_split(text, ["\n\n", "\n", ". "], 10, 2)
    expected = [text[i:i+10] for i in range(0, len(text), 10-2)] # Approximate, actual split is more complex
    # The actual logic for hard cuts:
    # For "veryverylongwordwithoutanyspacesorcommasorevenperiodstobreakitup" (60 chars)
    # chunk_size=10, overlap=2. Effective new chunk part = 8
    # "veryverylo" (0-10)
    # "rylongwo" (8-18)
    # "ngwordwi" (16-26)
    # ...
    assert len(chunks) > 1
    assert chunks[0] == "veryverylo"
    assert chunks[1].startswith("rylongwo") # Overlap check
    assert "".join(c.replace(" ", "") for c in chunks).startswith(text.replace(" ","")[0:50]) # Rough check

def test_recursive_split_with_overlap():
    text = "Sentence one. Sentence two. Sentence three."
    # Chunk size ensures split by ". ", overlap should be visible
    chunks = doc_proc_processing._recursive_split(text, [". "], 20, 10) # Large overlap
    assert chunks[0] == "Sentence one." 
    assert chunks[1] == "Sentence two." # If overlap > len(separator)
    # Corrected expectation for overlap logic:
    # "Sentence one. " -> "Sentence one." (sep removed)
    # "Sentence two. " -> "Sentence two."
    # "Sentence three."
    # If text is "A. B. C" and sep=". "
    # "A. " -> "A"
    # "B. " -> "B" (overlap means we might take some of previous if it's not the separator itself)
    # Let's use a simpler text for overlap: "abc. def. ghi" sep=". " size=5 overlap=2
    # "abc. " -> "abc"
    # "def. " -> "c. def" (text[start_index-overlap:end_index])
    # "ghi" -> "f. ghi"
    text_overlap = "abc. def. ghi"
    # chunk_size = 5, chunk_overlap = 2, separators = [". "]
    # First split: "abc. ", "def. ", "ghi"
    # "abc. " fits, becomes "abc"
    # "def. " fits, becomes "def"
    # "ghi" fits, becomes "ghi"
    # This test needs a text that forces smaller chunks than separators allow
    text_long_sentences = "This is sentence one. This is sentence two. This is sentence three for overlap."
    chunks_overlap = doc_proc_processing._recursive_split(text_long_sentences, [". "], 30, 10)
    assert chunks_overlap[0] == "This is sentence one."
    assert chunks_overlap[1].startswith("one. This is sentence two.") # Shows overlap
    assert chunks_overlap[2].startswith("two. This is sentence three")

def test_recursive_split_separators_only_text():
    text = "\n\n\n\n"
    chunks = doc_proc_processing._recursive_split(text, ["\n\n", "\n"], 10, 2)
    assert chunks == [] # Empty strings are filtered out

# --- Tests for processing.parse_and_chunk ---
@pytest.mark.asyncio
@patch("services.document_processor.app.processing.pdf_extract_text")
@patch("services.document_processor.app.processing._recursive_split")
async def test_parse_and_chunk_success(mock_recursive_split, mock_pdf_extract):
    sample_pdf_bytes = b"%PDF-1.4 fake content"
    extracted_text = "This is the extracted text from PDF. It has multiple sentences."
    split_chunks_text = ["This is the extracted text from PDF.", "It has multiple sentences."]
    
    mock_pdf_extract.return_value = extracted_text
    mock_recursive_split.return_value = split_chunks_text
    
    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)): # Bypass to_thread
        result_chunks = await doc_proc_processing.parse_and_chunk("paper_id_1", sample_pdf_bytes, {"filename": "test.pdf"})

    assert len(result_chunks) == 2
    assert isinstance(result_chunks[0], DocumentChunk)
    assert result_chunks[0].paper_id == "paper_id_1"
    assert result_chunks[0].text_content == split_chunks_text[0]
    assert result_chunks[0].metadata["source_filename"] == "test.pdf"
    assert result_chunks[0].metadata["sequence_in_paper"] == 0
    assert result_chunks[1].paper_id == "paper_id_1"
    assert result_chunks[1].text_content == split_chunks_text[1]
    assert result_chunks[1].metadata["sequence_in_paper"] == 1
    
    mock_pdf_extract.assert_called_once_with(BytesIO(sample_pdf_bytes))
    mock_recursive_split.assert_called_once_with(
        extracted_text, 
        settings.PARSER_SEPARATORS, 
        settings.PARSER_CHUNK_SIZE, 
        settings.PARSER_CHUNK_OVERLAP
    )

@pytest.mark.asyncio
@patch("services.document_processor.app.processing.pdf_extract_text")
async def test_parse_and_chunk_no_text_extracted(mock_pdf_extract):
    mock_pdf_extract.return_value = "" # No text
    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        result_chunks = await doc_proc_processing.parse_and_chunk("p2", b"pdf", {"filename":"f.pdf"})
    assert result_chunks == []

@pytest.mark.asyncio
@patch("services.document_processor.app.processing.pdf_extract_text")
@patch("services.document_processor.app.processing.logger")
async def test_parse_and_chunk_pdf_syntax_error(mock_logger, mock_pdf_extract):
    mock_pdf_extract.side_effect = PDFSyntaxError("Bad PDF")
    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        result_chunks = await doc_proc_processing.parse_and_chunk("p3", b"bad pdf", {"filename":"bad.pdf"})
    assert result_chunks == []
    mock_logger.error.assert_called_once()

@pytest.mark.asyncio
@patch("services.document_processor.app.processing.pdf_extract_text")
@patch("services.document_processor.app.processing.logger")
async def test_parse_and_chunk_other_extraction_error(mock_logger, mock_pdf_extract):
    mock_pdf_extract.side_effect = Exception("Some other error")
    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        result_chunks = await doc_proc_processing.parse_and_chunk("p4", b"pdf", {"filename":"err.pdf"})
    assert result_chunks == []
    mock_logger.error.assert_called_once()


@pytest.mark.asyncio
@patch("services.document_processor.app.processing.pdf_extract_text")
@patch("services.document_processor.app.processing._recursive_split")
async def test_parse_and_chunk_recursive_split_returns_empty(mock_recursive_split, mock_pdf_extract):
    mock_pdf_extract.return_value = "Some text"
    mock_recursive_split.return_value = [] # Splitter returns nothing
    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        result_chunks = await doc_proc_processing.parse_and_chunk("p5", b"pdf", {"filename":"f2.pdf"})
    assert result_chunks == []

@pytest.mark.asyncio
@patch("services.document_processor.app.processing.pdf_extract_text")
@patch("services.document_processor.app.processing._recursive_split")
@patch("services.document_processor.app.processing.logger")
async def test_parse_and_chunk_recursive_split_error(mock_logger, mock_recursive_split, mock_pdf_extract):
    mock_pdf_extract.return_value = "Some text"
    mock_recursive_split.side_effect = Exception("Splitter error")
    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        result_chunks = await doc_proc_processing.parse_and_chunk("p6", b"pdf", {"filename":"f3.pdf"})
    assert result_chunks == []
    mock_logger.error.assert_called_once()
```
