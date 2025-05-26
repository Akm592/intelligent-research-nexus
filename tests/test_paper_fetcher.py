import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

# Assuming your app is in services.paper_fetcher.app.main
from services.paper_fetcher.app.main import app
from core.models import PaperMetadata, FetchRequest

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

# --- Tests for /fetch endpoint ---
@pytest.mark.asyncio
async def test_fetch_papers_success(client: TestClient):
    mock_paper1 = PaperMetadata(id="arxiv:1111.1111", title="Test Paper 1", source="arxiv", authors=["Auth1"], abstract="Abstract1")
    mock_paper2 = PaperMetadata(id="arxiv:2222.2222", title="Test Paper 2", source="arxiv", authors=["Auth2"], abstract="Abstract2")
    
    # Patching within the test function using 'with' ensures mocks are active only for this test
    # and are automatically cleaned up.
    with patch("services.paper_fetcher.app.main.logic.search_academic_sources", new_callable=AsyncMock) as mock_search_sources, \
         patch("services.paper_fetcher.app.main.crud.save_paper_metadata", new_callable=AsyncMock) as mock_save_metadata, \
         patch("services.paper_fetcher.app.main.get_supabase_client") as mock_get_db_client: # Mock DB client
        
        mock_db_instance = AsyncMock() # Mock instance for the DB client
        mock_get_db_client.return_value = mock_db_instance # get_supabase_client returns the mock
        
        mock_search_sources.return_value = [mock_paper1, mock_paper2]
        # crud.save_paper_metadata in the app doesn't have a specific return value check for success,
        # it's assumed to succeed if no exception is raised.
        mock_save_metadata.return_value = None 

        payload = FetchRequest(query="test query", sources=["arxiv"], max_results=2)
        response = client.post("/fetch", json=payload.model_dump())

        assert response.status_code == 200
        response_data = response.json()
        assert len(response_data) == 2
        # Ensure model_dump uses exclude_none=True if your model has optional fields with None
        assert response_data[0] == mock_paper1.model_dump(exclude_none=True) 
        assert response_data[1] == mock_paper2.model_dump(exclude_none=True)

        mock_search_sources.assert_called_once_with(db_client=mock_db_instance, query="test query", sources=["arxiv"], max_results=2)
        assert mock_save_metadata.call_count == 2
        mock_save_metadata.assert_any_call(db_client=mock_db_instance, paper=mock_paper1)
        mock_save_metadata.assert_any_call(db_client=mock_db_instance, paper=mock_paper2)

@pytest.mark.asyncio
async def test_fetch_papers_search_error(client: TestClient):
    with patch("services.paper_fetcher.app.main.logic.search_academic_sources", new_callable=AsyncMock) as mock_search_sources, \
         patch("services.paper_fetcher.app.main.get_supabase_client") as mock_get_db_client:
        
        mock_get_db_client.return_value = AsyncMock()
        mock_search_sources.side_effect = Exception("Simulated search error")
        
        payload = FetchRequest(query="test query", sources=["arxiv"], max_results=1)
        response = client.post("/fetch", json=payload.model_dump())
        
        assert response.status_code == 500
        assert "Failed to fetch papers: Simulated search error" in response.json()["detail"]

@pytest.mark.asyncio
async def test_fetch_papers_save_error(client: TestClient):
    mock_paper = PaperMetadata(id="arxiv:1234.5678", title="Test Paper", source="arxiv")
    with patch("services.paper_fetcher.app.main.logic.search_academic_sources", new_callable=AsyncMock) as mock_search_sources, \
         patch("services.paper_fetcher.app.main.crud.save_paper_metadata", new_callable=AsyncMock) as mock_save_metadata, \
         patch("services.paper_fetcher.app.main.get_supabase_client") as mock_get_db_client:
        
        mock_db_instance = AsyncMock()
        mock_get_db_client.return_value = mock_db_instance
        
        mock_search_sources.return_value = [mock_paper]
        mock_save_metadata.side_effect = Exception("Simulated save error")

        payload = FetchRequest(query="test query", sources=["arxiv"], max_results=1)
        response = client.post("/fetch", json=payload.model_dump())

        assert response.status_code == 500
        assert "Failed to fetch papers: Simulated save error" in response.json()["detail"]
        mock_search_sources.assert_called_once_with(db_client=mock_db_instance, query="test query", sources=["arxiv"], max_results=1)
        mock_save_metadata.assert_called_once_with(db_client=mock_db_instance, paper=mock_paper)


# --- Tests for /paper/{paper_id} endpoint ---
# The current main.py for paper_fetcher has a DUMMY implementation for GET /paper/{paper_id}.
# These tests reflect the DUMMY behavior.
# Tests for the intended CRUD logic (get_paper_metadata returning 404 or 500)
# are commented out but provided for when the endpoint is fixed.

@pytest.mark.asyncio
async def test_get_paper_metadata_dummy_behavior(client: TestClient):
    # This test reflects the current dummy implementation in services.paper_fetcher.app.main.get_paper_metadata
    paper_id_to_test = "arxiv:9999.9999"
    response = client.get(f"/paper/{paper_id_to_test}")
    
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["id"] == paper_id_to_test
    assert json_response["title"] == f"Dummy Title for {paper_id_to_test}" 
    assert json_response["source"] == "dummy_source" 
    assert json_response["authors"] == ["Dummy Author"] 

# To test the intended behavior (non-dummy), the endpoint would need to be updated.
# Example tests for a corrected endpoint:
# @pytest.mark.asyncio
# async def test_get_paper_metadata_success_intended(client: TestClient):
#     mock_paper = PaperMetadata(id="arxiv:1234.5678", title="Real Title", source="arxiv")
#     with patch("services.paper_fetcher.app.main.crud.get_paper_metadata", new_callable=AsyncMock) as mock_get_metadata, \
#          patch("services.paper_fetcher.app.main.get_supabase_client") as mock_get_db_client:
#         mock_get_db_client.return_value = AsyncMock()
#         mock_get_metadata.return_value = mock_paper
        
#         response = client.get("/paper/arxiv:1234.5678")
#         assert response.status_code == 200
#         assert response.json() == mock_paper.model_dump(exclude_none=True)
#         mock_get_metadata.assert_called_once_with(db_client=mock_get_db_client.return_value, paper_id="arxiv:1234.5678")

# @pytest.mark.asyncio
# async def test_get_paper_metadata_not_found_intended(client: TestClient):
#     with patch("services.paper_fetcher.app.main.crud.get_paper_metadata", new_callable=AsyncMock) as mock_get_metadata, \
#          patch("services.paper_fetcher.app.main.get_supabase_client") as mock_get_db_client:
#         mock_get_db_client.return_value = AsyncMock()
#         mock_get_metadata.return_value = None # Simulate paper not found
        
#         response = client.get("/paper/nonexistent:id")
#         assert response.status_code == 404
#         assert "Paper metadata not found" in response.json()["detail"]

# @pytest.mark.asyncio
# async def test_get_paper_metadata_crud_error_intended(client: TestClient):
#     with patch("services.paper_fetcher.app.main.crud.get_paper_metadata", new_callable=AsyncMock) as mock_get_metadata, \
#          patch("services.paper_fetcher.app.main.get_supabase_client") as mock_get_db_client:
#         mock_get_db_client.return_value = AsyncMock()
#         mock_get_metadata.side_effect = Exception("Simulated DB error")
        
#         response = client.get("/paper/any:id")
#         assert response.status_code == 500
#         assert "Failed to retrieve paper metadata: Simulated DB error" in response.json()["detail"]


# --- Tests for /papers_by_ids endpoint (POST) ---
@pytest.mark.asyncio
async def test_get_papers_by_ids_success(client: TestClient):
    mock_paper1 = PaperMetadata(id="arxiv:1111.1111", title="Test Paper 1", source="arxiv")
    mock_paper2 = PaperMetadata(id="arxiv:2222.2222", title="Test Paper 2", source="arxiv")
    
    with patch("services.paper_fetcher.app.main.crud.get_papers_by_ids", new_callable=AsyncMock) as mock_get_by_ids, \
         patch("services.paper_fetcher.app.main.get_supabase_client") as mock_get_db_client:
        
        mock_db_instance = AsyncMock()
        mock_get_db_client.return_value = mock_db_instance
        mock_get_by_ids.return_value = [mock_paper1, mock_paper2]
        
        payload = ["arxiv:1111.1111", "arxiv:2222.2222"]
        response = client.post("/papers_by_ids", json=payload)
        
        assert response.status_code == 200
        response_data = response.json()
        assert len(response_data) == 2
        assert response_data[0] == mock_paper1.model_dump(exclude_none=True)
        assert response_data[1] == mock_paper2.model_dump(exclude_none=True)
        mock_get_by_ids.assert_called_once_with(db_client=mock_db_instance, paper_ids=payload)

@pytest.mark.asyncio
async def test_get_papers_by_ids_some_not_found(client: TestClient):
    mock_paper1 = PaperMetadata(id="arxiv:1111.1111", title="Test Paper 1", source="arxiv")
    with patch("services.paper_fetcher.app.main.crud.get_papers_by_ids", new_callable=AsyncMock) as mock_get_by_ids, \
         patch("services.paper_fetcher.app.main.get_supabase_client") as mock_get_db_client:
        
        mock_db_instance = AsyncMock()
        mock_get_db_client.return_value = mock_db_instance
        mock_get_by_ids.return_value = [mock_paper1] 
        
        payload = ["arxiv:1111.1111", "arxiv:nonexistent"]
        response = client.post("/papers_by_ids", json=payload)
        
        assert response.status_code == 200
        response_data = response.json()
        assert len(response_data) == 1
        assert response_data[0] == mock_paper1.model_dump(exclude_none=True)
        mock_get_by_ids.assert_called_once_with(db_client=mock_db_instance, paper_ids=payload)

@pytest.mark.asyncio
async def test_get_papers_by_ids_empty_input(client: TestClient):
    with patch("services.paper_fetcher.app.main.crud.get_papers_by_ids", new_callable=AsyncMock) as mock_get_by_ids, \
         patch("services.paper_fetcher.app.main.get_supabase_client") as mock_get_db_client:
        
        mock_db_instance = AsyncMock()
        mock_get_db_client.return_value = mock_db_instance
        mock_get_by_ids.return_value = []
        
        payload = []
        response = client.post("/papers_by_ids", json=payload)
        
        assert response.status_code == 200
        assert response.json() == []
        mock_get_by_ids.assert_called_once_with(db_client=mock_db_instance, paper_ids=[])

@pytest.mark.asyncio
async def test_get_papers_by_ids_crud_error(client: TestClient):
    with patch("services.paper_fetcher.app.main.crud.get_papers_by_ids", new_callable=AsyncMock) as mock_get_by_ids, \
         patch("services.paper_fetcher.app.main.get_supabase_client") as mock_get_db_client:
        
        mock_db_instance = AsyncMock()
        mock_get_db_client.return_value = mock_db_instance
        mock_get_by_ids.side_effect = Exception("Simulated DB error")
        
        payload = ["arxiv:1111.1111"]
        response = client.post("/papers_by_ids", json=payload)
        
        assert response.status_code == 500
        assert "Failed to retrieve papers by IDs: Simulated DB error" in response.json()["detail"]
        mock_get_by_ids.assert_called_once_with(db_client=mock_db_instance, paper_ids=payload)


# --- Tests for logic.py ---
from services.paper_fetcher.app import logic as paper_fetcher_logic
from services.paper_fetcher.app import crud as paper_fetcher_crud # For mocking get_supabase_client in crud tests
import arxiv # For mocking arxiv.Result
from datetime import datetime, timezone
import uuid
from postgrest.exceptions import APIError as PostgrestAPIError # Correct import for Supabase errors


# --- Tests for logic.parse_arxiv_id ---
@pytest.mark.parametrize("entry_id, expected_prefix, is_known", [
    ("http://arxiv.org/abs/2305.12345v1", "arxiv:2305.12345v1", True),
    ("https://arxiv.org/pdf/cond-mat/0123456.pdf", "arxiv:cond-mat/0123456", True),
    ("abs/1234.5678", "arxiv:1234.5678", True),
    ("0706.1234", "arxiv:0706.1234", True),
    ("arxiv:hep-th/0101001v2", "arxiv:hep-th/0101001v2", True),
    ("http://arxiv.org/abs/math.GT/0309136", "arxiv:math.GT/0309136", True),
    ("InvalidID", "arxiv:unknown_", False), # Expect unknown_ + uuid
    ("http://example.com/abs/2305.12345", "arxiv:unknown_", False),
    ("1234.5678.pdf", "arxiv:1234.5678.pdf", True), # Handles .pdf in ID part
])
def test_parse_arxiv_id(entry_id, expected_prefix, is_known):
    parsed_id = paper_fetcher_logic._parse_arxiv_id(entry_id)
    if is_known:
        assert parsed_id == expected_prefix
    else:
        assert parsed_id.startswith(expected_prefix)
        # Check if the part after "unknown_" is a valid UUID
        try:
            uuid.UUID(parsed_id.split('_')[-1])
            assert True
        except ValueError:
            assert False, "Suffix is not a valid UUID"

# --- Tests for logic.search_arxiv ---
@pytest.mark.asyncio
async def test_search_arxiv_success():
    mock_arxiv_result1 = AsyncMock(spec=arxiv.Result)
    mock_arxiv_result1.entry_id = "http://arxiv.org/abs/2301.00001v1"
    mock_arxiv_result1.pdf_url = "http://arxiv.org/pdf/2301.00001v1.pdf"
    mock_arxiv_result1.title = "Title One"
    mock_arxiv_result1.summary = "Summary One"
    mock_arxiv_result1.authors = [AsyncMock(name="Author A"), AsyncMock(name="Author B")]
    mock_arxiv_result1.published = datetime(2023, 1, 1, tzinfo=timezone.utc)
    mock_arxiv_result1.categories = ["cs.AI", "cs.LG"]

    mock_arxiv_result2 = AsyncMock(spec=arxiv.Result) # Missing pdf_url and summary
    mock_arxiv_result2.entry_id = "http://arxiv.org/abs/2301.00002"
    mock_arxiv_result2.pdf_url = None
    mock_arxiv_result2.title = "Title Two"
    mock_arxiv_result2.summary = None
    mock_arxiv_result2.authors = [AsyncMock(name="Author C")]
    mock_arxiv_result2.published = datetime(2023, 1, 2, tzinfo=timezone.utc)
    mock_arxiv_result2.categories = ["math.CO"]
    
    mock_arxiv_result_unknown_id = AsyncMock(spec=arxiv.Result) # Will be skipped
    mock_arxiv_result_unknown_id.entry_id = "invalid_id_format" 
    mock_arxiv_result_unknown_id.title = "Should be skipped"

    mock_search_instance = AsyncMock()
    mock_search_instance.results.return_value = [mock_arxiv_result1, mock_arxiv_result2, mock_arxiv_result_unknown_id]

    with patch("arxiv.Search", return_value=mock_search_instance) as mock_arxiv_search_class:
        results = await paper_fetcher_logic.search_arxiv(None, "test query", max_results=5)

        mock_arxiv_search_class.assert_called_once_with(query="test query", max_results=5, sort_by=arxiv.SortCriterion.Relevance)
        assert len(results) == 2 # mock_arxiv_result_unknown_id should be skipped

        # Paper 1 assertions
        assert results[0].id == "arxiv:2301.00001v1"
        assert results[0].title == "Title One"
        assert results[0].authors == ["Author A", "Author B"]
        assert results[0].abstract == "Summary One"
        assert results[0].url == "http://arxiv.org/abs/2301.00001v1" # entry_id is preferred for URL
        assert results[0].pdf_url == "http://arxiv.org/pdf/2301.00001v1.pdf"
        assert results[0].source == "arxiv"
        assert results[0].published_date == "2023-01-01"
        assert results[0].keywords == ["cs.AI", "cs.LG"]

        # Paper 2 assertions
        assert results[1].id == "arxiv:2301.00002"
        assert results[1].title == "Title Two"
        assert results[1].authors == ["Author C"]
        assert results[1].abstract is None # Summary was None
        assert results[1].url == "http://arxiv.org/abs/2301.00002"
        assert results[1].pdf_url is None # pdf_url was None
        assert results[1].source == "arxiv"
        assert results[1].published_date == "2023-01-02"
        assert results[1].keywords == ["math.CO"]

@pytest.mark.asyncio
async def test_search_arxiv_api_error():
    mock_search_instance = AsyncMock()
    mock_search_instance.results.side_effect = Exception("ArXiv API error")

    with patch("arxiv.Search", return_value=mock_search_instance):
        with pytest.raises(Exception, match="ArXiv API error"):
            await paper_fetcher_logic.search_arxiv(None, "test query", max_results=1)

# --- Tests for logic.search_semantic_scholar ---
@pytest.mark.asyncio
async def test_search_semantic_scholar_placeholder():
    # Test current placeholder behavior
    results = await paper_fetcher_logic.search_semantic_scholar(None, "any query", max_results=2)
    assert len(results) == 2
    assert results[0].id.startswith("s2:dummy_")
    assert results[0].source == "semantic_scholar"
    assert results[0].title.startswith("Dummy Semantic Scholar Paper")
    assert results[1].id.startswith("s2:dummy_")
    assert len(await paper_fetcher_logic.search_semantic_scholar(None, "q", max_results=0)) == 0
    assert len(await paper_fetcher_logic.search_semantic_scholar(None, "q", max_results=5)) == 5


# --- Tests for logic.search_academic_sources ---
@pytest.mark.asyncio
async def test_search_academic_sources_all_sources():
    mock_arxiv_paper = PaperMetadata(id="arxiv:001", title="Arxiv Paper", source="arxiv")
    mock_s2_paper = PaperMetadata(id="s2:001", title="S2 Paper", source="semantic_scholar")

    with patch("services.paper_fetcher.app.logic.search_arxiv", new_callable=AsyncMock) as mock_search_arxiv, \
         patch("services.paper_fetcher.app.logic.search_semantic_scholar", new_callable=AsyncMock) as mock_search_s2:
        
        mock_search_arxiv.return_value = [mock_arxiv_paper]
        mock_search_s2.return_value = [mock_s2_paper]

        results = await paper_fetcher_logic.search_academic_sources(None, "query", ["arxiv", "semantic_scholar"], max_results=5)
        assert len(results) == 2
        assert mock_arxiv_paper in results
        assert mock_s2_paper in results
        mock_search_arxiv.assert_called_once()
        mock_search_s2.assert_called_once()

@pytest.mark.asyncio
async def test_search_academic_sources_one_source():
    mock_arxiv_paper = PaperMetadata(id="arxiv:001", title="Arxiv Paper", source="arxiv")
    with patch("services.paper_fetcher.app.logic.search_arxiv", new_callable=AsyncMock) as mock_search_arxiv, \
         patch("services.paper_fetcher.app.logic.search_semantic_scholar", new_callable=AsyncMock) as mock_search_s2:
        
        mock_search_arxiv.return_value = [mock_arxiv_paper]
        results = await paper_fetcher_logic.search_academic_sources(None, "query", ["arxiv"], max_results=5)
        
        assert len(results) == 1
        assert mock_arxiv_paper in results
        mock_search_arxiv.assert_called_once()
        mock_search_s2.assert_not_called()

@pytest.mark.asyncio
async def test_search_academic_sources_deduplication_and_max_results():
    # Semantic Scholar might return papers also found on arXiv
    mock_arxiv_paper1 = PaperMetadata(id="arxiv:001", title="Arxiv Paper 1", source="arxiv")
    mock_arxiv_paper2 = PaperMetadata(id="arxiv:002", title="Arxiv Paper 2", source="arxiv")
    # s2_paper1 is a duplicate of arxiv_paper1 (same content, different ID prefix)
    mock_s2_paper1_dup = PaperMetadata(id="s2:dup_of_arxiv001_by_title", title="Arxiv Paper 1", source="semantic_scholar", authors=["Test Author"]) 
    mock_s2_paper2 = PaperMetadata(id="s2:002", title="S2 Paper 2", source="semantic_scholar")

    with patch("services.paper_fetcher.app.logic.search_arxiv", new_callable=AsyncMock) as mock_search_arxiv, \
         patch("services.paper_fetcher.app.logic.search_semantic_scholar", new_callable=AsyncMock) as mock_search_s2:
        
        mock_search_arxiv.return_value = [mock_arxiv_paper1, mock_arxiv_paper2]
        mock_search_s2.return_value = [mock_s2_paper1_dup, mock_s2_paper2] # S2 returns its own results

        # Max results = 3, should get arxiv:001, arxiv:002, s2:002 (s2:dup_of_arxiv001 is removed)
        results = await paper_fetcher_logic.search_academic_sources(None, "query", ["arxiv", "semantic_scholar"], max_results=3)
        
        assert len(results) == 3
        result_ids = {p.id for p in results}
        titles = {p.title for p in results}

        assert "arxiv:001" in result_ids
        assert "arxiv:002" in result_ids
        assert "s2:002" in result_ids 
        assert "Arxiv Paper 1" in titles # Check that the original title is preserved
        assert "Arxiv Paper 2" in titles
        assert "S2 Paper 2" in titles
        
        # Test max_results limits combined list
        results_limited = await paper_fetcher_logic.search_academic_sources(None, "query", ["arxiv", "semantic_scholar"], max_results=1)
        assert len(results_limited) == 1

@pytest.mark.asyncio
async def test_search_academic_sources_source_error_handling():
    mock_arxiv_paper = PaperMetadata(id="arxiv:001", title="Arxiv Paper", source="arxiv")
    with patch("services.paper_fetcher.app.logic.search_arxiv", new_callable=AsyncMock) as mock_search_arxiv, \
         patch("services.paper_fetcher.app.logic.search_semantic_scholar", new_callable=AsyncMock) as mock_search_s2:
        
        mock_search_arxiv.return_value = [mock_arxiv_paper]
        mock_search_s2.side_effect = Exception("S2 API is down")

        # Should still return results from arXiv
        results = await paper_fetcher_logic.search_academic_sources(None, "query", ["arxiv", "semantic_scholar"], max_results=5)
        assert len(results) == 1
        assert mock_arxiv_paper in results
        mock_search_arxiv.assert_called_once()
        mock_search_s2.assert_called_once() # Error is caught internally

# --- Tests for crud.py ---

# Mock Supabase client for CRUD operations
@pytest_asyncio.fixture
async def mock_db_client():
    mock_client = AsyncMock()
    
    # Fluent API mocks
    mock_table_response = AsyncMock()
    mock_select_response = AsyncMock()
    mock_upsert_response = AsyncMock()
    mock_eq_response = AsyncMock()
    mock_in_response = AsyncMock()
    mock_limit_response = AsyncMock()
    mock_maybe_single_response = AsyncMock()
    
    mock_client.table.return_value = mock_table_response
    mock_table_response.select.return_value = mock_select_response
    mock_table_response.upsert.return_value = mock_upsert_response
    
    # These need to return self to be chainable before execute
    mock_select_response.eq.return_value = mock_eq_response 
    mock_select_response.in_.return_value = mock_in_response
    mock_select_response.limit.return_value = mock_limit_response
    mock_eq_response.limit.return_value = mock_limit_response # For get_paper_metadata
    mock_eq_response.maybe_single.return_value = mock_maybe_single_response # For get_paper_metadata with maybe_single

    # Final execute() calls
    # We will set execute on these chained mocks per test
    mock_upsert_response.execute = AsyncMock() 
    mock_select_response.execute = AsyncMock() # For get_papers_by_ids without filters
    mock_eq_response.execute = AsyncMock() # For get_paper_metadata without maybe_single
    mock_in_response.execute = AsyncMock() # For get_papers_by_ids with in_ filter
    mock_maybe_single_response.execute = AsyncMock() # For get_paper_metadata with maybe_single

    return mock_client

# --- Tests for crud.save_paper_metadata ---
@pytest.mark.asyncio
async def test_save_paper_metadata_success(mock_db_client: AsyncMock):
    paper = PaperMetadata(id="test:001", title="Test", source="test")
    
    # Mock the successful execution of upsert
    # Supabase client returns an APIResponse-like object with a 'data' attribute
    mock_api_response = AsyncMock()
    mock_api_response.data = [{"id": "test:001", "title": "Test"}] # Example data
    
    # The mock for execute() is on the object returned by upsert()
    mock_db_client.table.return_value.upsert.return_value.execute.return_value = mock_api_response
    
    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)): # Bypass to_thread
        result = await paper_fetcher_crud.save_paper_metadata(mock_db_client, paper)

    assert result is True
    mock_db_client.table.assert_called_once_with("papers")
    # Use model_dump(exclude_none=True) to match how data is likely sent
    mock_db_client.table.return_value.upsert.assert_called_once_with(paper.model_dump(exclude_none=True))

@pytest.mark.asyncio
async def test_save_paper_metadata_api_error(mock_db_client: AsyncMock):
    paper = PaperMetadata(id="test:002", title="Test API Error", source="test")
    # Simulate PostgrestAPIError
    mock_db_client.table.return_value.upsert.return_value.execute.side_effect = PostgrestAPIError({"message": "DB error"}, response=None, request=None)

    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        with pytest.raises(PostgrestAPIError): # Expect the error to be re-raised
            await paper_fetcher_crud.save_paper_metadata(mock_db_client, paper)

@pytest.mark.asyncio
async def test_save_paper_metadata_other_exception(mock_db_client: AsyncMock):
    paper = PaperMetadata(id="test:003", title="Test Other Exception", source="test")
    mock_db_client.table.return_value.upsert.return_value.execute.side_effect = ValueError("Something else broke")

    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        with pytest.raises(ValueError, match="Something else broke"):
            await paper_fetcher_crud.save_paper_metadata(mock_db_client, paper)

@pytest.mark.asyncio
async def test_save_paper_metadata_no_data_returned(mock_db_client: AsyncMock):
    paper = PaperMetadata(id="test:004", title="Test No Data", source="test")
    mock_api_response = AsyncMock()
    mock_api_response.data = [] # Or None
    mock_db_client.table.return_value.upsert.return_value.execute.return_value = mock_api_response
    
    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        result = await paper_fetcher_crud.save_paper_metadata(mock_db_client, paper)
    assert result is False # Should indicate failure if no data/confirmation from DB

# --- Tests for crud.get_paper_metadata ---
@pytest.mark.asyncio
async def test_get_paper_metadata_success(mock_db_client: AsyncMock):
    paper_id = "test:001"
    db_data = {"id": paper_id, "title": "Found Paper", "source": "db", "authors": ["Auth X"], "abstract": "Abstract X"}
    mock_api_response = AsyncMock()
    mock_api_response.data = db_data
    
    # Mock for .select().eq().limit().maybe_single().execute()
    mock_db_client.table.return_value.select.return_value.eq.return_value.limit.return_value.maybe_single.return_value.execute.return_value = mock_api_response

    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        result = await paper_fetcher_crud.get_paper_metadata(mock_db_client, paper_id)

    assert isinstance(result, PaperMetadata)
    assert result.id == paper_id
    assert result.title == "Found Paper"
    mock_db_client.table.assert_called_once_with("papers")
    mock_db_client.table.return_value.select.assert_called_once_with("*")
    mock_db_client.table.return_value.select.return_value.eq.assert_called_once_with("id", paper_id)
    mock_db_client.table.return_value.select.return_value.eq.return_value.limit.assert_called_once_with(1)
    mock_db_client.table.return_value.select.return_value.eq.return_value.limit.return_value.maybe_single.assert_called_once()

@pytest.mark.asyncio
async def test_get_paper_metadata_not_found(mock_db_client: AsyncMock):
    paper_id = "test:nonexistent"
    mock_api_response = AsyncMock()
    mock_api_response.data = None # Simulate not found
    mock_db_client.table.return_value.select.return_value.eq.return_value.limit.return_value.maybe_single.return_value.execute.return_value = mock_api_response
    
    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        result = await paper_fetcher_crud.get_paper_metadata(mock_db_client, paper_id)
    assert result is None

@pytest.mark.asyncio
async def test_get_paper_metadata_api_error(mock_db_client: AsyncMock):
    paper_id = "test:dberror"
    mock_db_client.table.return_value.select.return_value.eq.return_value.limit.return_value.maybe_single.return_value.execute.side_effect = PostgrestAPIError({"message": "DB error"}, response=None, request=None)
    
    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        with pytest.raises(PostgrestAPIError):
            await paper_fetcher_crud.get_paper_metadata(mock_db_client, paper_id)

@pytest.mark.asyncio
async def test_get_paper_metadata_pydantic_error(mock_db_client: AsyncMock):
    paper_id = "test:bad_data"
    db_data = {"id": paper_id, "title": "Bad Data", "source": "db", "authors": "Not a list"} # Invalid authors
    mock_api_response = AsyncMock()
    mock_api_response.data = db_data
    mock_db_client.table.return_value.select.return_value.eq.return_value.limit.return_value.maybe_single.return_value.execute.return_value = mock_api_response

    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        with pytest.raises(Exception): # Pydantic's ValidationError or similar
             await paper_fetcher_crud.get_paper_metadata(mock_db_client, paper_id)


# --- Tests for crud.get_papers_by_ids ---
@pytest.mark.asyncio
async def test_get_papers_by_ids_success_multiple(mock_db_client: AsyncMock):
    paper_ids = ["id1", "id2"]
    db_data = [
        {"id": "id1", "title": "Paper 1", "source": "db"},
        {"id": "id2", "title": "Paper 2", "source": "db"}
    ]
    mock_api_response = AsyncMock()
    mock_api_response.data = db_data
    # Mock for .select().in_().execute()
    mock_db_client.table.return_value.select.return_value.in_.return_value.execute.return_value = mock_api_response

    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        results = await paper_fetcher_crud.get_papers_by_ids(mock_db_client, paper_ids)

    assert len(results) == 2
    assert results[0].id == "id1"
    assert results[1].id == "id2"
    mock_db_client.table.assert_called_once_with("papers")
    mock_db_client.table.return_value.select.assert_called_once_with("*")
    mock_db_client.table.return_value.select.return_value.in_.assert_called_once_with("id", paper_ids)

@pytest.mark.asyncio
async def test_get_papers_by_ids_empty_input_list(mock_db_client: AsyncMock):
    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        results = await paper_fetcher_crud.get_papers_by_ids(mock_db_client, [])
    assert results == []
    mock_db_client.table.return_value.select.return_value.in_.assert_not_called() # Should not query DB for empty list

@pytest.mark.asyncio
async def test_get_papers_by_ids_some_found(mock_db_client: AsyncMock):
    paper_ids = ["id1", "nonexistent_id"]
    db_data = [{"id": "id1", "title": "Paper 1", "source": "db"}] # Only id1 found
    mock_api_response = AsyncMock()
    mock_api_response.data = db_data
    mock_db_client.table.return_value.select.return_value.in_.return_value.execute.return_value = mock_api_response
    
    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        results = await paper_fetcher_crud.get_papers_by_ids(mock_db_client, paper_ids)
    
    assert len(results) == 1
    assert results[0].id == "id1"

@pytest.mark.asyncio
async def test_get_papers_by_ids_none_found(mock_db_client: AsyncMock):
    paper_ids = ["nonexistent1", "nonexistent2"]
    mock_api_response = AsyncMock()
    mock_api_response.data = [] # None found
    mock_db_client.table.return_value.select.return_value.in_.return_value.execute.return_value = mock_api_response
    
    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        results = await paper_fetcher_crud.get_papers_by_ids(mock_db_client, paper_ids)
    assert results == []

@pytest.mark.asyncio
async def test_get_papers_by_ids_api_error(mock_db_client: AsyncMock):
    paper_ids = ["id1"]
    mock_db_client.table.return_value.select.return_value.in_.return_value.execute.side_effect = PostgrestAPIError({"message": "DB error"}, response=None, request=None)
    
    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        with pytest.raises(PostgrestAPIError):
            await paper_fetcher_crud.get_papers_by_ids(mock_db_client, paper_ids)

@pytest.mark.asyncio
async def test_get_papers_by_ids_pydantic_error_item(mock_db_client: AsyncMock):
    paper_ids = ["id1", "id_bad_data"]
    db_data = [
        {"id": "id1", "title": "Good Paper", "source": "db"},
        {"id": "id_bad_data", "title": "Bad Data Paper", "authors": "not-a-list"} # Invalid authors
    ]
    mock_api_response = AsyncMock()
    mock_api_response.data = db_data
    mock_db_client.table.return_value.select.return_value.in_.return_value.execute.return_value = mock_api_response

    with patch("asyncio.to_thread", side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        # Depending on how errors are handled, this might raise an error
        # or skip the bad item. Current crud.py skips bad items.
        results = await paper_fetcher_crud.get_papers_by_ids(mock_db_client, paper_ids)
        assert len(results) == 1 # Only the valid paper should be returned
        assert results[0].id == "id1"
```
