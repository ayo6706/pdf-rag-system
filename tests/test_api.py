import pytest
import io
from pydantic import SecretStr

from tests.conftest import test_async_session_maker
from app.core.config import llm_settings

@pytest.mark.asyncio
async def test_health_check(async_client):
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_readiness_check(async_client, monkeypatch):
    monkeypatch.setattr(llm_settings, "google_api_key", SecretStr("test-key"))

    response = await async_client.get("/api/v1/health/ready")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["db"]["status"] == "ok"
    assert data["vector_store"]["status"] == "ok"
    assert data["llm"]["status"] == "ok"

@pytest.mark.asyncio
async def test_upload_valid_pdf(async_client, tmp_upload_dir):
    # Create a dummy PDF content
    pdf_content = b"%PDF-1.4 dummy content"
    files = {"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")}
    
    response = await async_client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "test.pdf"
    assert data["status"] == "pending"
    assert "id" in data

@pytest.mark.asyncio
async def test_upload_invalid_file_type(async_client, tmp_upload_dir):
    txt_content = b"Not a PDF"
    files = {"file": ("test.txt", io.BytesIO(txt_content), "text/plain")}
    
    response = await async_client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 422
    assert response.json()["detail"]["error_code"] == "INVALID_FILE_TYPE"

@pytest.mark.asyncio
async def test_upload_file_too_large(async_client, tmp_upload_dir, monkeypatch):
    import app.core.config
    monkeypatch.setattr(app.core.config.infra_settings, "max_upload_size_mb", 0)
    
    pdf_content = b"%PDF-1.4 large content"
    files = {"file": ("large.pdf", io.BytesIO(pdf_content), "application/pdf")}
    
    response = await async_client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 413
    assert response.json()["detail"]["error_code"] == "FILE_TOO_LARGE"

@pytest.mark.asyncio
async def test_get_documents(async_client, tmp_upload_dir):
    # Upload one first
    pdf_content = b"%PDF-1.4"
    files = {"file": ("test2.pdf", io.BytesIO(pdf_content), "application/pdf")}
    upload_resp = await async_client.post("/api/v1/documents/upload", files=files)
    assert upload_resp.status_code == 201, f"Upload failed: {upload_resp.text}"
    doc_id = upload_resp.json()["id"]

    # Get all
    response = await async_client.get("/api/v1/documents")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1
    assert any(d["id"] == doc_id for d in data["documents"])

@pytest.mark.asyncio
async def test_get_document_by_id(async_client, tmp_upload_dir):
    # Upload one
    pdf_content = b"%PDF-1.4"
    files = {"file": ("test3.pdf", io.BytesIO(pdf_content), "application/pdf")}
    upload_resp = await async_client.post("/api/v1/documents/upload", files=files)
    assert upload_resp.status_code == 201, f"Upload failed: {upload_resp.text}"
    doc_id = upload_resp.json()["id"]

    # Get specific
    response = await async_client.get(f"/api/v1/documents/{doc_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == doc_id
    assert data["filename"] == "test3.pdf"

@pytest.mark.asyncio
async def test_delete_document(async_client, tmp_upload_dir):
    # Upload one
    pdf_content = b"%PDF-1.4"
    files = {"file": ("test4.pdf", io.BytesIO(pdf_content), "application/pdf")}
    upload_resp = await async_client.post("/api/v1/documents/upload", files=files)
    assert upload_resp.status_code == 201, f"Upload failed: {upload_resp.text}"
    doc_id = upload_resp.json()["id"]

    # Delete
    response = await async_client.delete(f"/api/v1/documents/{doc_id}")
    assert response.status_code == 204

    # Verify not found
    response = await async_client.get(f"/api/v1/documents/{doc_id}")
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_crash_recovery():
    """Test that recover_stuck_documents resets PROCESSING documents to PENDING."""
    from app.models.document import Document, DocumentStatus
    from sqlalchemy import select
    from app.services.document import document_service
    import app.main
    
    # Create a document stuck in PROCESSING
    async with test_async_session_maker() as session:
        doc = Document(filename="stuck.pdf", storage_filename="stuck.pdf", status=DocumentStatus.PROCESSING)
        session.add(doc)
        await session.commit()
        doc_id = doc.id
        
    # Run the extracted recovery function
    await document_service.recover_stuck_documents()
        
    # Verify it was reset
    async with test_async_session_maker() as session:
        stmt = select(Document).where(Document.id == doc_id)
        result = await session.execute(stmt)
        updated_doc = result.scalar_one_or_none()
        assert updated_doc is not None
        assert updated_doc.status == DocumentStatus.PENDING
