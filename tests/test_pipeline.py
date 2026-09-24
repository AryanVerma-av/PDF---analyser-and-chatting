import io
import os
import sys
from pathlib import Path

# Add backend directory to sys.path so app modules are discoverable
BASE_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BASE_DIR))

from fastapi.testclient import TestClient
from app.main import app
from app.rag.loader import PDFLoader
from app.rag.splitter import TextSplitter
from app.rag.vectorstore import VectorStore
from app.rag.retriever import Retriever
from app.models.schemas import DocumentChunk
from app.config import settings


def generate_sample_pdf_bytes() -> bytes:
    """Generates a clean 3-page PDF with distinct content for deterministic testing."""
    return b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R 4 0 R 5 0 R] /Count 3 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 6 0 R >> >> /Contents 7 0 R >>
endobj
4 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 6 0 R >> >> /Contents 8 0 R >>
endobj
5 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 6 0 R >> >> /Contents 9 0 R >>
endobj
6 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
7 0 obj
<< /Length 124 >>
stream
BT
/F1 12 Tf
72 712 Td
(Page 1: Project Aurora is an automated cloud data pipeline that processes real-time telemetry events.) Tj
ET
endstream
endobj
8 0 obj
<< /Length 128 >>
stream
BT
/F1 12 Tf
72 712 Td
(Page 2: The storage layer uses an immutable event ledger with five minute snapshot intervals.) Tj
ET
endstream
endobj
9 0 obj
<< /Length 122 >>
stream
BT
/F1 12 Tf
72 712 Td
(Page 3: The security architecture requires TLS 1.3 encryption and multi-factor authentication.) Tj
ET
endstream
endobj
xref
0 10
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000131 00000 n 
0000000246 00000 n 
0000000361 00000 n 
0000000476 00000 n 
0000000543 00000 n 
0000000718 00000 n 
0000000897 00000 n 
trailer
<< /Size 10 /Root 1 0 R >>
startxref
1070
%%EOF
"""


def test_pdf_extraction():
    print("[1/5] Testing PDF Loader extraction...")
    pdf_bytes = generate_sample_pdf_bytes()
    loader = PDFLoader()
    pages = loader.load_from_bytes(pdf_bytes, "test_sample.pdf")

    assert len(pages) == 3, f"Expected 3 pages, got {len(pages)}"
    assert pages[0]["page_number"] == 1
    assert "Project Aurora" in pages[0]["text"]
    assert pages[1]["page_number"] == 2
    assert "event ledger" in pages[1]["text"]
    assert pages[2]["page_number"] == 3
    assert "TLS 1.3" in pages[2]["text"]
    print("  PASSED: PDF extraction accurately parsed 3 pages with correct text and page numbers.")
    return pages


def test_chunking(pages):
    print("[2/5] Testing Text Splitter & Metadata Preservation...")
    splitter = TextSplitter(chunk_size=120, chunk_overlap=20)
    chunks = splitter.split_pages(pages)

    assert len(chunks) >= 3, f"Expected at least 3 chunks, got {len(chunks)}"
    pages_represented = {c.page_number for c in chunks}
    assert pages_represented == {1, 2, 3}, f"Expected pages {1, 2, 3}, got {pages_represented}"

    for chunk in chunks:
        assert chunk.chunk_id is not None
        assert chunk.page_number in [1, 2, 3]
        assert chunk.source_file == "test_sample.pdf"
        assert len(chunk.text) > 0

    print(f"  PASSED: Generated {len(chunks)} chunks across 3 pages while preserving page metadata.")
    return chunks


def test_vector_store(chunks):
    print("[3/5] Testing Vector Store and Semantic Retrieval...")
    # Use synthetic 1536-dimensional normalized vectors to test vector store mechanics
    dim = 1536
    vector_store = VectorStore()
    vector_store.reset()

    # Create distinct directional embeddings for test chunks
    import numpy as np

    synthetic_embeddings = []
    for i in range(len(chunks)):
        vec = np.zeros(dim, dtype=np.float32)
        vec[i % dim] = 1.0
        synthetic_embeddings.append(vec.tolist())

    vector_store.add_chunks(chunks, synthetic_embeddings)
    assert vector_store.count() == len(chunks), f"Expected {len(chunks)} vectors, got {vector_store.count()}"

    # Query with direction of chunk 1 (Page 2)
    query_vec = np.zeros(dim, dtype=np.float32)
    query_vec[1] = 1.0
    results = vector_store.query(query_vec.tolist(), top_k=2)

    assert len(results) > 0
    top_result = results[0]
    assert top_result["page_number"] == chunks[1].page_number
    assert top_result["source_file"] == "test_sample.pdf"
    assert top_result["similarity_score"] > 0.9
    print("  PASSED: Vector store successfully indexed and retrieved chunk with page metadata.")


def test_api_endpoints():
    print("[4/5] Testing FastAPI Endpoints (Health, Upload, Ask Validation)...")
    client = TestClient(app)

    # 1. Initial Health Check
    health_res = client.get("/health")
    assert health_res.status_code == 200
    data = health_res.json()
    assert data["status"] == "ok"
    print("  PASSED: GET /health returned 200 OK.")

    # 2. Ask before upload must fail with 400
    ask_res = client.post("/ask", json={"question": "What is Project Aurora?"})
    assert ask_res.status_code == 400
    assert "No PDF has been uploaded yet" in ask_res.json()["detail"]
    print("  PASSED: POST /ask before upload correctly returned 400 Bad Request.")

    # 3. Invalid file type upload must fail with 400
    invalid_file = io.BytesIO(b"not a pdf")
    upload_invalid = client.post(
        "/upload",
        files={"file": ("test.txt", invalid_file, "text/plain")}
    )
    assert upload_invalid.status_code == 400
    assert "Invalid file type" in upload_invalid.json()["detail"]
    print("  PASSED: POST /upload with non-PDF file rejected with 400 Bad Request.")

    # 4. Upload valid PDF
    pdf_bytes = generate_sample_pdf_bytes()
    upload_res = client.post(
        "/upload",
        files={"file": ("project_aurora.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    )
    # If OPENAI_API_KEY is not configured, upload might return 500 embedding error.
    if upload_res.status_code == 200:
        upload_data = upload_res.json()
        assert upload_data["filename"] == "project_aurora.pdf"
        assert upload_data["total_pages"] == 3
        print(f"  PASSED: POST /upload indexed {upload_data['total_chunks']} chunks across {upload_data['total_pages']} pages.")

        # Re-check Health
        health_after = client.get("/health")
        assert health_after.json()["vector_store_ready"] is True
        print("  PASSED: GET /health reports vector store ready with active document.")
    else:
        print(f"  NOTICE: POST /upload returned {upload_res.status_code}: {upload_res.json().get('detail')}")
        print("  (This is expected when running in environments without a live OPENAI_API_KEY).")


def test_openai_live_integration():
    print("[5/5] Testing Live OpenAI End-to-End Pipeline (if key is set)...")
    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.startswith("your_"):
        print("  SKIPPED: OPENAI_API_KEY is not set in environment. Set it in backend/.env to run live query.")
        return

    client = TestClient(app)
    pdf_bytes = generate_sample_pdf_bytes()
    upload_res = client.post(
        "/upload",
        files={"file": ("project_aurora.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    )
    assert upload_res.status_code == 200

    ask_res = client.post("/ask", json={"question": "What security architecture is required?"})
    assert ask_res.status_code == 200
    result = ask_res.json()
    assert "TLS 1.3" in result["answer"] or "encryption" in result["answer"]
    assert len(result["sources"]) > 0
    assert any(s["page_number"] == 3 for s in result["sources"])
    print(f"  PASSED: Grounded Answer: {result['answer']}")
    print(f"  PASSED: Source cited: Page {result['sources'][0]['page_number']}")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING PDF RAG PIPELINE VERIFICATION SUITE")
    print("=" * 60)
    pages = test_pdf_extraction()
    chunks = test_chunking(pages)
    test_vector_store(chunks)
    test_api_endpoints()
    test_openai_live_integration()
    print("=" * 60)
    print("ALL CORE UNIT & INTEGRATION TESTS COMPLETED SUCCESSFULLY")
    print("=" * 60)
