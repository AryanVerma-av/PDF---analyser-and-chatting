from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, status
from app.models.schemas import UploadResponse
from app.services.state import get_pipeline
from app.rag.pipeline import RAGPipeline
from app.rag.loader import PDFLoaderError
from app.rag.embeddings import EmbeddingServiceError
from app.rag.vectorstore import VectorStoreError

router = APIRouter(tags=["PDF Ingestion"])

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """
    Upload and process a PDF document through the RAG pipeline.
    Extracts text, creates chunks, computes embeddings, and stores them in the vector index.
    """
    filename = file.filename or "uploaded_document.pdf"

    # Validate file extension
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PDF documents (.pdf) are supported.",
        )

    # Read file content into memory
    try:
        content = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(exc)}",
        )

    # Validate file size
    if not content or len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is completely empty.",
        )

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File size exceeds maximum allowed limit (25 MB).",
        )

    # Ingest through the RAG pipeline
    try:
        result = pipeline.ingest_pdf(content, filename)
    except PDFLoaderError as ple:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"PDF extraction error: {str(ple)}",
        )
    except EmbeddingServiceError as ese:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation error: {str(ese)}",
        )
    except VectorStoreError as vse:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector store error: {str(vse)}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing failed: {str(exc)}",
        )

    return UploadResponse(
        message="PDF processed and indexed successfully.",
        filename=result["filename"],
        total_pages=result["total_pages"],
        total_chunks=result["total_chunks"],
        processing_time_seconds=result["processing_time_seconds"],
    )
