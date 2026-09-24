from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents an extracted and chunked snippet of text with metadata."""
    chunk_id: str
    text: str
    page_number: int
    source_file: str


class SourceCitation(BaseModel):
    """Source reference for an answer citing specific PDF pages."""
    page_number: int
    excerpt: str
    relevance_score: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = "ok"
    active_document: Optional[str] = None
    total_pages: int = 0
    total_chunks: int = 0
    vector_store_ready: bool = False


class UploadResponse(BaseModel):
    """Response returned upon successful PDF ingestion."""
    message: str
    filename: str
    total_pages: int
    total_chunks: int
    processing_time_seconds: float


class AskRequest(BaseModel):
    """Request schema for asking a question against the uploaded PDF."""
    question: str = Field(..., min_length=1, description="Question text to query against the PDF")


class AskResponse(BaseModel):
    """Response returned containing the grounded answer and source citations."""
    question: str
    answer: str
    sources: List[SourceCitation]
    latency_seconds: float
