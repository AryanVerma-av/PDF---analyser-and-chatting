from fastapi import APIRouter, Depends
from app.models.schemas import HealthResponse
from app.services.state import get_pipeline
from app.rag.pipeline import RAGPipeline

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def check_health(pipeline: RAGPipeline = Depends(get_pipeline)):
    """
    Returns server status, current indexed PDF info, and vector store readiness.
    """
    return HealthResponse(
        status="ok",
        active_document=pipeline.active_filename,
        total_pages=pipeline.total_pages,
        total_chunks=pipeline.total_chunks,
        vector_store_ready=pipeline.is_ready(),
    )
