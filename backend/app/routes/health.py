from fastapi import APIRouter, Depends
from app.models.schemas import HealthResponse
from app.services.state import get_pipeline
from app.rag.pipeline import RAGPipeline

from app.config import settings

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def check_health(pipeline: RAGPipeline = Depends(get_pipeline)):
    """
    Returns server status, current indexed PDF info, and vector store readiness.
    """
    has_gemini = bool(settings.GEMINI_API_KEY and not settings.GEMINI_API_KEY.startswith("your_"))
    has_groq = bool(settings.GROQ_API_KEY and not settings.GROQ_API_KEY.startswith("your_"))
    has_openai = bool(settings.OPENAI_API_KEY and not settings.OPENAI_API_KEY.startswith("your_"))

    return HealthResponse(
        status="ok",
        active_document=pipeline.active_filename,
        total_pages=pipeline.total_pages,
        total_chunks=pipeline.total_chunks,
        vector_store_ready=pipeline.is_ready(),
        has_keys=bool(has_gemini or has_groq or has_openai),
        has_gemini_key=has_gemini,
        has_groq_key=has_groq,
        has_openai_key=has_openai,
        active_provider=settings.ACTIVE_PROVIDER,
    )
