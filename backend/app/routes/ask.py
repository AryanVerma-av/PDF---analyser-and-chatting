from fastapi import APIRouter, HTTPException, Depends, status
from app.models.schemas import AskRequest, AskResponse
from app.services.state import get_pipeline
from app.rag.pipeline import RAGPipeline
from app.rag.embeddings import EmbeddingServiceError
from app.rag.vectorstore import VectorStoreError
from app.rag.generator import GeneratorError

router = APIRouter(tags=["Question Answering"])


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    payload: AskRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """
    Answers a question based strictly on the retrieved context from the uploaded PDF.
    """
    # Verify a PDF is uploaded and ready
    if not pipeline.is_ready():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No PDF has been uploaded yet. Please upload a PDF before asking a question.",
        )

    clean_question = payload.question.strip()
    if not clean_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty or whitespace.",
        )

    try:
        answer, sources, latency = pipeline.answer_question(clean_question)
    except EmbeddingServiceError as ese:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query embedding error: {str(ese)}",
        )
    except VectorStoreError as vse:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval error: {str(vse)}",
        )
    except GeneratorError as ge:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM generation error: {str(ge)}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate answer: {str(exc)}",
        )

    return AskResponse(
        question=clean_question,
        answer=answer,
        sources=sources,
        latency_seconds=latency,
    )
