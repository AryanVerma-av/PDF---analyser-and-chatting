from typing import List, Tuple
from app.models.schemas import SourceCitation
from app.rag.embeddings import EmbeddingService
from app.rag.vectorstore import VectorStore
from app.config import settings


class Retriever:
    """Retrieves relevant text chunks from the vector store for a given query."""

    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = None) -> Tuple[str, List[SourceCitation]]:
        """
        Embeds the query, retrieves top_k matching chunks, formats context text and citations.
        
        Returns:
            formatted_context: str to inject into LLM prompt
            citations: List[SourceCitation] for the API response
        """
        k = top_k or settings.TOP_K
        query_vector = self.embedding_service.embed_query(query)
        raw_matches = self.vector_store.query(query_vector, top_k=k)

        if not raw_matches:
            return "", []

        context_blocks: List[str] = []
        citations: List[SourceCitation] = []

        for item in raw_matches:
            page_num = item["page_number"]
            text = item["text"]
            score = item.get("similarity_score", None)

            context_blocks.append(f"--- [Page {page_num}] ---\n{text}")

            # Short preview excerpt for citation display
            excerpt_preview = text[:180] + "..." if len(text) > 180 else text
            citations.append(
                SourceCitation(
                    page_number=page_num,
                    excerpt=excerpt_preview,
                    relevance_score=score,
                )
            )

        formatted_context = "\n\n".join(context_blocks)
        return formatted_context, citations
