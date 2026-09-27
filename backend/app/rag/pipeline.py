import time
from typing import Dict, Any, Tuple, List, Optional
from app.config import settings
from app.models.schemas import SourceCitation
from app.rag.loader import PDFLoader
from app.rag.splitter import TextSplitter
from app.rag.embeddings import EmbeddingService
from app.rag.vectorstore import VectorStore
from app.rag.retriever import Retriever
from app.rag.generator import LLMGenerator


class RAGPipeline:
    """
    Coordinates the complete RAG lifecycle:
    PDF -> PDF Document Loader -> Text Chunks -> Embeddings -> Vector Store -> Retriever -> LLM -> Answer
    """

    def __init__(self):
        self.loader = PDFLoader()
        self.splitter = TextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.retriever = Retriever(self.embedding_service, self.vector_store)
        self.generator = LLMGenerator()

        self.active_filename: Optional[str] = None
        self.total_pages: int = 0
        self.total_chunks: int = 0

    def is_ready(self) -> bool:
        """Checks if a PDF has been successfully ingested and indexed."""
        return bool(self.active_filename and self.vector_store.count() > 0)

    def ingest_pdf(
        self,
        file_bytes: bytes,
        filename: str,
        gemini_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Processes an uploaded PDF through:
        1. Text extraction page by page
        2. Chunking with page metadata
        3. Embedding generation (Gemini or OpenAI)
        4. Vector store indexing
        """
        start_time = time.time()

        # Step 1: Load and extract pages
        pages_data = self.loader.load_from_bytes(file_bytes, filename)
        page_count = len(pages_data)

        # Step 2: Split text into chunks
        chunks = self.splitter.split_pages(pages_data)
        if not chunks:
            raise ValueError("No text chunks could be generated from the document.")

        # Step 3: Generate embeddings
        chunk_texts = [c.text for c in chunks]
        embedding_key = (gemini_key or openai_key or "").strip() or None
        emb_provider = "gemini" if gemini_key else (provider or None)
        embeddings = self.embedding_service.embed_texts(chunk_texts, api_key=embedding_key, provider=emb_provider)

        # Step 4: Clear previous index and store new embeddings in vector database
        self.vector_store.reset()
        self.vector_store.add_chunks(chunks, embeddings)

        elapsed = round(time.time() - start_time, 2)
        self.active_filename = filename
        self.total_pages = page_count
        self.total_chunks = len(chunks)

        return {
            "filename": filename,
            "total_pages": self.total_pages,
            "total_chunks": self.total_chunks,
            "processing_time_seconds": elapsed,
        }

    def answer_question(
        self,
        question: str,
        gemini_key: Optional[str] = None,
        groq_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        provider: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[str, List[SourceCitation], float]:
        """
        Answers a user question through:
        1. Retrieval of top relevant chunks
        2. Grounded LLM generation
        """
        if not self.is_ready():
            raise RuntimeError("No PDF has been uploaded yet. Please upload a PDF first.")

        start_time = time.time()

        # Step 5 & 6: Retrieve relevant chunks with citations
        embedding_key = (gemini_key or openai_key or "").strip() or None
        emb_provider = "gemini" if gemini_key else (provider or None)
        k_val = top_k if top_k is not None else settings.TOP_K
        context, citations = self.retriever.retrieve(
            question,
            top_k=k_val,
            embedding_key=embedding_key,
            provider=emb_provider,
        )

        # Step 7 & 8: Generate grounded answer from LLM
        answer = self.generator.generate(
            question,
            context,
            gemini_key=gemini_key,
            groq_key=groq_key,
            openai_key=openai_key,
            provider=provider,
        )

        latency = round(time.time() - start_time, 2)
        return answer, citations, latency
