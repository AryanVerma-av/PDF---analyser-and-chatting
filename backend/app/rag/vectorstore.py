from typing import List, Dict, Any, Optional, Tuple
import math

# SQLite compatibility shim for serverless/Linux environments if pysqlite3 is available
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMA_AVAILABLE = True
except Exception:
    CHROMA_AVAILABLE = False

from app.models.schemas import DocumentChunk, SourceCitation
from app.config import settings


class VectorStoreError(Exception):
    """Exception raised when vector database operations fail."""
    pass


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Computes cosine similarity between two numeric vectors."""
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for a, b in zip(vec1, vec2):
        dot += a * b
        norm1 += a * a
        norm2 += b * b
    if norm1 <= 0.0 or norm2 <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm1) * math.sqrt(norm2))


class VectorStore:
    """
    Manages vector storage and similarity search.
    Defaults to ChromaDB (persistent or ephemeral) and gracefully falls back to
    an in-memory vector index if running in constrained or serverless environments.
    """

    COLLECTION_NAME = "active_pdf_collection"

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIRECTORY
        self.use_in_memory_fallback = not CHROMA_AVAILABLE
        self._memory_chunks: List[DocumentChunk] = []
        self._memory_embeddings: List[List[float]] = []

        if not self.use_in_memory_fallback:
            try:
                # Attempt persistent client first
                self.client = chromadb.PersistentClient(
                    path=self.persist_dir,
                    settings=ChromaSettings(anonymized_telemetry=False)
                )
                self._init_collection()
            except Exception:
                try:
                    # Attempt in-memory Chroma client
                    self.client = chromadb.Client(
                        settings=ChromaSettings(anonymized_telemetry=False)
                    )
                    self._init_collection()
                except Exception:
                    # Fall back to pure in-memory cosine similarity store
                    self.use_in_memory_fallback = True

    def _init_collection(self):
        """Get existing collection or create a fresh one."""
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def reset(self):
        """Clear all indexed documents for fresh PDF ingestion."""
        self._memory_chunks = []
        self._memory_embeddings = []
        if not self.use_in_memory_fallback:
            try:
                self.client.delete_collection(name=self.COLLECTION_NAME)
            except Exception:
                pass
            try:
                self._init_collection()
            except Exception:
                self.use_in_memory_fallback = True

    def count(self) -> int:
        """Return the number of stored chunk vectors."""
        if self.use_in_memory_fallback:
            return len(self._memory_chunks)
        try:
            return self.collection.count()
        except Exception:
            return len(self._memory_chunks)

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        """
        Add document chunks and their corresponding embedding vectors.
        Preserves page_number and source_file in metadata.
        """
        if not chunks or not embeddings:
            return

        if len(chunks) != len(embeddings):
            raise VectorStoreError("Mismatch between chunk count and embedding count.")

        # Always store in memory cache for seamless fallback
        self._memory_chunks = list(chunks)
        self._memory_embeddings = list(embeddings)

        if not self.use_in_memory_fallback:
            ids = [chunk.chunk_id for chunk in chunks]
            documents = [chunk.text for chunk in chunks]
            metadatas = [
                {
                    "page_number": chunk.page_number,
                    "source_file": chunk.source_file,
                }
                for chunk in chunks
            ]

            try:
                batch_size = 200
                for i in range(0, len(ids), batch_size):
                    end_i = i + batch_size
                    self.collection.add(
                        ids=ids[i:end_i],
                        documents=documents[i:end_i],
                        embeddings=embeddings[i:end_i],
                        metadatas=metadatas[i:end_i],
                    )
            except Exception:
                # If ChromaDB collection insertion fails (e.g. read-only disk), activate in-memory mode
                self.use_in_memory_fallback = True

    def query(self, query_embedding: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Query vector store for top_k most similar chunks.
        Returns list of dicts with 'text', 'page_number', 'source_file', 'similarity_score'.
        """
        if self.count() == 0:
            return []

        if not self.use_in_memory_fallback:
            try:
                effective_k = min(top_k, self.count())
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=effective_k,
                    include=["documents", "metadatas", "distances"],
                )

                retrieved: List[Dict[str, Any]] = []
                if results and results.get("documents") and results["documents"][0]:
                    docs = results["documents"][0]
                    metas = results["metadatas"][0] if results.get("metadatas") else []
                    distances = results["distances"][0] if results.get("distances") else []

                    for idx, doc in enumerate(docs):
                        meta = metas[idx] if idx < len(metas) else {}
                        dist = distances[idx] if idx < len(distances) else 0.0
                        sim_score = round(max(0.0, 1.0 - float(dist)), 4)
                        retrieved.append({
                            "text": doc,
                            "page_number": int(meta.get("page_number", 1)),
                            "source_file": str(meta.get("source_file", "unknown")),
                            "similarity_score": sim_score,
                        })

                if retrieved:
                    return retrieved
            except Exception:
                self.use_in_memory_fallback = True

        # In-memory cosine similarity query
        scored: List[Tuple[float, DocumentChunk]] = []
        for chunk, emb in zip(self._memory_chunks, self._memory_embeddings):
            score = _cosine_similarity(query_embedding, emb)
            scored.append((score, chunk))

        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)
        top_matches = scored[:top_k]

        return [
            {
                "text": chunk.text,
                "page_number": chunk.page_number,
                "source_file": chunk.source_file,
                "similarity_score": round(max(0.0, float(score)), 4),
            }
            for score, chunk in top_matches
        ]
