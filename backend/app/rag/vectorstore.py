from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.models.schemas import DocumentChunk, SourceCitation
from app.config import settings


class VectorStoreError(Exception):
    """Exception raised when vector database operations fail."""
    pass


class VectorStore:
    """
    Manages vector storage and similarity search using ChromaDB.
    Supports single-document index replacement and metadata preservation.
    """

    COLLECTION_NAME = "active_pdf_collection"

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIRECTORY
        try:
            # Using persistent client to maintain index across requests
            self.client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            self._init_collection()
        except Exception as exc:
            # Fall back to ephemeral in-memory client if file system issues arise
            try:
                self.client = chromadb.Client(
                    settings=ChromaSettings(anonymized_telemetry=False)
                )
                self._init_collection()
            except Exception as inner_exc:
                raise VectorStoreError(f"Failed to initialize ChromaDB: {str(inner_exc)}") from inner_exc

    def _init_collection(self):
        """Get existing collection or create a fresh one."""
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def reset(self):
        """Clear all indexed documents for fresh PDF ingestion."""
        try:
            self.client.delete_collection(name=self.COLLECTION_NAME)
        except Exception:
            pass
        self._init_collection()

    def count(self) -> int:
        """Return the number of stored chunk vectors."""
        try:
            return self.collection.count()
        except Exception:
            return 0

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        """
        Add document chunks and their corresponding embedding vectors into ChromaDB.
        Preserves page_number and source_file in metadata.
        """
        if not chunks or not embeddings:
            return

        if len(chunks) != len(embeddings):
            raise VectorStoreError("Mismatch between chunk count and embedding count.")

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
            # Add to ChromaDB collection in batches to stay within size limits
            batch_size = 200
            for i in range(0, len(ids), batch_size):
                end_i = i + batch_size
                self.collection.add(
                    ids=ids[i:end_i],
                    documents=documents[i:end_i],
                    embeddings=embeddings[i:end_i],
                    metadatas=metadatas[i:end_i],
                )
        except Exception as exc:
            raise VectorStoreError(f"Failed to insert vectors into ChromaDB: {str(exc)}") from exc

    def query(self, query_embedding: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Query vector store for top_k most similar chunks.
        Returns list of dicts with 'text', 'page_number', 'source_file', 'score'.
        """
        if self.count() == 0:
            return []

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
                    # For cosine distance: cosine_similarity = 1 - distance
                    sim_score = round(max(0.0, 1.0 - float(dist)), 4)
                    retrieved.append({
                        "text": doc,
                        "page_number": int(meta.get("page_number", 1)),
                        "source_file": str(meta.get("source_file", "unknown")),
                        "similarity_score": sim_score,
                    })

            return retrieved
        except Exception as exc:
            raise VectorStoreError(f"Vector search failed: {str(exc)}") from exc
