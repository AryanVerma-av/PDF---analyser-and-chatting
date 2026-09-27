from typing import List, Optional
from app.config import settings


class EmbeddingServiceError(Exception):
    """Exception raised when embedding generation encounters an error."""
    pass


class EmbeddingService:
    """Generates dense vector embeddings using Google Gemini or OpenAI."""

    def __init__(self):
        prov = settings.ACTIVE_PROVIDER
        if prov == "groq":
            # Groq does not provide embedding models; route vectors through Gemini or OpenAI
            if settings.GEMINI_API_KEY and not settings.GEMINI_API_KEY.startswith("your_"):
                self._provider = "gemini"
            elif settings.OPENAI_API_KEY and not settings.OPENAI_API_KEY.startswith("your_"):
                self._provider = "openai"
            else:
                self._provider = "gemini"
        else:
            self._provider = prov

    def _embed_gemini(self, texts: List[str], is_query: bool = False, api_key: Optional[str] = None) -> List[List[float]]:
        key = (api_key or settings.GEMINI_API_KEY or "").strip()
        if not key or key.startswith("your_"):
            raise EmbeddingServiceError(
                "GEMINI_API_KEY is not configured. Please add GEMINI_API_KEY in Vercel Project Settings -> Environment Variables, or enter it in the API Key settings on the page."
            )

        try:
            import google.generativeai as genai
            genai.configure(api_key=key)

            task = "retrieval_query" if is_query else "retrieval_document"
            embeddings: List[List[float]] = []

            model_candidate = settings.GEMINI_EMBEDDING_MODEL
            if "text-embedding-004" in model_candidate:
                model_candidate = "models/gemini-embedding-001"

            # Process in batches of 50
            batch_size = 50
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                try:
                    res = genai.embed_content(
                        model=model_candidate,
                        content=batch,
                        task_type=task,
                    )
                except Exception as model_err:
                    if "404" in str(model_err) or "not found" in str(model_err).lower():
                        fallback = "models/gemini-embedding-001" if model_candidate != "models/gemini-embedding-001" else "models/gemini-embedding-2"
                        res = genai.embed_content(
                            model=fallback,
                            content=batch,
                            task_type=task,
                        )
                    else:
                        raise model_err

                emb_data = res.get("embedding", [])
                # If single item batch, it might return a single vector
                if emb_data and isinstance(emb_data[0], (int, float)):
                    embeddings.append(emb_data)
                else:
                    embeddings.extend(emb_data)

            return embeddings
        except Exception as exc:
            raise EmbeddingServiceError(f"Google Gemini embedding API failed: {str(exc)}") from exc

    def _embed_openai(self, texts: List[str], api_key: Optional[str] = None) -> List[List[float]]:
        key = (api_key or settings.OPENAI_API_KEY or "").strip()
        if not key or key.startswith("your_"):
            raise EmbeddingServiceError(
                "OPENAI_API_KEY is not configured. Please add OPENAI_API_KEY or GEMINI_API_KEY in Vercel Project Settings -> Environment Variables, or enter it in the API Key settings on the page."
            )

        try:
            from openai import OpenAI
            client = OpenAI(api_key=key)
            embeddings: List[List[float]] = []
            batch_size = 100

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                cleaned_batch = [t.replace("\n", " ") for t in batch]
                response = client.embeddings.create(
                    model=settings.OPENAI_EMBEDDING_MODEL,
                    input=cleaned_batch,
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            return embeddings
        except Exception as exc:
            raise EmbeddingServiceError(f"OpenAI embedding API failed: {str(exc)}") from exc

    @property
    def provider(self) -> str:
        return getattr(self, "_provider", "gemini")

    def embed_texts(self, texts: List[str], api_key: Optional[str] = None, provider: Optional[str] = None) -> List[List[float]]:
        """Generate embeddings for a list of document chunk texts."""
        if not texts:
            return []

        prov = provider or self.provider
        if prov == "gemini":
            return self._embed_gemini(texts, is_query=False, api_key=api_key)
        return self._embed_openai(texts, api_key=api_key)

    def embed_query(self, query: str, api_key: Optional[str] = None, provider: Optional[str] = None) -> List[float]:
        """Generate embedding vector for a single search query."""
        clean_q = query.strip()
        if not clean_q:
            raise EmbeddingServiceError("Query cannot be empty.")

        prov = provider or self.provider
        if prov == "gemini":
            res = self._embed_gemini([clean_q], is_query=True, api_key=api_key)
        else:
            res = self._embed_openai([clean_q], api_key=api_key)

        if not res:
            raise EmbeddingServiceError("Failed to generate embedding for query.")
        return res[0]
