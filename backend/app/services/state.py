from app.rag.pipeline import RAGPipeline

# Global pipeline instance serving API requests
pipeline_instance = RAGPipeline()


def get_pipeline() -> RAGPipeline:
    """Dependency injector / accessor for the active RAG pipeline."""
    return pipeline_instance
