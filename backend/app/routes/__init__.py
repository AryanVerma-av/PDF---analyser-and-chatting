from .health import router as health_router
from .upload import router as upload_router
from .ask import router as ask_router

__all__ = ["health_router", "upload_router", "ask_router"]
