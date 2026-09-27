import sys
from pathlib import Path

# Add backend directory to sys.path so "app.*" imports work when run from repo root (e.g. Vercel)
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.routes.health import router as health_router
from app.routes.upload import router as upload_router
from app.routes.ask import router as ask_router
from app.config import settings

# Create FastAPI instance
app = FastAPI(
    title="PDF RAG Assistant API",
    description="Minimal and professional PDF Question-Answering system powered by RAG",
    version="1.0.0",
)

# CORS configuration for local development and deployed frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(health_router)
app.include_router(upload_router)
app.include_router(ask_router)

# Mount frontend files for convenient single-server running
candidate_dirs = [
    Path(__file__).resolve().parent.parent.parent / "frontend",
    Path(__file__).resolve().parent.parent.parent / "public",
    Path(__file__).resolve().parent.parent / "frontend",
    Path.cwd() / "frontend",
    Path.cwd() / "public",
]

FRONTEND_DIR = None
for candidate in candidate_dirs:
    if candidate.exists() and (candidate / "index.html").exists():
        FRONTEND_DIR = candidate
        break

if FRONTEND_DIR:
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/index.html", include_in_schema=False)
    async def serve_index_html():
        return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
