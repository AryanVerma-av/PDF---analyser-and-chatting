# PDF RAG Assistant

A minimal, professional PDF Question-Answering application built with a Retrieval-Augmented Generation (RAG) pipeline.

---

## Architecture

The system implements a true RAG workflow without shortcutting or sending full document context directly to the LLM:

```
PDF Document
     |
     v
PDF Document Loader (pypdf page-by-page extraction)
     |
     v
Text Splitter (recursive chunking with page metadata)
     |
     v
Embedding Generator (OpenAI text-embedding-3-small)
     |
     v
Vector Store (ChromaDB collection with cosine similarity)
     |
     v
Retriever (top-k semantic matching + page metadata extraction)
     |
     v
LLM Generator (OpenAI gpt-4o-mini with grounded system prompt)
     |
     v
Answer + Page Citations (e.g. Source: Page 4, Source: Page 7)
```

---

## Project Structure

```
project/
|-- backend/
|   |-- app/
|   |   |-- __init__.py
|   |   |-- main.py              # FastAPI application, CORS, static mounting
|   |   |-- config.py            # Environment configuration
|   |   |-- models/
|   |   |   |-- __init__.py
|   |   |   `-- schemas.py       # Pydantic request/response schemas
|   |   |-- rag/
|   |   |   |-- __init__.py
|   |   |   |-- loader.py        # PDF document loader (per-page extraction)
|   |   |   |-- splitter.py      # Recursive chunker with page metadata
|   |   |   |-- embeddings.py    # OpenAI embedding service wrapper
|   |   |   |-- vectorstore.py   # ChromaDB vector store manager
|   |   |   |-- retriever.py     # Semantic top-k retriever
|   |   |   |-- generator.py     # Grounded LLM response generator
|   |   |   `-- pipeline.py      # Master coordinator for RAG pipeline
|   |   |-- routes/
|   |   |   |-- __init__.py
|   |   |   |-- health.py        # GET /health
|   |   |   |-- upload.py        # POST /upload
|   |   |   `-- ask.py           # POST /ask
|   |   `-- services/
|   |       |-- __init__.py
|   |       `-- state.py         # Singleton pipeline state provider
|   |-- requirements.txt
|   `-- .env.example
|-- frontend/
|   |-- index.html               # Minimalist modern SaaS interface (no emojis)
|   |-- style.css                # Clean typography, subtle borders, high contrast
|   `-- app.js                   # Client state machine and API communication
|-- tests/
|   `-- test_pipeline.py         # Automated verification test suite
|-- .env.example
|-- .gitignore
`-- README.md
```

---

## Requirements

- Python 3.10+ (tested on Python 3.14)
- An active OpenAI API key with access to:
  - `text-embedding-3-small` (embeddings)
  - `gpt-4o-mini` (chat completion)

---

## Installation & Setup

### 1. Clone or Open Workspace

Navigate to the project root directory:

```bash
cd "project #2"
```

### 2. Configure Environment Variables

Copy the example configuration to `.env`:

```bash
cp .env.example backend/.env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example backend\.env
```

Edit `backend/.env` and insert your desired provider API key:

```env
# 1. Google Gemini (Fast & free embeddings + generation)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_LLM_MODEL=gemini-flash-latest
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001

# 2. Groq (Ultra-fast LLM generation using LPU inference)
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_LLM_MODEL=llama-3.3-70b-versatile

# 3. OpenAI (Alternative)
OPENAI_API_KEY=sk-your_openai_api_key_here
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Set active provider: "gemini", "groq", or "openai"
LLM_PROVIDER=gemini
```

### 3. Install Dependencies

Install the required packages using pip:

```bash
python -m pip install -r backend/requirements.txt
```

---

## Running the Application

### Option A: Unified Server (Recommended)

The backend server is configured to serve both the REST API and the frontend client simultaneously:

```bash
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

Or from inside the `backend` directory:

```bash
cd backend
python -m uvicorn app.main:app --reload
```

Once running, open your web browser and navigate to:

```
http://127.0.0.1:8000
```

### Option B: Standalone Frontend

If you prefer to serve the frontend via a separate static file server:

1. Start the backend on `http://127.0.0.1:8000`.
2. Open `frontend/index.html` directly in your browser or run:
   ```bash
   python -m http.server 3000 --directory frontend
   ```
   and visit `http://127.0.0.1:3000`. The frontend client will automatically connect to `http://127.0.0.1:8000`.

---

## API Endpoints

### 1. GET /health
Checks if the backend is running and returns current index metadata.

**Response:**
```json
{
  "status": "ok",
  "active_document": "sample_document.pdf",
  "total_pages": 12,
  "total_chunks": 48,
  "vector_store_ready": true
}
```

### 2. POST /upload
Accepts a PDF multipart upload (`file`), extracts text page by page, splits into overlapping chunks, generates embeddings, and replaces the active ChromaDB vector index.

**Response:**
```json
{
  "message": "PDF processed and indexed successfully.",
  "filename": "sample_document.pdf",
  "total_pages": 12,
  "total_chunks": 48,
  "processing_time_seconds": 1.45
}
```

### 3. POST /ask
Accepts a JSON payload with a question, embeds the query, retrieves the top-k most similar chunks, and generates a grounded response with page citations.

**Request:**
```json
{
  "question": "What is the primary evaluation metric mentioned in the document?"
}
```

**Response:**
```json
{
  "question": "What is the primary evaluation metric mentioned in the document?",
  "answer": "According to Page 5, the primary evaluation metric used is the F1-score across all validated test samples.",
  "sources": [
    {
      "page_number": 5,
      "excerpt": "Evaluation methodology: All models were benchmarked using weighted F1-score...",
      "relevance_score": 0.8921
    }
  ],
  "latency_seconds": 0.78
}
```

---

## Verification Test Suite

An automated test suite is included to verify all pipeline stages end-to-end:

```bash
python tests/test_pipeline.py
```

The test validates:
- Multi-page PDF text extraction and accurate page indexing.
- Recursive chunking with metadata preservation.
- Vector store indexing and cosine similarity retrieval.
- FastAPI endpoints (input validation, error handling, status codes).
- Live OpenAI question answering and citation generation (when `OPENAI_API_KEY` is present).

---

## Troubleshooting

- **Error: `OPENAI_API_KEY is not configured`**
  Ensure you have created `backend/.env` containing a valid `OPENAI_API_KEY`.
- **Error: `Invalid file type. Only PDF documents (.pdf) are supported.`**
  Ensure the uploaded file has a `.pdf` extension.
- **Error: `No readable text found in PDF`**
  The uploaded PDF may be a scanned document or image-only file. Use an OCR tool or a text-based PDF.
- **Error: `No PDF has been uploaded yet`**
  You must upload and process a PDF before submitting questions.

---

## Deployment Guide

### Deploying to Vercel (Recommended)

This repository is pre-configured for **zero-config Vercel deployment** with the FastAPI ASGI runtime:

1. Push this repository to GitHub:
   ```bash
   git add .
   git commit -m "Configure Vercel deployment and entrypoints"
   git push origin main
   ```
2. Import the repository in [Vercel](https://vercel.com/new).
3. In **Project Settings -> Environment Variables**, add your API keys:
   - `GEMINI_API_KEY` (or `GROQ_API_KEY` or `OPENAI_API_KEY`)
   - `LLM_PROVIDER`: `gemini` (or `groq` / `openai`)
4. Click **Deploy**. Vercel will:
   - Automatically detect the FastAPI application via `pyproject.toml` (`backend.app.main:app`).
   - Serve the frontend and static files from the global Edge CDN via `public/`.
   - Run the API routes on Vercel Functions with automatic scaling.

### Deploying with Docker / Container Platforms

1. Build the Docker container:
   ```bash
   docker build -t pdf-rag-assistant .
   ```
2. Run the container:
   ```bash
   docker run -p 8000:8000 --env-file backend/.env pdf-rag-assistant
   ```
3. Visit `http://localhost:8000`.

### Deploying to Render or Railway

- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
- Set your environment variables (`GEMINI_API_KEY` / `GROQ_API_KEY` / `OPENAI_API_KEY`).

