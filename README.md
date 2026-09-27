# 📄 PDF RAG Assistant

A minimal, professional, and high-performance PDF Question-Answering application powered by a Retrieval-Augmented Generation (RAG) pipeline and an interactive **Streamlit** user interface.

Supports **Groq** (ultra-fast LLaMA 3.3 70B), **Google Gemini** (Gemini Flash), and **OpenAI** (GPT-4o-mini).

---

## 🚀 Quickstart (Run with Streamlit)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` (or edit your existing `.env` file):
```bash
cp .env.example .env
```
Add your API key (Groq, Gemini, or OpenAI):
```env
# Groq (Recommended - Free & ultra-fast)
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_LLM_MODEL=llama-3.3-70b-versatile
LLM_PROVIDER=groq

# Gemini (Alternative)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_LLM_MODEL=gemini-flash-latest
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001

# OpenAI (Alternative)
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Launch Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Deploy to Streamlit Community Cloud (1-Click Free Hosting)

1. **Push your repository to GitHub** (Ensure `.env` is ignored by `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
3. Click **New app** and configure:
   - **Repository:** `AryanVerma-av/PDF---analyser-and-chatting`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Advanced settings...** -> **Secrets** and paste your environment variables:
   ```toml
   GROQ_API_KEY = "gsk_your_groq_api_key_here"
   GEMINI_API_KEY = "your_gemini_api_key_here"
   LLM_PROVIDER = "groq"
   ```
5. Click **Deploy!**

---

## 🏗️ Architecture

The application implements a genuine RAG workflow without shortcutting or sending full document context directly to the LLM:

```
PDF Document
     │
     ▼
PDF Document Loader (pypdf page-by-page extraction)
     │
     ▼
Text Splitter (recursive chunking preserving page metadata)
     │
     ▼
Embedding Generator (Gemini, OpenAI, or local deterministic embeddings)
     │
     ▼
Vector Store (ChromaDB collection with cosine similarity)
     │
     ▼
Retriever (top-k semantic matching + page metadata extraction)
     │
     ▼
LLM Generator (Groq LLaMA 3.3, Gemini Flash, or OpenAI GPT-4o-mini)
     │
     ▼
Answer + Exact Page Citations (e.g., Source: Page 3, Source: Page 5)
```

---

## 📂 Project Structure

```
PDF---analyser-and-chatting/
├── app.py                  # Main Streamlit web application
├── requirements.txt        # Streamlit & core RAG dependencies
├── .env                    # Local environment variables (kept private)
├── .env.example            # Environment template for keys & models
├── .gitignore              # Protects keys, cache, and vector databases
├── backend/
│   ├── app/
│   │   ├── config.py       # Configuration loader (.env and st.secrets)
│   │   ├── main.py         # Optional FastAPI endpoints
│   │   ├── models/
│   │   │   └── schemas.py  # Pydantic data schemas
│   │   ├── rag/
│   │   │   ├── loader.py       # Per-page PDF parser
│   │   │   ├── splitter.py     # Recursive text chunker
│   │   │   ├── embeddings.py   # Embedding services with local fallback
│   │   │   ├── vectorstore.py  # ChromaDB vector store
│   │   │   ├── retriever.py    # Semantic top-k retriever
│   │   │   ├── generator.py    # Multi-provider LLM response generator
│   │   │   └── pipeline.py     # RAG pipeline coordinator
│   │   └── routes/         # REST API routes (health, upload, ask)
│   └── requirements.txt
└── tests/
    └── test_pipeline.py    # Automated verification test suite
```

---

## 🧪 Running the Verification Test Suite

Run the end-to-end test suite to verify PDF loading, chunking, indexing, retrieval, and schema integrity:

```bash
python tests/test_pipeline.py
```

---

## 🛡️ Security & Environment Variables

- `.env` files contain sensitive API credentials and are automatically ignored by git.
- When deploying on Streamlit Cloud, credentials should always be stored in **Streamlit Secrets** (`st.secrets`), which the application automatically loads into settings.
