import os
import sys
import time
from pathlib import Path
from typing import List

# Prepend directories to sys.path
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Map 'app' to backend.app package so submodules can resolve 'from app...'
import backend.app as backend_app
sys.modules["app"] = backend_app

from dotenv import load_dotenv

# Load environment variables from root or backend
load_dotenv(dotenv_path=ROOT_DIR / ".env")
load_dotenv(dotenv_path=BACKEND_DIR / ".env")

import streamlit as st
from backend.app.config import settings
from backend.app.rag.pipeline import RAGPipeline
from backend.app.rag.loader import PDFLoaderError
from backend.app.rag.embeddings import EmbeddingServiceError
from backend.app.rag.vectorstore import VectorStoreError
from backend.app.rag.generator import GeneratorError

# Page Configuration
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: var(--text-color);
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        background-color: #f1f5f9;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 12px;
    }
    .citation-card {
        background-color: #f8fafc;
        border-left: 3px solid #3b82f6;
        padding: 8px 12px;
        margin-top: 6px;
        margin-bottom: 6px;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    .latency-tag {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_default_key(name: str) -> str:
    """Retrieve secret from Streamlit secrets, env, or settings."""
    try:
        if hasattr(st, "secrets") and name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    val = os.getenv(name, "").strip()
    if val and not val.startswith("your_"):
        return val
    config_val = getattr(settings, name, "")
    if config_val and not str(config_val).startswith("your_"):
        return str(config_val)
    return ""


# Initialize Session State
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_doc" not in st.session_state:
    st.session_state.indexed_doc = None


# Sidebar
with st.sidebar:
    st.header("📄 Document Ingestion")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF document to parse, chunk, and index for semantic Q&A.",
    )

    if uploaded_file is not None:
        # Check if new document needs ingestion
        current_doc = st.session_state.indexed_doc
        if not current_doc or current_doc.get("name") != uploaded_file.name:
            with st.spinner("Processing document: extracting text, chunking, and indexing..."):
                file_bytes = uploaded_file.read()
                try:
                    active_gemini_key = st.session_state.get("gemini_key_input", get_default_key("GEMINI_API_KEY"))
                    active_openai_key = st.session_state.get("openai_key_input", get_default_key("OPENAI_API_KEY"))
                    active_provider = st.session_state.get("selected_provider", (settings.ACTIVE_PROVIDER or "groq").lower())

                    result = st.session_state.pipeline.ingest_pdf(
                        file_bytes=file_bytes,
                        filename=uploaded_file.name,
                        gemini_key=active_gemini_key.strip() or None,
                        openai_key=active_openai_key.strip() or None,
                        provider=active_provider,
                    )
                    st.session_state.indexed_doc = {
                        "name": result["filename"],
                        "pages": result["total_pages"],
                        "chunks": result["total_chunks"],
                        "time": result["processing_time_seconds"],
                    }
                    # Add welcome system message to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"I have processed and indexed **{uploaded_file.name}** ({result['total_pages']} pages, {result['total_chunks']} chunks). Ask me any question based on this document!",
                        "sources": [],
                        "latency": result["processing_time_seconds"],
                    })
                    st.success("PDF indexed successfully!")
                except Exception as exc:
                    st.error(f"Error processing PDF: {str(exc)}")

    # Show Document Metadata if loaded
    if st.session_state.indexed_doc:
        doc = st.session_state.indexed_doc
        st.markdown(
            f"""
            <div style="background-color: #f1f5f9; padding: 12px; border-radius: 8px; margin-top: 10px; margin-bottom: 15px;">
                <div style="font-weight: 600; font-size: 0.9rem; word-break: break-word;">📑 {doc['name']}</div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">
                    Pages: <b>{doc['pages']}</b> | Chunks: <b>{doc['chunks']}</b><br>
                    Processed in <b>{doc['time']}s</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("🗑️ Clear Document & Chat", use_container_width=True):
            st.session_state.pipeline = RAGPipeline()
            st.session_state.messages = []
            st.session_state.indexed_doc = None
            st.rerun()

    st.markdown("---")
    st.header("⚙️ Model Configuration")

    provider_options = ["Groq", "Gemini", "OpenAI"]
    default_prov = settings.ACTIVE_PROVIDER.title() if settings.ACTIVE_PROVIDER else "Groq"
    prov_idx = provider_options.index(default_prov) if default_prov in provider_options else 0

    selected_provider = st.selectbox(
        "Active LLM Provider",
        provider_options,
        index=prov_idx,
        help="Select which LLM provider to query for answers.",
        key="selected_provider",
    ).lower()

    # Provider-specific keys
    groq_key_default = get_default_key("GROQ_API_KEY")
    gemini_key_default = get_default_key("GEMINI_API_KEY")
    openai_key_default = get_default_key("OPENAI_API_KEY")

    groq_api_key = st.text_input(
        "Groq API Key",
        value=groq_key_default,
        type="password",
        help="Paste your Groq API key (starts with gsk_)",
        key="groq_key_input",
    )

    gemini_api_key = st.text_input(
        "Gemini API Key",
        value=gemini_key_default,
        type="password",
        help="Paste your Google Gemini API key",
        key="gemini_key_input",
    )

    openai_api_key = st.text_input(
        "OpenAI API Key (Optional)",
        value=openai_key_default,
        type="password",
        help="Paste your OpenAI API key (starts with sk-)",
        key="openai_key_input",
    )

    # Retrieval parameter
    st.markdown("---")
    st.header("🔍 Retrieval Settings")
    top_k = st.slider("Top Chunks (k)", min_value=1, max_value=8, value=settings.TOP_K, help="Number of chunks retrieved per question.")


# Main Application Area
st.markdown('<div class="main-header">📄 PDF RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Retrieval-Augmented Generation system with exact page-by-page citations.</div>', unsafe_allow_html=True)

# If no document is uploaded, show welcome instructions
if not st.session_state.indexed_doc:
    st.info("👋 **Welcome!** Please upload a PDF in the left sidebar to begin analyzing and chatting with your document.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 1. Ingestion")
        st.write("Extracts text page by page with full structural metadata preservation.")
    with col2:
        st.markdown("#### 2. Vector Indexing")
        st.write("Generates dense semantic embeddings and indexes chunks with cosine similarity.")
    with col3:
        st.markdown("#### 3. Grounded Q&A")
        st.write("Retrieves the most relevant excerpts and cites exact source pages.")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display citations if available
        if message.get("sources"):
            with st.expander(f"📚 Sources Cited ({len(message['sources'])} page reference{'s' if len(message['sources']) > 1 else ''})"):
                for citation in message["sources"]:
                    relevance = f" • Relevance: {int(citation.relevance_score * 100)}%" if citation.relevance_score else ""
                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <strong>Source: Page {citation.page_number}</strong>{relevance}<br>
                            <em>"{citation.excerpt}"</em>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        if message.get("latency"):
            st.markdown(f'<div class="latency-tag">⏱️ Generated in {message["latency"]}s</div>', unsafe_allow_html=True)


# User Question Input
if prompt := st.chat_input("Ask a question about the uploaded document..."):
    if not st.session_state.pipeline.is_ready():
        st.warning("⚠️ Please upload a PDF in the sidebar before asking questions.")
    else:
        # Display user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching document chunks and formulating answer..."):
                start_time = time.time()
                try:
                    answer, citations, latency = st.session_state.pipeline.answer_question(
                        question=prompt,
                        gemini_key=gemini_api_key.strip() or None,
                        groq_key=groq_api_key.strip() or None,
                        openai_key=openai_api_key.strip() or None,
                        provider=selected_provider,
                        top_k=top_k,
                    )

                    st.markdown(answer)

                    if citations:
                        with st.expander(f"📚 Sources Cited ({len(citations)} page reference{'s' if len(citations) > 1 else ''})"):
                            for citation in citations:
                                relevance = f" • Relevance: {int(citation.relevance_score * 100)}%" if citation.relevance_score else ""
                                st.markdown(
                                    f"""
                                    <div class="citation-card">
                                        <strong>Source: Page {citation.page_number}</strong>{relevance}<br>
                                        <em>"{citation.excerpt}"</em>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    st.markdown(f'<div class="latency-tag">⏱️ Generated in {latency}s</div>', unsafe_allow_html=True)

                    # Append to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": citations,
                        "latency": latency,
                    })

                except Exception as exc:
                    st.error(f"Failed to generate answer: {str(exc)}")
