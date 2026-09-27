import os
from pathlib import Path
from dotenv import load_dotenv

# Locate base directories
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"


class Settings:
    """Application configuration dynamically loaded from environment variables."""

    def _reload_env(self):
        # Check root .env and backend .env
        root_env = BASE_DIR.parent / ".env"
        if root_env.exists():
            load_dotenv(dotenv_path=root_env, override=False)
        if ENV_PATH.exists():
            load_dotenv(dotenv_path=ENV_PATH, override=False)
        load_dotenv(override=False)

        # Check Streamlit secrets if running inside Streamlit Cloud
        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                for sec_key in ["GEMINI_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "LLM_PROVIDER", "GROQ_LLM_MODEL", "GEMINI_LLM_MODEL", "OPENAI_LLM_MODEL"]:
                    if sec_key in st.secrets and (sec_key not in os.environ or not os.environ[sec_key]):
                        os.environ[sec_key] = str(st.secrets[sec_key]).strip()
        except Exception:
            pass

    @property
    def GEMINI_API_KEY(self) -> str:
        self._reload_env()
        key = os.getenv("GEMINI_API_KEY", "").strip()
        # Fallback check if user put it in .env.example
        if not key or key.startswith("your_"):
            for expath in [BASE_DIR / ".env.example", BASE_DIR.parent / ".env.example"]:
                if expath.exists():
                    try:
                        for line in expath.read_text(encoding="utf-8").splitlines():
                            if line.startswith("GEMINI_API_KEY=") and not line.endswith("your_gemini_api_key_here"):
                                cand = line.split("=", 1)[1].strip()
                                if cand and not cand.startswith("your_"):
                                    return cand
                    except Exception:
                        pass
        return key

    @property
    def OPENAI_API_KEY(self) -> str:
        self._reload_env()
        return os.getenv("OPENAI_API_KEY", "").strip()

    @property
    def GROQ_API_KEY(self) -> str:
        self._reload_env()
        key = os.getenv("GROQ_API_KEY", "").strip()
        if not key or key.startswith("your_"):
            for expath in [BASE_DIR / ".env.example", BASE_DIR.parent / ".env.example"]:
                if expath.exists():
                    try:
                        for line in expath.read_text(encoding="utf-8").splitlines():
                            if line.startswith("GROQ_API_KEY=") and not line.endswith("your_groq_api_key_here"):
                                cand = line.split("=", 1)[1].strip()
                                if cand and not cand.startswith("your_"):
                                    return cand
                    except Exception:
                        pass
        return key

    @property
    def ACTIVE_PROVIDER(self) -> str:
        self._reload_env()
        forced = os.getenv("LLM_PROVIDER", "").strip().lower()
        if not forced:
            for expath in [BASE_DIR / ".env.example", BASE_DIR.parent / ".env.example"]:
                if expath.exists():
                    try:
                        for line in expath.read_text(encoding="utf-8").splitlines():
                            if line.startswith("LLM_PROVIDER="):
                                cand = line.split("=", 1)[1].strip().lower()
                                if cand in ["gemini", "openai", "groq"]:
                                    forced = cand
                                    break
                    except Exception:
                        pass
        if forced in ["gemini", "openai", "groq"]:
            return forced
        if self.GROQ_API_KEY and not self.GROQ_API_KEY.startswith("your_"):
            return "groq"
        if self.GEMINI_API_KEY and not self.GEMINI_API_KEY.startswith("your_"):
            return "gemini"
        if self.OPENAI_API_KEY and not self.OPENAI_API_KEY.startswith("your_"):
            return "openai"
        return "gemini" if self.GEMINI_API_KEY else "openai"

    # Gemini Models
    @property
    def GEMINI_LLM_MODEL(self) -> str:
        model = os.getenv("GEMINI_LLM_MODEL", "gemini-flash-latest").strip()
        if model in ["gemini-1.5-flash", "models/gemini-1.5-flash", "gemini-2.5-flash", "models/gemini-2.5-flash"]:
            return "gemini-flash-latest"
        return model

    @property
    def GEMINI_EMBEDDING_MODEL(self) -> str:
        model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001").strip()
        if model in ["models/text-embedding-004", "text-embedding-004"]:
            return "models/gemini-embedding-001"
        return model

    # OpenAI Models
    @property
    def OPENAI_LLM_MODEL(self) -> str:
        return os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")

    @property
    def OPENAI_EMBEDDING_MODEL(self) -> str:
        return os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Groq Models
    @property
    def GROQ_LLM_MODEL(self) -> str:
        return os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile").strip()

    # Chunking & Retrieval Parameters
    @property
    def CHUNK_SIZE(self) -> int:
        return int(os.getenv("CHUNK_SIZE", "800"))

    @property
    def CHUNK_OVERLAP(self) -> int:
        return int(os.getenv("CHUNK_OVERLAP", "150"))

    @property
    def TOP_K(self) -> int:
        return int(os.getenv("TOP_K", "4"))

    # Vector Store
    @property
    def CHROMA_PERSIST_DIRECTORY(self) -> str:
        # On Vercel / serverless runtimes, the filesystem is read-only except /tmp
        default_dir = "/tmp/chroma_db" if os.getenv("VERCEL") else str(BASE_DIR / "chroma_db")
        return os.getenv("CHROMA_PERSIST_DIRECTORY", default_dir)

    # Server Configuration
    @property
    def HOST(self) -> str:
        # Bind to 0.0.0.0 on cloud hosts (e.g. Render, Railway, Docker) or when PORT is provided
        return os.getenv("HOST", "0.0.0.0" if os.getenv("PORT") else "127.0.0.1")

    @property
    def PORT(self) -> int:
        return int(os.getenv("PORT", "8000"))


settings = Settings()
