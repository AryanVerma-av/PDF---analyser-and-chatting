// PDF RAG Assistant - Frontend Client Logic

const API_BASE = window.location.origin.includes("http") && !window.location.origin.includes("5500") && !window.location.origin.includes("3000")
  ? "" 
  : "http://127.0.0.1:8000";

// DOM Element References
const connectionStatus = document.getElementById("connection-status");
const connectionText = document.getElementById("connection-text");
const errorBanner = document.getElementById("error-banner");
const errorMessage = document.getElementById("error-message");
const dismissErrorBtn = document.getElementById("dismiss-error-btn");

const keysWarningBanner = document.getElementById("keys-warning-banner");
const bannerConfigKeysBtn = document.getElementById("banner-config-keys-btn");
const openApiModalBtn = document.getElementById("open-api-modal-btn");
const apiModal = document.getElementById("api-modal");
const closeApiModalBtn = document.getElementById("close-api-modal-btn");
const apiKeyForm = document.getElementById("api-key-form");
const geminiKeyInput = document.getElementById("gemini-key-input");
const groqKeyInput = document.getElementById("groq-key-input");
const openaiKeyInput = document.getElementById("openai-key-input");
const clearApiKeysBtn = document.getElementById("clear-api-keys-btn");
const apiKeyBtnText = document.getElementById("api-key-btn-text");

const dropZone = document.getElementById("drop-zone");
const pdfInput = document.getElementById("pdf-input");
const selectFileBtn = document.getElementById("select-file-btn");

const uploadedInfo = document.getElementById("uploaded-info");
const metaFilename = document.getElementById("meta-filename");
const metaPages = document.getElementById("meta-pages");
const metaChunks = document.getElementById("meta-chunks");

const askForm = document.getElementById("ask-form");
const questionInput = document.getElementById("question-input");
const askBtn = document.getElementById("ask-btn");

const loadingIndicator = document.getElementById("loading-indicator");
const loadingText = document.getElementById("loading-text");

const resultsSection = document.getElementById("results-section");
const answerContent = document.getElementById("answer-content");
const latencyTag = document.getElementById("latency-tag");
const sourcesContainer = document.getElementById("sources-container");
const sourcesList = document.getElementById("sources-list");

// State
let isDocumentReady = false;

// Initialize Application
async function initApp() {
  loadSavedKeys();
  bindEvents();
  await checkBackendHealth();
}

function getCustomHeaders() {
  const headers = {};
  const gemini = localStorage.getItem("rag_gemini_key") || "";
  const groq = localStorage.getItem("rag_groq_key") || "";
  const openai = localStorage.getItem("rag_openai_key") || "";

  if (gemini.trim()) headers["X-Gemini-API-Key"] = gemini.trim();
  if (groq.trim()) headers["X-Groq-API-Key"] = groq.trim();
  if (openai.trim()) headers["X-OpenAI-API-Key"] = openai.trim();
  return headers;
}

function loadSavedKeys() {
  const gemini = localStorage.getItem("rag_gemini_key") || "";
  const groq = localStorage.getItem("rag_groq_key") || "";
  const openai = localStorage.getItem("rag_openai_key") || "";

  if (geminiKeyInput) geminiKeyInput.value = gemini;
  if (groqKeyInput) groqKeyInput.value = groq;
  if (openaiKeyInput) openaiKeyInput.value = openai;

  updateKeyButtonUI();
}

function updateKeyButtonUI() {
  const gemini = localStorage.getItem("rag_gemini_key");
  const groq = localStorage.getItem("rag_groq_key");
  const openai = localStorage.getItem("rag_openai_key");
  const hasLocalKey = Boolean(gemini || groq || openai);

  if (hasLocalKey) {
    openApiModalBtn.classList.add("configured");
    apiKeyBtnText.textContent = "API Keys (Set)";
  } else {
    openApiModalBtn.classList.remove("configured");
    apiKeyBtnText.textContent = "API Keys";
  }
}

// Event Bindings
function bindEvents() {
  selectFileBtn.addEventListener("click", () => pdfInput.click());
  pdfInput.addEventListener("change", handleFileSelect);

  // Modal handlers
  if (openApiModalBtn) openApiModalBtn.addEventListener("click", () => showApiModal(true));
  if (closeApiModalBtn) closeApiModalBtn.addEventListener("click", () => showApiModal(false));
  if (bannerConfigKeysBtn) bannerConfigKeysBtn.addEventListener("click", () => showApiModal(true));

  if (apiModal) {
    apiModal.addEventListener("click", (e) => {
      if (e.target === apiModal) showApiModal(false);
    });
  }

  if (apiKeyForm) {
    apiKeyForm.addEventListener("submit", (e) => {
      e.preventDefault();
      saveApiKeys();
    });
  }

  if (clearApiKeysBtn) {
    clearApiKeysBtn.addEventListener("click", clearApiKeys);
  }

  // Drag and drop handlers
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  askForm.addEventListener("submit", handleAskQuestion);
  dismissErrorBtn.addEventListener("click", hideError);
}

function showApiModal(show) {
  if (!apiModal) return;
  if (show) {
    loadSavedKeys();
    apiModal.classList.remove("hidden");
    if (geminiKeyInput) geminiKeyInput.focus();
  } else {
    apiModal.classList.add("hidden");
  }
}

function saveApiKeys() {
  const gemini = (geminiKeyInput.value || "").trim();
  const groq = (groqKeyInput.value || "").trim();
  const openai = (openaiKeyInput.value || "").trim();

  if (gemini) localStorage.setItem("rag_gemini_key", gemini);
  else localStorage.removeItem("rag_gemini_key");

  if (groq) localStorage.setItem("rag_groq_key", groq);
  else localStorage.removeItem("rag_groq_key");

  if (openai) localStorage.setItem("rag_openai_key", openai);
  else localStorage.removeItem("rag_openai_key");

  updateKeyButtonUI();
  showApiModal(false);
  if (keysWarningBanner) keysWarningBanner.classList.add("hidden");
  hideError();
}

function clearApiKeys() {
  localStorage.removeItem("rag_gemini_key");
  localStorage.removeItem("rag_groq_key");
  localStorage.removeItem("rag_openai_key");

  if (geminiKeyInput) geminiKeyInput.value = "";
  if (groqKeyInput) groqKeyInput.value = "";
  if (openaiKeyInput) openaiKeyInput.value = "";

  updateKeyButtonUI();
  showApiModal(false);
}

// Health Check
async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error("Health check failed");
    const data = await res.json();

    connectionStatus.className = "status-indicator online";
    connectionText.textContent = "Backend connected";

    const hasAnyKey = data.has_keys || Boolean(
      localStorage.getItem("rag_gemini_key") || 
      localStorage.getItem("rag_openai_key") ||
      localStorage.getItem("rag_groq_key")
    );

    if (!hasAnyKey && keysWarningBanner) {
      keysWarningBanner.classList.remove("hidden");
    } else if (keysWarningBanner) {
      keysWarningBanner.classList.add("hidden");
    }

    if (data.vector_store_ready && data.active_document) {
      setDocumentReady(data.active_document, data.total_pages, data.total_chunks);
    }
  } catch (err) {
    connectionStatus.className = "status-indicator offline";
    connectionText.textContent = "Backend offline (127.0.0.1:8000)";
  }
}

// File Selection Handler
function handleFileSelect(e) {
  const file = e.target.files[0];
  if (file) {
    handleFileUpload(file);
  }
}

// File Upload & Ingestion
async function handleFileUpload(file) {
  hideError();

  if (!file.name.toLowerCase().endsWith(".pdf")) {
    showError("Please select a valid PDF document (.pdf).");
    return;
  }

  // Set Processing State
  setLoading(true, "Processing PDF: extracting text, chunking, and generating embeddings...");
  disableQuestionForm();

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${API_BASE}/upload`, {
      method: "POST",
      headers: getCustomHeaders(),
      body: formData,
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Failed to process PDF.");
    }

    setDocumentReady(data.filename, data.total_pages, data.total_chunks);
  } catch (err) {
    const msg = err.message || "Failed to upload and process PDF.";
    showError(msg);
    if (msg.includes("API_KEY is not configured") || msg.includes("GEMINI_API_KEY")) {
      showApiModal(true);
    }
    isDocumentReady = false;
    disableQuestionForm();
  } finally {
    setLoading(false);
  }
}

// Set Ready State
function setDocumentReady(filename, pages, chunks) {
  isDocumentReady = true;
  metaFilename.textContent = filename;
  metaPages.textContent = `${pages} pages`;
  metaChunks.textContent = `${chunks} chunks indexed`;
  uploadedInfo.classList.remove("hidden");

  enableQuestionForm();
  questionInput.focus();
}

// Question Submission Handler
async function handleAskQuestion(e) {
  e.preventDefault();
  hideError();

  if (!isDocumentReady) {
    showError("Please upload a PDF document before asking questions.");
    return;
  }

  const question = questionInput.value.trim();
  if (!question) {
    showError("Please enter a question.");
    return;
  }

  setLoading(true, "Retrieving relevant text and generating answer...");
  resultsSection.classList.add("hidden");
  askBtn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers: { 
        "Content-Type": "application/json",
        ...getCustomHeaders(),
      },
      body: JSON.stringify({ question }),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Failed to generate answer.");
    }

    displayAnswer(data);
  } catch (err) {
    const msg = err.message || "An error occurred while answering your question.";
    showError(msg);
    if (msg.includes("API_KEY is not configured") || msg.includes("GEMINI_API_KEY") || msg.includes("GROQ_API_KEY")) {
      showApiModal(true);
    }
  } finally {
    setLoading(false);
    askBtn.disabled = false;
  }
}

// Display Answer & Citations
function displayAnswer(data) {
  answerContent.textContent = data.answer;
  latencyTag.textContent = `${data.latency_seconds}s`;

  // Render Citations
  sourcesList.innerHTML = "";
  if (data.sources && data.sources.length > 0) {
    data.sources.forEach((source) => {
      const li = document.createElement("li");
      li.className = "source-item";

      const pageHeading = document.createElement("span");
      pageHeading.className = "source-page";
      pageHeading.textContent = `Source: Page ${source.page_number}`;

      const excerpt = document.createElement("p");
      excerpt.className = "source-excerpt";
      excerpt.textContent = source.excerpt;

      li.appendChild(pageHeading);
      li.appendChild(excerpt);
      sourcesList.appendChild(li);
    });
    sourcesContainer.classList.remove("hidden");
  } else {
    sourcesContainer.classList.add("hidden");
  }

  resultsSection.classList.remove("hidden");
}

// UI Helpers
function setLoading(isLoading, message = "") {
  if (isLoading) {
    loadingText.textContent = message;
    loadingIndicator.classList.remove("hidden");
  } else {
    loadingIndicator.classList.add("hidden");
  }
}

function enableQuestionForm() {
  questionInput.disabled = false;
  askBtn.disabled = false;
}

function disableQuestionForm() {
  questionInput.disabled = true;
  askBtn.disabled = true;
}

function showError(msg) {
  errorMessage.textContent = msg;
  errorBanner.classList.remove("hidden");
}

function hideError() {
  errorBanner.classList.add("hidden");
  errorMessage.textContent = "";
}

// Start application
document.addEventListener("DOMContentLoaded", initApp);
