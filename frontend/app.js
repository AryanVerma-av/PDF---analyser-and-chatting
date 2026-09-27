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
  bindEvents();
  await checkBackendHealth();
}

// Event Bindings
function bindEvents() {
  selectFileBtn.addEventListener("click", () => pdfInput.click());
  pdfInput.addEventListener("change", handleFileSelect);

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

// Health Check
async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error("Health check failed");
    const data = await res.json();

    connectionStatus.className = "status-indicator online";
    connectionText.textContent = "Backend connected";

    if (data.vector_store_ready && data.active_document) {
      setDocumentReady(data.active_document, data.total_pages, data.total_chunks);
    }
  } catch (err) {
    connectionStatus.className = "status-indicator offline";
    connectionText.textContent = "Backend offline";
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
      body: formData,
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Failed to process PDF.");
    }

    setDocumentReady(data.filename, data.total_pages, data.total_chunks);
  } catch (err) {
    showError(err.message || "Failed to upload and process PDF.");
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
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Failed to generate answer.");
    }

    displayAnswer(data);
  } catch (err) {
    showError(err.message || "An error occurred while answering your question.");
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
