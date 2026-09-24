import io
from typing import List, Dict, Any
from pypdf import PdfReader


class PDFLoaderError(Exception):
    """Exception raised when PDF loading or extraction fails."""
    pass


class PDFLoader:
    """Loads and extracts text page-by-page from PDF files."""

    @staticmethod
    def load_from_bytes(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Extract text from PDF byte content page by page.
        
        Returns a list of dicts with 'text', 'page_number' (1-indexed), and 'source_file'.
        Raises PDFLoaderError if extraction fails or document contains no extractable text.
        """
        if not file_bytes:
            raise PDFLoaderError("The uploaded file is empty.")

        try:
            stream = io.BytesIO(file_bytes)
            reader = PdfReader(stream)
        except Exception as exc:
            raise PDFLoaderError(f"Failed to parse PDF format: {str(exc)}") from exc

        total_pages = len(reader.pages)
        if total_pages == 0:
            raise PDFLoaderError("The PDF document contains 0 pages.")

        pages_data: List[Dict[str, Any]] = []
        total_extracted_chars = 0

        for page_index, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
                # Normalize line breaks and multiple spaces
                cleaned_text = page_text.strip()
                if cleaned_text:
                    total_extracted_chars += len(cleaned_text)
                    pages_data.append({
                        "text": cleaned_text,
                        "page_number": page_index + 1,
                        "source_file": filename,
                    })
            except Exception as exc:
                # Log or handle individual page extraction error
                continue

        if total_extracted_chars == 0:
            raise PDFLoaderError(
                "No readable text found in PDF. The document may be scanned or image-only."
            )

        return pages_data
