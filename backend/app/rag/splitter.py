import uuid
from typing import List, Dict, Any
from app.models.schemas import DocumentChunk


class TextSplitter:
    """
    Splits document pages into smaller overlapping chunks while preserving page metadata.
    Uses recursive splitting across natural boundaries: paragraph, newline, sentence, and word.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text by separators until pieces fit within chunk_size."""
        final_chunks: List[str] = []
        separator = separators[-1]
        remaining_separators: List[str] = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                remaining_separators = []
                break
            if sep in text:
                separator = sep
                remaining_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator != "" else list(text)

        current_piece: List[str] = []
        current_len = 0

        for piece in splits:
            piece_len = len(piece) + (len(separator) if current_piece else 0)
            if current_len + piece_len <= self.chunk_size:
                current_piece.append(piece)
                current_len += piece_len
            else:
                if current_piece:
                    merged = separator.join(current_piece)
                    if len(merged) > self.chunk_size and remaining_separators:
                        final_chunks.extend(self._split_text_recursive(merged, remaining_separators))
                    else:
                        final_chunks.append(merged)
                current_piece = [piece]
                current_len = len(piece)

        if current_piece:
            merged = separator.join(current_piece)
            if len(merged) > self.chunk_size and remaining_separators:
                final_chunks.extend(self._split_text_recursive(merged, remaining_separators))
            else:
                final_chunks.append(merged)

        return final_chunks

    def _add_overlap(self, pieces: List[str]) -> List[str]:
        """Combine pieces with specified character overlap."""
        if not pieces:
            return []

        chunks: List[str] = []
        for i, piece in enumerate(pieces):
            chunk = piece.strip()
            if not chunk:
                continue

            # If not first chunk and overlap is configured, prepend suffix of previous chunk
            if i > 0 and self.chunk_overlap > 0:
                prev_text = pieces[i - 1].strip()
                overlap_text = prev_text[-self.chunk_overlap:]
                # Try to align to word boundary if possible
                first_space = overlap_text.find(" ")
                if first_space != -1 and first_space < len(overlap_text) - 1:
                    overlap_text = overlap_text[first_space + 1:]
                chunk = f"{overlap_text} {chunk}".strip()

            chunks.append(chunk)

        return chunks

    def split_pages(self, pages_data: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """
        Split each page into chunks while strictly attaching page_number and source_file.
        """
        all_chunks: List[DocumentChunk] = []

        for page in pages_data:
            page_text = page["text"]
            page_num = page["page_number"]
            source_file = page["source_file"]

            raw_pieces = self._split_text_recursive(page_text, self.separators)
            overlapped_pieces = self._add_overlap(raw_pieces)

            for chunk_idx, text_content in enumerate(overlapped_pieces):
                clean_text = text_content.strip()
                if not clean_text:
                    continue

                chunk_id = f"p{page_num}_c{chunk_idx}_{uuid.uuid4().hex[:6]}"
                all_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=clean_text,
                        page_number=page_num,
                        source_file=source_file,
                    )
                )

        return all_chunks
