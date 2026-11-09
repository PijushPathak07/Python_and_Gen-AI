import io
import re
from typing import Tuple

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


def normalize_transcript_text(raw_text: str) -> str:
    """Clean control characters, excessive line breaks, and page artifacts."""
    if not raw_text:
        return ""
    cleaned = re.sub(r"\r\n|\r", "\n", raw_text)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_text_from_upload(
    file_bytes: bytes, filename: str
) -> Tuple[bool, str]:
    """Extract normalized text from an uploaded .pdf or .txt file."""
    lower_name = (filename or "").lower()

    if lower_name.endswith(".txt"):
        try:
            decoded = file_bytes.decode("utf-8", errors="replace")
            return True, normalize_transcript_text(decoded)
        except Exception as exc:
            return False, f"Failed to decode text file: {str(exc)}"

    if lower_name.endswith(".pdf"):
        if PdfReader is None:
            return False, "PDF parsing requires 'pypdf'. Install with: pip install pypdf"
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            pages = [page.extract_text() or "" for page in reader.pages]
            combined = "\n".join(pages)
            normalized = normalize_transcript_text(combined)
            if not normalized:
                return False, "No readable text found inside the uploaded PDF."
            return True, normalized
        except Exception as exc:
            return False, f"Error reading PDF file: {str(exc)}"

    return False, "Unsupported file format. Please upload a .pdf or .txt file."