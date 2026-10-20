"""
Day 1: Document Ingestion & Chunking
--------------------------------------
Loads PDFs from data/papers/, extracts their text, and splits it into
overlapping chunks small enough to embed and retrieve accurately.
 
WHY chunking matters: embedding models and LLMs work best on focused
pieces of text. Embed a whole 10-page paper as one vector and retrieval
becomes useless -- it can't tell which part is relevant to a question.
Overlapping chunks keep each vector focused on one idea while preserving
context across chunk boundaries.
"""
 
import json
from pathlib import Path

from pypdf import PdfReader

DATA_DIR = Path("data/papers")
OUTPUT_FILE = Path("data/chunks.json")

CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 150   # characters shared between consecutive chunks

def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text

def chunk_text(text: str, source: str) -> list[dict]:
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + CHUNK_SIZE
        piece = text[start:end].strip()
        if piece:
            chunks.append({"text": piece, "source": source})
        start += CHUNK_SIZE - CHUNK_OVERLAP  # slide forward, keep overlap

    return chunks

def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"'{DATA_DIR}/' doesn't exist yet. Create it and add your PDFs first."
        )

    # Magic Fix: Reads both lowercase and uppercase extensions
    pdf_files = sorted(list(DATA_DIR.glob("*.pdf")) + list(DATA_DIR.glob("*.PDF")))
    
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {DATA_DIR}/. Add some (see README) and rerun."
        )

    all_chunks = []
    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.name}...")
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text, source=pdf_path.name)
        all_chunks.extend(chunks)
        print(f"  -> {len(chunks)} chunks")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\nDone. {len(all_chunks)} total chunks saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()