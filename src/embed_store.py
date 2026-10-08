"""
Day 2: Embeddings + Vector Store
---------------------------------
Turns every chunk from data/chunks.json into an embedding (a list of numbers
that captures the chunk's meaning) and stores them in a local Chroma database.
 
WHY embeddings: "attention mechanism" and "how does the model focus on
important words" share almost no words, but their embeddings land close
together. That is what lets us search by meaning instead of keyword match.
"""
 
import json
from pathlib import Path
 
import chromadb
from sentence_transformers import SentenceTransformer
 
CHUNKS_FILE = Path("data/chunks.json")
DB_DIR = "chroma_db"
COLLECTION_NAME = "papers"
MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast, free, runs on CPU
BATCH_SIZE = 64
 
 
def main():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError("data/chunks.json not found. Run src/ingest.py first.")
 
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")
 
    print(f"Loading embedding model '{MODEL_NAME}' (first run downloads ~90MB)...")
    model = SentenceTransformer(MODEL_NAME)
 
    client = chromadb.PersistentClient(path=DB_DIR)
 
    # Rebuild from scratch each run so re-running never creates duplicates
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # compare vectors by cosine similarity
    )
 
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, normalize_embeddings=True).tolist()
 
        collection.add(
            ids=[f"chunk_{start + i}" for i in range(len(batch))],
            documents=texts,
            embeddings=embeddings,
            metadatas=[{"source": c["source"]} for c in batch],
        )
        print(f"  stored {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)}")
 
    print(f"\nDone. {collection.count()} chunks stored in '{DB_DIR}/'")
 
 
if __name__ == "__main__":
    main()