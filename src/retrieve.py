"""
Day 2: Retrieval
-----------------
Given a question, find the most relevant chunks in the vector store.
Day 3 will import retrieve() and hand these chunks to the local LLM.
 
Test it from the terminal:
    python src/retrieve.py "your question here"
"""
 
import sys
import os
import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = "chroma_db"
COLLECTION_NAME = "papers"
MODEL_NAME = "all-MiniLM-L6-v2"   # must match the model used in embed_store.py

_model = None
_collection = None

def _load():
    """Load the model and database once. If database is empty, build it."""
    global _model, _collection
    if _model is None:
        print("Loading embedding model...")
        _model = SentenceTransformer(MODEL_NAME)
        client = chromadb.PersistentClient(path=DB_DIR)
        
        # Try to get collection, if it fails, create it
        try:
            _collection = client.get_collection(COLLECTION_NAME)
        except Exception:
            _collection = client.create_collection(COLLECTION_NAME)
            
        # MAGIC FIX: If database is empty (like on Streamlit Cloud), build it!
        if _collection.count() == 0:
            print("Database is empty. Building it now. First time me 2-3 minute lag sakte hain...")
            sys.path.append(os.path.dirname(__file__))
            try:
                from ingest import main as run_ingest
                from embed_store import main as run_embed
                
                # Run the ingestion and embedding steps
                run_ingest()
                run_embed()
                
                # Reload the collection after building
                _collection = client.get_collection(COLLECTION_NAME)
                print(f"Database built successfully with {_collection.count()} chunks.")
            except Exception as e:
                print(f"Error building database: {e}")
                raise e
                
    return _model, _collection

def retrieve(query: str, k: int = 8) -> list[dict]:
    model, collection = _load()
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()
    result = collection.query(query_embeddings=query_embedding, n_results=k)
 
    chunks = []
    for text, meta, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        chunks.append({
            "text": text,
            "source": meta["source"],
            "score": round(1 - distance, 3),   # higher = more similar
        })
    return chunks
 
if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is the main idea of these papers?"
    print(f"Question: {question}\n")
    for i, chunk in enumerate(retrieve(question), start=1):
        print(f"#{i}  score={chunk['score']}  source={chunk['source']}")
        print(chunk["text"][:400].replace("\n", " "))
        print("-" * 60)