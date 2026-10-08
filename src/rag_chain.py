"""
Day 3: RAG Chain
-----------------
Wires retrieval to a local LLM (via Ollama) so questions get answered
using ONLY the retrieved paper chunks -- not the model's own (possibly
wrong or outdated) memory. This "grounding" is the whole point of RAG.
 
Run interactively:
    python src/rag_chain.py
"""
 
import ollama
 
from retrieve import retrieve
 
MODEL_NAME = "llama3.2:3b"
TOP_K = 8
 
SYSTEM_PROMPT = (
    "You are a research assistant. Answer the question using ONLY the provided context. "
    "If the context has partial information, summarize it. "
    "If the context does not contain the answer, say 'I don't have enough information.' "
    "Do NOT try to compare things unless the context explicitly compares them. "
    "Always cite the source paper."
)
 
 
def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
 
 
def ask_with_sources(question: str) -> tuple[str, list[dict]]:
    chunks = retrieve(question, k=TOP_K)
    prompt = build_prompt(question, chunks)
 
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response["message"]["content"], chunks
 
 
def ask(question: str) -> str:
    """Convenience wrapper for the CLI script -- answer text only."""
    answer, _ = ask_with_sources(question)
    return answer
 
 
if __name__ == "__main__":
    print(f"RAG Paper Chat -- using {MODEL_NAME} (type 'quit' to exit)\n")
    while True:
        question = input("Ask a question: ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue
        print("\nThinking...\n")
        print(ask(question))
        print("-" * 60)

