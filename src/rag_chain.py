"""
Day 3: RAG Chain
-----------------
Wires retrieval to a local LLM (via Ollama) so questions get answered
using ONLY the retrieved paper chunks -- not the model's own (possibly
wrong or outdated) memory. This "grounding" is the whole point of RAG.
 
Run interactively:
    python src/rag_chain.py
"""
import os
from groq import Groq
from dotenv import load_dotenv
from retrieve import retrieve

# Load environment variables (for local testing)
load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"
TOP_K = 8

SYSTEM_PROMPT = (
    "You are a research assistant that answers questions about machine "
    "learning papers using ONLY the context provided below. Do not use "
    "any outside knowledge, even if you think you know the answer. "
    "If the context does not contain the answer, say "
    "\"I don't have enough information in these papers to answer that.\" "
    "Always mention which paper (source) your answer comes from."
)

# Groq Client setup
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    # Streamlit Cloud ke liye (later hum isko handle karenge)
    import streamlit as st
    api_key = st.secrets.get("GROQ_API_KEY")

client = Groq(api_key=api_key)

def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

def ask_with_sources(question: str) -> tuple[str, list[dict]]:
    chunks = retrieve(question, k=TOP_K)
    prompt = build_prompt(question, chunks)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content, chunks

def ask(question: str) -> str:
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

