"""
Day 4: Streamlit UI
---------------------
A simple chat interface for the RAG pipeline. This is the part people
actually see in a demo video or live link -- it turns the backend
scripts into something you can show off.
 
Run:
    streamlit run src/app.py
"""
 
import streamlit as st
 
from rag_chain import ask_with_sources
 
st.set_page_config(page_title="RAG Paper Chat", page_icon="📄")
 
st.title("📄 RAG Paper Chat")
st.caption(
    "Ask questions about classic ML papers (Attention, BERT, ResNet, GPT-3, LoRA). "
    "Answers are grounded only in the paper text -- no made-up outside knowledge."
)
 
if "messages" not in st.session_state:
    st.session_state.messages = []
 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for s in message["sources"]:
                    st.markdown(f"**{s['source']}** · score {s['score']}")
                    st.caption(s["text"][:300] + "...")
 
question = st.chat_input("Ask something about the papers...")
 
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
 
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, chunks = ask_with_sources(question)
        st.markdown(answer)
        with st.expander("Sources"):
            for s in chunks:
                st.markdown(f"**{s['source']}** · score {s['score']}")
                st.caption(s["text"][:300] + "...")
 
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": chunks,
    })