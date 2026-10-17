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
import os
from rag_chain import ask_with_sources

st.set_page_config(page_title="RAG Paper Chat", page_icon="📚")
st.title("📚 RAG Paper Chat")
st.caption("Ask questions about ML papers. Answers are grounded in the provided PDFs.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("Sources"):
                for src in message["sources"]:
                    st.write(f"- {src}")

# Accept user input
if prompt := st.chat_input("Ask something about the papers..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, chunks = ask_with_sources(prompt)
            st.markdown(answer)
            
            sources = list(set([c['source'] for c in chunks]))
            with st.expander("Sources"):
                for src in sources:
                    st.write(f"- {src}")
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})