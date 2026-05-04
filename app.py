"""
app.py — Streamlit chat UI for the Wikipedia RAG Assistant.

Features:
- Chat-style Q&A interface
- Shows retrieved source chunks (expandable)
- Query type indicator (person / place / both)
- Clear/reset button
- Streaming response support
"""

from __future__ import annotations

import streamlit as st
import sqlite3
from pathlib import Path

from retriever import retrieve, classify_query
from generator import generate_stream

DB_PATH = Path(__file__).parent / "data" / "wiki.db"


@st.cache_data
def entity_urls() -> dict[str, str]:
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute("SELECT name, wiki_url FROM entities").fetchall()
    conn.close()
    return {name: url for name, url in rows}

st.set_page_config(
    page_title="Wikipedia RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("Wikipedia RAG Assistant")
st.caption("Ask anything about famous people and places. Powered by local Ollama + ChromaDB.")

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_sources" not in st.session_state:
    st.session_state.show_sources = True

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    st.session_state.show_sources = st.toggle("Show source chunks", value=True)
    n_results = st.slider("Retrieved chunks", min_value=2, max_value=10, value=5)

    st.divider()
    st.markdown("**Example questions:**")
    examples = [
        "Who was Albert Einstein?",
        "What did Marie Curie discover?",
        "Where is the Eiffel Tower located?",
        "What was the Colosseum used for?",
        "Compare Lionel Messi and Cristiano Ronaldo",
        "Which person is associated with electricity?",
        "Compare the Eiffel Tower and the Statue of Liberty",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending_query = ex

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Display chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and st.session_state.show_sources and msg.get("chunks"):
            with st.expander(f"Sources ({len(msg['chunks'])} chunks)"):
                for i, chunk in enumerate(msg["chunks"], 1):
                    url = entity_urls().get(chunk['entity'], "")
                    link = f"[{chunk['entity']}]({url})" if url else chunk['entity']
                    st.markdown(f"**[{i}] {link}** ({chunk['type']}) — score: `{chunk['score']}`")
                    st.text(chunk["text"][:400] + ("..." if len(chunk["text"]) > 400 else ""))
                    st.divider()

# --- Handle input ---
pending = st.session_state.pop("pending_query", None)
user_input = st.chat_input("Ask about a famous person or place...") or pending

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        qtype = classify_query(user_input)
        st.caption(f"Query type detected: **{qtype}**")

        chunks = retrieve(user_input, n_results=n_results, query_type=qtype)

        if not chunks:
            answer = "I don't know based on the available data."
            st.markdown(answer)
        else:
            # Stream the response
            placeholder = st.empty()
            full_answer = ""
            for token in generate_stream(user_input, chunks):
                full_answer += token
                placeholder.markdown(full_answer + "▌")
            placeholder.markdown(full_answer)
            answer = full_answer

            if st.session_state.show_sources:
                with st.expander(f"Sources ({len(chunks)} chunks)"):
                    for i, chunk in enumerate(chunks, 1):
                        url = entity_urls().get(chunk['entity'], "")
                        link = f"[{chunk['entity']}]({url})" if url else chunk['entity']
                        st.markdown(f"**[{i}] {link}** ({chunk['type']}) — score: `{chunk['score']}`")
                        st.text(chunk["text"][:400] + ("..." if len(chunk["text"]) > 400 else ""))
                        st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "chunks": chunks if chunks else [],
    })
