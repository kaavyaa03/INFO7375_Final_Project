import streamlit as st
import requests

API = "http://127.0.0.1:8000"

st.title("PaperPulse RAG Demo")

if st.button("Build Knowledge Base"):
    r = requests.post(f"{API}/ingest", json={
        "input_dir": "data/kb_samples",
        "chunk_size": 900,
        "chunk_overlap": 180
    })
    st.json(r.json())

question = st.text_area("Ask a question", "What is RAG and why is it useful?")
mode = st.selectbox("Mode", ["ask", "summarize"])

if st.button("Run"):
    r = requests.post(f"{API}/query", json={
        "question": question,
        "mode": mode,
        "top_k": 8,
        "similarity_threshold": 0.25
    })
    st.write(r.json()["answer"])
    st.json(r.json()["citations"])
    st.json(r.json()["metrics"])
