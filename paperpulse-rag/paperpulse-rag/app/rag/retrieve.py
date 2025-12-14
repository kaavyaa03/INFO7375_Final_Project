import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedder():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_vectorstore(chunks, persist_dir: str):
    os.makedirs(persist_dir, exist_ok=True)
    vs = FAISS.from_documents(chunks, get_embedder())
    vs.save_local(persist_dir)
    return vs

def load_vectorstore(persist_dir: str):
    return FAISS.load_local(persist_dir, get_embedder(), allow_dangerous_deserialization=True)

def retrieve(vs, query: str, top_k: int, threshold: float):
    docs_scores = vs.similarity_search_with_score(query, k=top_k)
    kept = []
    for doc, score in docs_scores:
        similarity = 1 / (1 + score)
        if similarity >= threshold:
            kept.append(doc)
    return kept
